"""Bind the audited SSD4Rec BiSSD blocks to IP&M domain item embeddings."""
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import sys

import torch
from torch import nn

SOURCE_ROOT = Path(__file__).resolve().parents[4]
VENDOR = SOURCE_ROOT / "RCCT/DuoGap"
OFFICIAL = VENDOR / "vendor/ssd4rec_official"
PINNED = {
    VENDOR / "scripts/paper_ssd4rec_rcct_adapter.py": "72ff63818b8b25dadd34affe57b4d4f9882195fcea6a34cb05d40ec4da90f362",
    VENDOR / "scripts/rcct_carrier_contract.py": "6e33236ad4894a996c33ab286f5c736592b36dafbe48d4b30a4a967abb8fab22",
    OFFICIAL / "PROVENANCE.json": "83fe7d1786139b55471fc80c84b34777cb86dcbdcafb02c7ee025322136c11f5",
    OFFICIAL / "ssd4rec.py": "2b31ea7f5418391b49e3c76776d1c0dae75905729f65ddef4c8242ba2f1a43d1",
}


def load_factory(device):
    for path, expected in PINNED.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"SSD4Rec source differs from the audited snapshot: {path}")
    sys.path.insert(0, str(VENDOR / "scripts"))
    from paper_ssd4rec_rcct_adapter import SSD4RecCarrier, official_ssd4rec_block_factory
    if device.type == "cpu":
        # shortcut: differentiable CPU fixture; CUDA always uses pinned BiSSD.
        block = FixtureBlock
        runtime = {"kind": "cpu_fixture", "torch": str(torch.__version__)}
    else:
        pins = json.loads((OFFICIAL / "PROVENANCE.json").read_text())["formal_dependency_contract"]
        runtime = {"python": platform.python_version(), "torch": str(torch.__version__)}
        for name in pins.keys() - runtime.keys():
            try:
                runtime[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError as exc:
                raise RuntimeError(f"Missing SSD4Rec dependency {name}=={pins[name]}") from exc
        mismatches = {k: [v, runtime[k]] for k, v in pins.items() if runtime[k] != v}
        if mismatches:
            raise RuntimeError(f"SSD4Rec runtime mismatch (expected / observed): {mismatches}")
        spec = importlib.util.spec_from_file_location("ipm_pinned_ssd4rec", OFFICIAL / "ssd4rec.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        block = official_ssd4rec_block_factory(module)
        runtime.update(kind="official_cuda", cuda=torch.version.cuda)
    return block, SSD4RecCarrier._packed_registers, runtime


class SharedSSD4Rec(nn.Module):
    """Use the observed prefix's final state without adding a second item head."""

    def __init__(self, width, length, settings, factory, registers):
        super().__init__()
        self.length = length
        self.registers = registers
        self.blocks = nn.ModuleList([
            factory(settings["beta"], width, settings["d_state"], settings["d_conv"],
                    settings["expand"], 0.0, settings["layers"], settings["headdim"])
            for _ in range(settings["layers"])
        ])
        for block in self.blocks:
            block.apply(self._init)

    @staticmethod
    def _init(layer):
        if isinstance(layer, nn.Linear):
            nn.init.normal_(layer.weight, mean=0.0, std=0.02)
            if layer.bias is not None and not getattr(layer.bias, "_no_reinit", False):
                nn.init.zeros_(layer.bias)

    def forward(self, embeddings, padding_mask):
        if embeddings.shape[:2] != padding_mask.shape or padding_mask.shape[1] > self.length:
            raise ValueError("Invalid SSD4Rec embedding or padding shape")
        valid = ~padding_mask
        registers, flip = self.registers(valid)
        packed = embeddings[valid].unsqueeze(0)
        for block in self.blocks:
            packed = block(packed, registers.unsqueeze(0), flip)
        return packed[0, valid.sum(dim=1).cumsum(dim=0) - 1]


class FixtureBlock(nn.Module):
    def __init__(self, beta, width, *_):
        super().__init__()
        self.beta = beta
        self.projection = nn.Linear(width, width)

    def forward(self, packed, registers, flip):
        _, lengths = torch.unique_consecutive(registers[0], return_counts=True)
        pieces = torch.split(packed[0, flip], lengths.tolist())
        backward = torch.cat([x.cumsum(0).flip(0) for x in pieces])
        return packed + self.beta * self.projection(backward).unsqueeze(0)
