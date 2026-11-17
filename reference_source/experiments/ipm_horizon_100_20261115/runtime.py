"""Load the unchanged model and checked arrays for an admitted job."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

from paths import (BACKBONE_SRC, CORE, GROUPS, INPUTS_PATH, PLAN, PLAN_PATH, YELP,
                   development, parent_family, read, require, sha)

DEVICE = torch.device("cuda")


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def configure(group):
    sys.path.insert(0, str(CORE))
    import train
    require(train.code_fingerprint() == PLAN["core_sha256"], "Archived shared trainer changed")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Expected one admitted CUDA card")
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    recipe = GROUPS[group]
    backbone = recipe["backbone"]
    if backbone == "sasrec":
        torch.backends.mha.set_fastpath_enabled(False)
        metadata = {"source": train.code_fingerprint(), "torch": str(torch.__version__)}
    else:
        sys.path.insert(0, str(BACKBONE_SRC))
        original = module("ipm_pinned_backbone", BACKBONE_SRC / "run.py")
        metadata = {"source": original.configure(backbone, DEVICE), "runtime": original.RUNTIME}
    train.TARGET = recipe["target"]
    return train, metadata


def rows_for(group, phase, domains=None, small=False):
    recipe = GROUPS[group]
    source = recipe["source"]
    sealed = phase == "test"
    root = (CORE if source == "amazon" else YELP) / "data/test_sealed" if sealed else development(source)
    manifest_path = root / "manifest.json"
    digest = sha(manifest_path)
    require(digest == PLAN["data"][source]["test" if sealed else "development"], "Chronological manifest changed")
    manifest = read(manifest_path)
    windows = ("train", "validation") if phase == "fit" else (phase,)
    rows = {window: {} for window in windows}
    counts = {name: entry["item_count"] for name, entry in manifest["domains"].items()}
    for domain, entry in manifest["domains"].items():
        if domains is not None and domain not in domains:
            continue
        for window in windows:
            record = entry["windows"][window]
            path = root / record["file"]
            require(sha(path) == record["sha256"], f"Changed arrays: {path}")
            with np.load(path, allow_pickle=False) as archive:
                labels = archive["target"]
                require(len(labels) == record["rows"], f"Wrong request count: {path}")
                if small and window == "validation":
                    idx = np.flatnonzero((archive["gap"] > 365) & (labels >= 0))[:128]
                else:
                    idx = slice(0, 512 if small else None)
                keys = ("seq", "target", "gap") if window == "train" else ("seq", "target", "gap", "user", "untied")
                batch = {key: torch.from_numpy(archive[key][idx].copy() if small else archive[key])
                         for key in keys}
            if source == "yelp":
                batch["seq"] = batch["seq"].to(torch.int32)
            rows[window][domain] = batch
    return counts, rows, manifest["cutoff_days_inclusive"][0], digest


def verified_file(entry):
    path = Path(entry["path"])
    require(sha(path) == entry["sha256"], f"Changed artifact: {path}")
    return path


def inventory():
    value = read(INPUTS_PATH)
    require(value["plan_sha256"] == sha(PLAN_PATH), "Input inventory is for a different plan")
    return value


def parent_path(group, seed):
    row = inventory()["parents"][parent_family(group)][str(seed)]
    weights = row["weights"]
    if row.get("new"):
        meta = read(Path(row["fit"]["path"]))
        require(meta["contract"]["seed"] == seed, "New parent seed differs")
        weights = {**weights, "sha256": meta["checkpoint_sha"]}
    return verified_file(weights), weights["sha256"]
