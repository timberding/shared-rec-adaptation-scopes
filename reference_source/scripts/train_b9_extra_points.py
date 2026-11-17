#!/usr/bin/env python3
"""Train same-protocol B9 extra points for the shared-backbone tradeoff plot."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_shared_backbone_seed42 import (  # noqa: E402
    TimeConcatTargetModel,
    capped_samples,
    evaluate_all,
    load_json,
    load_processed,
    max_abs_drift,
    metric_delta,
    state_snapshot,
    tensorize_samples,
    train_epoch,
    train_target_model,
)
from src.shared_backbone import (  # noqa: E402
    DomainItemHead,
    SharedBackboneDeltaGateAdapter,
    SharedSASRecEncoder,
    freeze_module,
    trainable_parameter_report,
)


class HeadOnlyTargetModel(nn.Module):
    """Frozen encoder plus trainable target head."""

    def __init__(self, encoder: SharedSASRecEncoder, target_head: DomainItemHead) -> None:
        super().__init__()
        self.encoder = encoder
        self.target_head = target_head

    def forward(self, seq: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        del delta_t
        padding_mask = seq.eq(0)
        hidden = self.encoder(self.target_head.embed_sequence(seq), padding_mask)
        return self.target_head.logits(hidden)


class GapOnlyDeltaGateAdapter(nn.Module):
    """DeltaGate ablation whose gate observes only the macro-gap scalar."""

    def __init__(
        self,
        encoder: SharedSASRecEncoder,
        target_head: DomainItemHead,
        embedding_dim: int,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.target_head = target_head
        self.hidden_dim = hidden_dim or embedding_dim * 2
        self.gate_mlp = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, embedding_dim),
            nn.Sigmoid(),
        )
        self.global_prior = nn.Parameter(torch.zeros(embedding_dim))
        nn.init.zeros_(self.global_prior)

    def forward(self, seq: torch.Tensor, delta_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        padding_mask = seq.eq(0)
        hidden = self.encoder(self.target_head.embed_sequence(seq), padding_mask)
        gate = self.gate_mlp(torch.log1p(delta_t).unsqueeze(-1))
        final = gate * hidden + (1.0 - gate) * self.global_prior.unsqueeze(0)
        return self.target_head.logits(final), gate.mean(dim=-1)


class FrozenTimeAdapterTargetModel(nn.Module):
    """Frozen encoder/head plus trainable time projection on the target representation."""

    def __init__(self, encoder: SharedSASRecEncoder, target_head: DomainItemHead, embedding_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.target_head = target_head
        self.time_projection = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, seq: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        padding_mask = seq.eq(0)
        hidden = self.encoder(self.target_head.embed_sequence(seq), padding_mask)
        hidden = hidden + self.time_projection(torch.log1p(delta_t).unsqueeze(-1))
        return self.target_head.logits(hidden)


def deltagate_trainable_params(embedding_dim: int) -> int:
    # gate_mlp: (1+d)->2d plus 2d->d, plus global_prior.
    return 4 * embedding_dim * embedding_dim + 6 * embedding_dim


def time_adapter_trainable_params(embedding_dim: int, hidden_dim: int) -> int:
    # time_projection: 1->h plus h->d, including biases.
    return hidden_dim * (embedding_dim + 2) + embedding_dim


def gap_only_trainable_params(embedding_dim: int, hidden_dim: int) -> int:
    # gate_mlp: 1->h plus h->d, including biases, plus global_prior.
    return hidden_dim * (embedding_dim + 2) + 2 * embedding_dim


def capacity_matched_hidden_dim(embedding_dim: int, target_params: int) -> int:
    return max(1, round((target_params - embedding_dim) / (embedding_dim + 2)))


def capacity_matched_gap_only_hidden_dim(embedding_dim: int, target_params: int) -> int:
    return max(1, round((target_params - 2 * embedding_dim) / (embedding_dim + 2)))


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def fmt_pp(value: float) -> str:
    return f"{100.0 * value:+.2f} pp"


def load_data(config: dict[str, Any], args: argparse.Namespace, rng: random.Random, device: torch.device):
    dataset = config["dataset"]
    processed_root = Path(dataset["processed_root"])
    target_domain = dataset["target_domain"]
    target_bucket = dataset["domains"][target_domain].get("primary_bucket", ">365d")
    loaded = {}
    train_samples = {}
    eval_samples = {}
    domain_summary = {}
    for domain, domain_config in sorted(dataset["domains"].items()):
        data = load_processed(processed_root / domain_config["processed_file"])
        loaded[domain] = data
        train_bucket = target_bucket if domain == target_domain and args.prioritize_target_bucket else None
        selected_train = capped_samples(data["train_samples"], args.max_train_samples_per_domain, rng, train_bucket)
        selected_eval = capped_samples(data["test_samples"], args.max_eval_samples_per_domain, rng)
        train_samples[domain] = tensorize_samples(
            selected_train,
            args.max_seq_len,
            pin_memory=args.pin_memory and device.type == "cuda",
        )
        eval_samples[domain] = tensorize_samples(
            selected_eval,
            args.max_seq_len,
            pin_memory=args.pin_memory and device.type == "cuda",
        )
        domain_summary[domain] = {
            "item_num": int(data["item_vocab_size"]),
            "train_samples": len(train_samples[domain]),
            "eval_samples": len(eval_samples[domain]),
        }
    return loaded, train_samples, eval_samples, domain_summary, target_domain, target_bucket


def train_m0(loaded: dict[str, Any], train_samples, eval_samples, target_domain: str, target_bucket: str, args, device):
    encoder = SharedSASRecEncoder(
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
    ).to(device)
    heads = {
        domain: DomainItemHead(int(data["item_vocab_size"]), embedding_dim=args.embedding_dim).to(device)
        for domain, data in loaded.items()
    }
    checkpoint_source = None
    def load_encoder_heads(checkpoint_path: Path) -> dict[str, Any]:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if not isinstance(checkpoint, dict) or "encoder" not in checkpoint or "heads" not in checkpoint:
            raise TypeError(f"{checkpoint_path} must contain encoder and heads state dicts")
        checkpoint_heads = checkpoint["heads"]
        missing_heads = sorted(set(heads) - set(checkpoint_heads))
        extra_heads = sorted(set(checkpoint_heads) - set(heads))
        if missing_heads or extra_heads:
            raise ValueError(
                f"{checkpoint_path} head domains do not match protocol: "
                f"missing={missing_heads}, extra={extra_heads}"
            )
        encoder.load_state_dict(checkpoint["encoder"])
        for domain, head in heads.items():
            head.load_state_dict(checkpoint_heads[domain])
        return checkpoint

    if args.load_m0_checkpoint:
        checkpoint_path = Path(args.load_m0_checkpoint)
        load_encoder_heads(checkpoint_path)
        losses: list[float] = []
        checkpoint_source = str(checkpoint_path)
        encoder.eval()
        for head in heads.values():
            head.eval()
        metrics = evaluate_all(
            eval_samples,
            encoder,
            heads,
            args.eval_batch_size,
            device,
            target_domain,
            target_bucket=target_bucket,
            amp=args.amp,
            amp_dtype=args.amp_dtype,
        )
        return encoder, heads, losses, metrics, checkpoint_source

    optimizer = torch.optim.AdamW(
        [parameter for module in [encoder, *heads.values()] for parameter in module.parameters() if parameter.requires_grad],
        lr=args.lr,
    )
    losses = []
    start_epoch = 0
    if args.resume_m0_checkpoint:
        checkpoint_path = Path(args.resume_m0_checkpoint)
        checkpoint = load_encoder_heads(checkpoint_path)
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch_completed", 0))
        losses = [float(loss) for loss in checkpoint.get("losses", [])]
        if start_epoch < 0 or start_epoch > args.m0_epochs:
            raise ValueError(
                f"{checkpoint_path} has invalid epoch_completed={start_epoch} "
                f"for requested m0_epochs={args.m0_epochs}"
            )
        print(
            "PROGRESS "
            f"phase=b9_m0_resume epoch_completed={start_epoch}/{args.m0_epochs} "
            f"source={checkpoint_path}",
            flush=True,
        )
    started = time.time()
    for epoch in range(start_epoch, args.m0_epochs):
        epoch_losses = train_epoch(
            train_samples,
            encoder,
            heads,
            optimizer,
            args.batch_size,
            device,
            args.amp,
            args.amp_dtype,
            args.m0_domain_schedule,
        )
        losses.extend(epoch_losses)
        print(
            "PROGRESS "
            f"phase=b9_m0 epoch={epoch + 1}/{args.m0_epochs} "
            f"steps={len(epoch_losses)} loss_mean={sum(epoch_losses) / max(len(epoch_losses), 1):.6f} "
            f"elapsed_seconds={time.time() - started:.1f}",
            flush=True,
        )
        if args.m0_epoch_checkpoint_out:
            checkpoint_path = Path(args.m0_epoch_checkpoint_out)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "encoder": encoder.state_dict(),
                    "heads": {domain: head.state_dict() for domain, head in heads.items()},
                    "optimizer": optimizer.state_dict(),
                    "epoch_completed": epoch + 1,
                    "m0_epochs_requested": args.m0_epochs,
                    "losses": losses,
                },
                checkpoint_path,
            )
            print(
                "PROGRESS "
                f"phase=b9_m0_checkpoint epoch_completed={epoch + 1}/{args.m0_epochs} "
                f"path={checkpoint_path}",
                flush=True,
            )
    encoder.eval()
    for head in heads.values():
        head.eval()
    metrics = evaluate_all(
        eval_samples,
        encoder,
        heads,
        args.eval_batch_size,
        device,
        target_domain,
        target_bucket=target_bucket,
        amp=args.amp,
        amp_dtype=args.amp_dtype,
    )
    return encoder, heads, losses, metrics, checkpoint_source


def run_extra_points(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    rng = random.Random(args.seed)
    if args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)
    if args.torch_num_interop_threads > 0:
        torch.set_num_interop_threads(args.torch_num_interop_threads)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if args.matmul_precision:
        torch.set_float32_matmul_precision(args.matmul_precision)

    loaded, train_samples, eval_samples, domain_summary, target_domain, target_bucket = load_data(config, args, rng, device)
    run_root = Path(config["run_root"])

    encoder, heads, m0_losses, m0_metrics, m0_checkpoint_source = train_m0(
        loaded, train_samples, eval_samples, target_domain, target_bucket, args, device
    )
    m0_report = trainable_parameter_report(nn.ModuleDict({"encoder": encoder, "heads": nn.ModuleDict(heads)}))

    if args.save_checkpoints:
        (run_root / "checkpoints").mkdir(parents=True, exist_ok=True)
        torch.save(
            {"encoder": encoder.state_dict(), "heads": {domain: head.state_dict() for domain, head in heads.items()}},
            run_root / "checkpoints" / "b9_m0_shared_sasrec_base.pt",
        )

    standard_deltagate_payload: dict[str, Any] | None = None
    if args.include_standard_deltagate:
        dg_encoder = copy.deepcopy(encoder).to(device)
        dg_heads = {domain: copy.deepcopy(head).to(device) for domain, head in heads.items()}
        dg_encoder_before = state_snapshot(dg_encoder)
        dg_target_head_before = state_snapshot(dg_heads[target_domain])
        dg_retained_before = {
            domain: state_snapshot(head)
            for domain, head in dg_heads.items()
            if domain != target_domain
        }
        freeze_module(dg_encoder)
        for head in dg_heads.values():
            freeze_module(head)
        dg_adapter = SharedBackboneDeltaGateAdapter(dg_encoder, dg_heads[target_domain], args.embedding_dim).to(device)
        dg_adapter.train()
        dg_encoder.eval()
        for head in dg_heads.values():
            head.eval()
        dg_param_before = trainable_parameter_report(dg_adapter)
        dg_epochs = args.standard_deltagate_epochs if args.standard_deltagate_epochs is not None else args.gap_only_epochs
        dg_losses = train_target_model(
            train_samples[target_domain],
            lambda seq, delta_t: dg_adapter(seq, delta_t)[0],
            [parameter for parameter in dg_adapter.parameters() if parameter.requires_grad],
            dg_epochs,
            args.batch_size,
            device,
            args.lr,
            args.amp,
            args.amp_dtype,
            "mr_e4_standard_deltagate",
            eval_modules_during_training=[dg_encoder],
        )
        dg_encoder.eval()
        for head in dg_heads.values():
            head.eval()
        dg_adapter.eval()
        dg_metrics = evaluate_all(
            eval_samples,
            dg_encoder,
            dg_heads,
            args.eval_batch_size,
            device,
            target_domain,
            target_adapter=dg_adapter,
            target_bucket=target_bucket,
            amp=args.amp,
            amp_dtype=args.amp_dtype,
        )
        dg_encoder_drift = max_abs_drift(dg_encoder_before, dg_encoder)
        dg_target_head_drift = max_abs_drift(dg_target_head_before, dg_heads[target_domain])
        dg_retained_head_drift = max(
            [max_abs_drift(dg_retained_before[domain], dg_heads[domain]) for domain in dg_retained_before] or [0.0]
        )
        dg_param_after = trainable_parameter_report(dg_adapter)
        standard_deltagate_payload = {
            "losses": dg_losses,
            "metrics": dg_metrics,
            "delta_vs_m0": metric_delta(dg_metrics, m0_metrics),
            "parameter_report_before": dg_param_before.__dict__,
            "parameter_report_after": dg_param_after.__dict__,
            "trainable_name_ok": all(
                name == "global_prior" or name.startswith("gate_mlp.")
                for name in dg_param_after.trainable_names
            ),
            "encoder_max_abs_drift": dg_encoder_drift,
            "target_head_max_abs_drift": dg_target_head_drift,
            "retained_head_max_abs_drift": dg_retained_head_drift,
            "surface": {
                "type": "standard_deltagate_reference",
                "gate_inputs": ["log1p_gap", "personalized_hidden_state"],
            },
        }

    gap_only_payload: dict[str, Any] | None = None
    if args.include_gap_only:
        gap_encoder = copy.deepcopy(encoder).to(device)
        gap_heads = {domain: copy.deepcopy(head).to(device) for domain, head in heads.items()}
        gap_encoder_before = state_snapshot(gap_encoder)
        gap_target_head_before = state_snapshot(gap_heads[target_domain])
        gap_retained_before = {
            domain: state_snapshot(head)
            for domain, head in gap_heads.items()
            if domain != target_domain
        }
        freeze_module(gap_encoder)
        for head in gap_heads.values():
            freeze_module(head)
        gap_hidden_dim = args.gap_only_hidden_dim or args.embedding_dim * 2
        gap_adapter = GapOnlyDeltaGateAdapter(
            gap_encoder,
            gap_heads[target_domain],
            args.embedding_dim,
            hidden_dim=gap_hidden_dim,
        ).to(device)
        gap_adapter.train()
        gap_encoder.eval()
        for head in gap_heads.values():
            head.eval()
        gap_param_before = trainable_parameter_report(gap_adapter)
        gap_losses = train_target_model(
            train_samples[target_domain],
            lambda seq, delta_t: gap_adapter(seq, delta_t)[0],
            [parameter for parameter in gap_adapter.parameters() if parameter.requires_grad],
            args.gap_only_epochs,
            args.batch_size,
            device,
            args.lr,
            args.amp,
            args.amp_dtype,
            "mr_e4_gap_only_deltagate",
            eval_modules_during_training=[gap_encoder],
        )
        gap_encoder.eval()
        for head in gap_heads.values():
            head.eval()
        gap_adapter.eval()
        gap_metrics = evaluate_all(
            eval_samples,
            gap_encoder,
            gap_heads,
            args.eval_batch_size,
            device,
            target_domain,
            target_adapter=gap_adapter,
            target_bucket=target_bucket,
            amp=args.amp,
            amp_dtype=args.amp_dtype,
        )
        gap_encoder_drift = max_abs_drift(gap_encoder_before, gap_encoder)
        gap_target_head_drift = max_abs_drift(gap_target_head_before, gap_heads[target_domain])
        gap_retained_head_drift = max(
            [max_abs_drift(gap_retained_before[domain], gap_heads[domain]) for domain in gap_retained_before] or [0.0]
        )
        gap_param_after = trainable_parameter_report(gap_adapter)
        gap_only_payload = {
            "losses": gap_losses,
            "metrics": gap_metrics,
            "delta_vs_m0": metric_delta(gap_metrics, m0_metrics),
            "parameter_report_before": gap_param_before.__dict__,
            "parameter_report_after": gap_param_after.__dict__,
            "trainable_name_ok": all(
                name == "global_prior" or name.startswith("gate_mlp.")
                for name in gap_param_after.trainable_names
            ),
            "encoder_max_abs_drift": gap_encoder_drift,
            "target_head_max_abs_drift": gap_target_head_drift,
            "retained_head_max_abs_drift": gap_retained_head_drift,
            "surface": {
                "type": "gap_only_deltagate_ablation",
                "gate_inputs": ["log1p_gap"],
                "removed_inputs": ["personalized_hidden_state"],
                "gate_hidden_dim": gap_hidden_dim,
            },
        }

    capacity_gap_only_payload: dict[str, Any] | None = None
    if args.include_capacity_matched_gap_only:
        target_params = args.capacity_gap_only_target_params or deltagate_trainable_params(args.embedding_dim)
        capacity_gap_hidden_dim = (
            args.capacity_gap_only_hidden_dim
            or capacity_matched_gap_only_hidden_dim(args.embedding_dim, target_params)
        )
        expected_params = gap_only_trainable_params(args.embedding_dim, capacity_gap_hidden_dim)
        relative_error = abs(expected_params - target_params) / max(target_params, 1)
        if relative_error > args.capacity_gap_only_tolerance:
            raise ValueError(
                "capacity-matched gap-only adapter is outside tolerance: "
                f"target={target_params}, hidden_dim={capacity_gap_hidden_dim}, params={expected_params}, "
                f"tolerance={args.capacity_gap_only_tolerance}"
            )

        capacity_gap_encoder = copy.deepcopy(encoder).to(device)
        capacity_gap_heads = {domain: copy.deepcopy(head).to(device) for domain, head in heads.items()}
        capacity_gap_encoder_before = state_snapshot(capacity_gap_encoder)
        capacity_gap_target_head_before = state_snapshot(capacity_gap_heads[target_domain])
        capacity_gap_retained_before = {
            domain: state_snapshot(head)
            for domain, head in capacity_gap_heads.items()
            if domain != target_domain
        }
        freeze_module(capacity_gap_encoder)
        for head in capacity_gap_heads.values():
            freeze_module(head)
        capacity_gap_adapter = GapOnlyDeltaGateAdapter(
            capacity_gap_encoder,
            capacity_gap_heads[target_domain],
            args.embedding_dim,
            hidden_dim=capacity_gap_hidden_dim,
        ).to(device)
        capacity_gap_adapter.train()
        capacity_gap_encoder.eval()
        for head in capacity_gap_heads.values():
            head.eval()
        capacity_gap_param_before = trainable_parameter_report(capacity_gap_adapter)
        capacity_gap_losses = train_target_model(
            train_samples[target_domain],
            lambda seq, delta_t: capacity_gap_adapter(seq, delta_t)[0],
            [parameter for parameter in capacity_gap_adapter.parameters() if parameter.requires_grad],
            args.capacity_gap_only_epochs,
            args.batch_size,
            device,
            args.lr,
            args.amp,
            args.amp_dtype,
            "p087_capacity_matched_gap_only",
            eval_modules_during_training=[capacity_gap_encoder],
        )
        capacity_gap_encoder.eval()
        for head in capacity_gap_heads.values():
            head.eval()
        capacity_gap_adapter.eval()
        capacity_gap_metrics = evaluate_all(
            eval_samples,
            capacity_gap_encoder,
            capacity_gap_heads,
            args.eval_batch_size,
            device,
            target_domain,
            target_adapter=capacity_gap_adapter,
            target_bucket=target_bucket,
            amp=args.amp,
            amp_dtype=args.amp_dtype,
        )
        capacity_gap_encoder_drift = max_abs_drift(capacity_gap_encoder_before, capacity_gap_encoder)
        capacity_gap_target_head_drift = max_abs_drift(
            capacity_gap_target_head_before,
            capacity_gap_heads[target_domain],
        )
        capacity_gap_retained_head_drift = max(
            [
                max_abs_drift(capacity_gap_retained_before[domain], capacity_gap_heads[domain])
                for domain in capacity_gap_retained_before
            ]
            or [0.0]
        )
        capacity_gap_param_after = trainable_parameter_report(capacity_gap_adapter)
        capacity_gap_only_payload = {
            "losses": capacity_gap_losses,
            "metrics": capacity_gap_metrics,
            "delta_vs_m0": metric_delta(capacity_gap_metrics, m0_metrics),
            "parameter_report_before": capacity_gap_param_before.__dict__,
            "parameter_report_after": capacity_gap_param_after.__dict__,
            "trainable_name_ok": all(
                name == "global_prior" or name.startswith("gate_mlp.")
                for name in capacity_gap_param_after.trainable_names
            ),
            "encoder_max_abs_drift": capacity_gap_encoder_drift,
            "target_head_max_abs_drift": capacity_gap_target_head_drift,
            "retained_head_max_abs_drift": capacity_gap_retained_head_drift,
            "capacity_match": {
                "target_params": target_params,
                "actual_params": capacity_gap_param_after.trainable,
                "hidden_dim": capacity_gap_hidden_dim,
                "relative_error": abs(capacity_gap_param_after.trainable - target_params) / max(target_params, 1),
                "tolerance": args.capacity_gap_only_tolerance,
            },
            "surface": {
                "type": "capacity_matched_gap_only_deltagate_ablation",
                "gate_inputs": ["log1p_gap"],
                "removed_inputs": ["personalized_hidden_state"],
                "gate_hidden_dim": capacity_gap_hidden_dim,
            },
        }

    # B9-H / MR-E5: target head only, frozen shared encoder and retained heads.
    head_encoder = copy.deepcopy(encoder).to(device)
    head_heads = {domain: copy.deepcopy(head).to(device) for domain, head in heads.items()}
    head_encoder_before = state_snapshot(head_encoder)
    retained_before = {domain: state_snapshot(head_heads[domain]) for domain in head_heads if domain != target_domain}
    freeze_module(head_encoder)
    for domain, head in head_heads.items():
        if domain != target_domain:
            freeze_module(head)
    head_model = HeadOnlyTargetModel(head_encoder, head_heads[target_domain]).to(device)
    head_losses = train_target_model(
        train_samples[target_domain],
        head_model,
        [parameter for parameter in head_model.parameters() if parameter.requires_grad],
        args.head_epochs,
        args.batch_size,
        device,
        args.lr,
        args.amp,
        args.amp_dtype,
        "b9_frozen_target_head",
        eval_modules_during_training=[head_encoder],
    )
    head_encoder.eval()
    for head in head_heads.values():
        head.eval()
    head_model.eval()
    head_metrics = evaluate_all(
        eval_samples,
        head_encoder,
        head_heads,
        args.eval_batch_size,
        device,
        target_domain,
        target_bucket=target_bucket,
        amp=args.amp,
        amp_dtype=args.amp_dtype,
    )
    head_encoder_drift = max_abs_drift(head_encoder_before, head_encoder)
    head_retained_drift = max(
        [max_abs_drift(retained_before[domain], head_heads[domain]) for domain in retained_before] or [0.0]
    )
    head_report = trainable_parameter_report(head_model)

    # B9-T: time adapter only, frozen encoder and all heads.
    time_encoder = copy.deepcopy(encoder).to(device)
    time_heads = {domain: copy.deepcopy(head).to(device) for domain, head in heads.items()}
    time_encoder_before = state_snapshot(time_encoder)
    time_heads_before = {domain: state_snapshot(head) for domain, head in time_heads.items()}
    freeze_module(time_encoder)
    for head in time_heads.values():
        freeze_module(head)
    time_model = TimeConcatTargetModel(time_encoder, time_heads[target_domain], args.embedding_dim).to(device)
    time_losses = train_target_model(
        train_samples[target_domain],
        time_model,
        [parameter for parameter in time_model.parameters() if parameter.requires_grad],
        args.time_adapter_epochs,
        args.batch_size,
        device,
        args.lr,
        args.amp,
        args.amp_dtype,
        "b9_frozen_time_adapter",
        eval_modules_during_training=[time_encoder],
    )
    time_encoder.eval()
    for head in time_heads.values():
        head.eval()
    time_model.eval()
    time_metrics = evaluate_all(
        eval_samples,
        time_encoder,
        time_heads,
        args.eval_batch_size,
        device,
        target_domain,
        target_time_model=time_model,
        target_bucket=target_bucket,
        amp=args.amp,
        amp_dtype=args.amp_dtype,
    )
    time_encoder_drift = max_abs_drift(time_encoder_before, time_encoder)
    time_head_drift = max([max_abs_drift(time_heads_before[domain], time_heads[domain]) for domain in time_heads] or [0.0])
    time_report = trainable_parameter_report(time_model)
    if args.save_checkpoints:
        torch.save(
            {
                "encoder": time_encoder.state_dict(),
                "heads": {domain: head.state_dict() for domain, head in time_heads.items()},
                "time_model": time_model.state_dict(),
            },
            run_root / "checkpoints" / "b9_frozen_time_adapter.pt",
        )

    capacity_payload: dict[str, Any] | None = None
    if args.include_capacity_matched:
        target_params = args.capacity_match_target_params or deltagate_trainable_params(args.embedding_dim)
        capacity_hidden_dim = args.capacity_time_hidden_dim or capacity_matched_hidden_dim(args.embedding_dim, target_params)
        expected_params = time_adapter_trainable_params(args.embedding_dim, capacity_hidden_dim)
        if abs(expected_params - target_params) / max(target_params, 1) > args.capacity_match_tolerance:
            raise ValueError(
                "capacity-matched time adapter is outside tolerance: "
                f"target={target_params}, hidden_dim={capacity_hidden_dim}, params={expected_params}, "
                f"tolerance={args.capacity_match_tolerance}"
            )

        capacity_encoder = copy.deepcopy(encoder).to(device)
        capacity_heads = {domain: copy.deepcopy(head).to(device) for domain, head in heads.items()}
        capacity_encoder_before = state_snapshot(capacity_encoder)
        capacity_heads_before = {domain: state_snapshot(head) for domain, head in capacity_heads.items()}
        freeze_module(capacity_encoder)
        for head in capacity_heads.values():
            freeze_module(head)
        capacity_model = FrozenTimeAdapterTargetModel(
            capacity_encoder,
            capacity_heads[target_domain],
            args.embedding_dim,
            capacity_hidden_dim,
        ).to(device)
        capacity_losses = train_target_model(
            train_samples[target_domain],
            capacity_model,
            [parameter for parameter in capacity_model.parameters() if parameter.requires_grad],
            args.capacity_time_adapter_epochs,
            args.batch_size,
            device,
            args.lr,
            args.amp,
            args.amp_dtype,
            "b9_capacity_matched_time_adapter",
            eval_modules_during_training=[capacity_encoder],
        )
        capacity_encoder.eval()
        for head in capacity_heads.values():
            head.eval()
        capacity_model.eval()
        capacity_metrics = evaluate_all(
            eval_samples,
            capacity_encoder,
            capacity_heads,
            args.eval_batch_size,
            device,
            target_domain,
            target_time_model=capacity_model,
            target_bucket=target_bucket,
            amp=args.amp,
            amp_dtype=args.amp_dtype,
        )
        capacity_encoder_drift = max_abs_drift(capacity_encoder_before, capacity_encoder)
        capacity_head_drift = max(
            [max_abs_drift(capacity_heads_before[domain], capacity_heads[domain]) for domain in capacity_heads] or [0.0]
        )
        capacity_report = trainable_parameter_report(capacity_model)
        capacity_payload = {
            "losses": capacity_losses,
            "metrics": capacity_metrics,
            "delta_vs_m0": metric_delta(capacity_metrics, m0_metrics),
            "parameter_report": capacity_report.__dict__,
            "encoder_max_abs_drift": capacity_encoder_drift,
            "head_max_abs_drift": capacity_head_drift,
            "capacity_match": {
                "target_params": target_params,
                "actual_params": capacity_report.trainable,
                "hidden_dim": capacity_hidden_dim,
                "relative_error": abs(capacity_report.trainable - target_params) / max(target_params, 1),
                "tolerance": args.capacity_match_tolerance,
            },
        }

    methods = {
        "b9_m0_shared_sasrec_base": {
            "losses": m0_losses,
            "metrics": m0_metrics,
            "parameter_report": m0_report.__dict__,
            "checkpoint_source": m0_checkpoint_source,
        },
        "b9_frozen_target_head": {
            "losses": head_losses,
            "metrics": head_metrics,
            "delta_vs_m0": metric_delta(head_metrics, m0_metrics),
            "parameter_report": head_report.__dict__,
            "encoder_max_abs_drift": head_encoder_drift,
            "retained_head_max_abs_drift": head_retained_drift,
        },
        "b9_frozen_time_adapter": {
            "losses": time_losses,
            "metrics": time_metrics,
            "delta_vs_m0": metric_delta(time_metrics, m0_metrics),
            "parameter_report": time_report.__dict__,
            "encoder_max_abs_drift": time_encoder_drift,
            "head_max_abs_drift": time_head_drift,
        },
    }
    if capacity_payload is not None:
        methods["b9_capacity_matched_time_adapter"] = capacity_payload
    if standard_deltagate_payload is not None:
        methods["mr_e4_standard_deltagate_adapter"] = standard_deltagate_payload
    if gap_only_payload is not None:
        methods["mr_e4_gap_only_deltagate_adapter"] = gap_only_payload
    if capacity_gap_only_payload is not None:
        methods["p087_capacity_matched_gap_only_adapter"] = capacity_gap_only_payload
    finite_losses = all(
        math.isfinite(loss)
        for value in methods.values()
        for loss in value.get("losses", [])
    )
    errors = []
    if not finite_losses:
        errors.append("non-finite loss observed")
    if head_encoder_drift != 0.0 or head_retained_drift != 0.0:
        errors.append("frozen target-head point drifted retained service components")
    if time_encoder_drift != 0.0 or time_head_drift != 0.0:
        errors.append("frozen time-adapter point drifted frozen components")
    if capacity_payload is not None and (
        capacity_payload["encoder_max_abs_drift"] != 0.0 or capacity_payload["head_max_abs_drift"] != 0.0
    ):
        errors.append("capacity-matched time-adapter point drifted frozen components")
    if standard_deltagate_payload is not None and (
        not standard_deltagate_payload["trainable_name_ok"]
            or standard_deltagate_payload["encoder_max_abs_drift"] != 0.0
            or standard_deltagate_payload["target_head_max_abs_drift"] != 0.0
            or standard_deltagate_payload["retained_head_max_abs_drift"] != 0.0
    ):
        errors.append("standard DeltaGate reference violated frozen contract")
    if gap_only_payload is not None and (
        not gap_only_payload["trainable_name_ok"]
            or gap_only_payload["encoder_max_abs_drift"] != 0.0
            or gap_only_payload["target_head_max_abs_drift"] != 0.0
            or gap_only_payload["retained_head_max_abs_drift"] != 0.0
    ):
        errors.append("gap-only DeltaGate ablation violated frozen contract")
    if capacity_gap_only_payload is not None and (
        not capacity_gap_only_payload["trainable_name_ok"]
            or capacity_gap_only_payload["encoder_max_abs_drift"] != 0.0
            or capacity_gap_only_payload["target_head_max_abs_drift"] != 0.0
            or capacity_gap_only_payload["retained_head_max_abs_drift"] != 0.0
        or capacity_gap_only_payload["capacity_match"]["relative_error"]
        > capacity_gap_only_payload["capacity_match"]["tolerance"]
    ):
        errors.append("capacity-matched gap-only DeltaGate ablation violated frozen contract or capacity tolerance")

    return {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "run_type": args.run_type,
        "seed": args.seed,
        "device": str(device),
        "target_domain": target_domain,
        "target_bucket": target_bucket,
        "retained_domains": [domain for domain in sorted(loaded) if domain != target_domain],
        "elapsed_seconds": time.time() - started,
        "limits": {
            "max_train_samples_per_domain": args.max_train_samples_per_domain,
            "max_eval_samples_per_domain": args.max_eval_samples_per_domain,
            "m0_epochs": args.m0_epochs,
            "head_epochs": args.head_epochs,
            "time_adapter_epochs": args.time_adapter_epochs,
            "capacity_time_adapter_epochs": args.capacity_time_adapter_epochs,
            "include_capacity_matched": args.include_capacity_matched,
            "include_standard_deltagate": args.include_standard_deltagate,
            "standard_deltagate_epochs": args.standard_deltagate_epochs,
            "include_gap_only": args.include_gap_only,
            "gap_only_epochs": args.gap_only_epochs,
            "gap_only_hidden_dim": args.gap_only_hidden_dim or args.embedding_dim * 2,
            "include_capacity_matched_gap_only": args.include_capacity_matched_gap_only,
            "capacity_gap_only_epochs": args.capacity_gap_only_epochs,
            "capacity_gap_only_hidden_dim": (
                methods.get("p087_capacity_matched_gap_only_adapter", {})
                .get("capacity_match", {})
                .get("hidden_dim")
            ),
            "capacity_time_hidden_dim": (
                methods.get("b9_capacity_matched_time_adapter", {})
                .get("capacity_match", {})
                .get("hidden_dim")
            ),
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "m0_domain_schedule": args.m0_domain_schedule,
            "embedding_dim": args.embedding_dim,
            "num_layers": args.num_layers,
            "max_seq_len": args.max_seq_len,
            "amp": args.amp,
            "amp_dtype": args.amp_dtype,
            "pin_memory": args.pin_memory,
            "matmul_precision": args.matmul_precision,
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
        },
        "domain_summary": domain_summary,
        **methods,
    }


def write_markdown(path: Path, report: dict[str, Any], target_slice: str = "video_games_>365d") -> None:
    target_slice = f"{report.get('target_domain', 'video_games')}_{report.get('target_bucket', '>365d')}"
    retained_slices = [
        f"{domain}_all"
        for domain in report.get("retained_domains", ["movies_tv", "cds_vinyl"])
    ]
    lines = [
        "# B9 Extra Points Summary",
        "",
        f"Status: `{report['status']}`",
        "",
        "| Method | target Hit@10 | target NDCG@10 | delta Hit@10 vs M0 | retained max Hit@10 drift | trainable params |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    m0 = report["b9_m0_shared_sasrec_base"]["metrics"]
    rows = [
        ("Shared-SASRec base", "b9_m0_shared_sasrec_base"),
        ("Standard DeltaGate", "mr_e4_standard_deltagate_adapter"),
        ("Gap-only DeltaGate", "mr_e4_gap_only_deltagate_adapter"),
        ("Frozen target head", "b9_frozen_target_head"),
        ("Frozen time adapter", "b9_frozen_time_adapter"),
    ]
    if "b9_capacity_matched_time_adapter" in report:
        rows.append(("Capacity-matched time adapter", "b9_capacity_matched_time_adapter"))
    for label, key in rows:
        if key not in report:
            continue
        method = report[key]
        metrics = method["metrics"]
        target = metrics[target_slice]
        delta = 0.0 if key == "b9_m0_shared_sasrec_base" else target["hit10"] - m0[target_slice]["hit10"]
        retained_drifts = []
        for slice_name in retained_slices:
            retained_drifts.append(abs(metrics[slice_name]["hit10"] - m0[slice_name]["hit10"]))
        if "parameter_report" in method:
            params = method["parameter_report"]["trainable"]
        elif "parameter_report_after" in method:
            params = method["parameter_report_after"]["trainable"]
        else:
            params = 0
        lines.append(
            f"| {label} | {fmt_pct(float(target['hit10']))} | {fmt_pct(float(target['ndcg10']))} | "
            f"{fmt_pp(float(delta))} | {fmt_pp(max(retained_drifts))} | {int(params):,} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- These are same-seed diagnostic points, not a multi-seed paper table by themselves.",
            "- MR-E4 compares standard DeltaGate against a gap-only gate under the same frozen contract when both rows are present.",
            "- P087 adds a capacity-matched gap-only gate with the same gap-only input, so its comparison with standard DeltaGate isolates capacity from personalized gate input.",
            "- MR-E5 uses the frozen target-head row to fill the target-output-only surface.",
            "- All extra points freeze the shared encoder; retained drift should remain zero unless a contract check fails.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/shared_backbone_seed42_protocol.json"))
    parser.add_argument("--json-out", type=Path, default=Path("reports/b9_extra_points_seed42096.json"))
    parser.add_argument("--md-out", type=Path, default=Path("reports/b9_extra_points_seed42096.md"))
    parser.add_argument("--run-type", default="bounded_smoke", choices=["bounded_smoke", "seed42"])
    parser.add_argument("--seed", type=int, default=42096)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-train-samples-per-domain", type=int, default=64)
    parser.add_argument("--max-eval-samples-per-domain", type=int, default=64)
    parser.add_argument("--prioritize-target-bucket", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--m0-epochs", type=int, default=1)
    parser.add_argument("--head-epochs", type=int, default=1)
    parser.add_argument("--time-adapter-epochs", type=int, default=1)
    parser.add_argument("--include-capacity-matched", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--capacity-time-adapter-epochs", type=int, default=None)
    parser.add_argument("--include-standard-deltagate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--standard-deltagate-epochs", type=int, default=None)
    parser.add_argument("--include-gap-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gap-only-epochs", type=int, default=1)
    parser.add_argument("--gap-only-hidden-dim", type=int, default=0)
    parser.add_argument("--include-capacity-matched-gap-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--capacity-gap-only-epochs", type=int, default=None)
    parser.add_argument("--capacity-gap-only-hidden-dim", type=int, default=0)
    parser.add_argument("--capacity-gap-only-target-params", type=int, default=0)
    parser.add_argument("--capacity-gap-only-tolerance", type=float, default=0.001)
    parser.add_argument("--capacity-time-hidden-dim", type=int, default=0)
    parser.add_argument("--capacity-match-target-params", type=int, default=0)
    parser.add_argument("--capacity-match-tolerance", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--m0-domain-schedule", choices=["max_cycle", "once"], default="once")
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=50)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--amp-dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--matmul-precision", choices=["highest", "high", "medium"], default="high")
    parser.add_argument("--torch-num-threads", type=int, default=0)
    parser.add_argument("--torch-num-interop-threads", type=int, default=0)
    parser.add_argument("--save-checkpoints", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--load-m0-checkpoint", type=Path, default=None)
    parser.add_argument("--resume-m0-checkpoint", type=Path, default=None)
    parser.add_argument("--m0-epoch-checkpoint-out", type=Path, default=None)
    args = parser.parse_args()
    if args.capacity_time_adapter_epochs is None:
        args.capacity_time_adapter_epochs = args.time_adapter_epochs
    if args.capacity_gap_only_epochs is None:
        args.capacity_gap_only_epochs = args.gap_only_epochs

    if args.run_type == "seed42" and (
        args.max_train_samples_per_domain > 0 or args.max_eval_samples_per_domain > 0
    ):
        parser.error(
            "--run-type seed42 requires --max-train-samples-per-domain 0 "
            "and --max-eval-samples-per-domain 0; use bounded_smoke for capped runs"
        )

    config = load_json(args.config)
    report = run_extra_points(config, args)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(args.md_out, report)
    print(json.dumps({"status": report["status"], "json": str(args.json_out), "md": str(args.md_out)}, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
