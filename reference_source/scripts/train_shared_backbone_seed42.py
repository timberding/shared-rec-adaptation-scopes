#!/usr/bin/env python3
"""Train/evaluate the shared-backbone seed42 comparison suite.

The suite follows `refine-logs/SHARED_BACKBONE_SEED42_PROTOCOL.md`:

- M0: shared SASRec encoder with domain-specific heads.
- M1: TimeConcat-style target finetune from M0, updating the encoder.
- M2: strict frozen-main DeltaGate adapter from M0.

Default arguments are conservative smoke settings. Full runs should explicitly
raise sample caps, epochs, and model size in a separate launch record.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import pickle
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.shared_backbone import (  # noqa: E402
    DomainItemHead,
    SharedBackboneDeltaGateAdapter,
    SharedGRUEncoder,
    SharedSASRecEncoder,
    freeze_module,
    trainable_parameter_report,
)


@dataclass(frozen=True)
class Batch:
    seq: torch.Tensor
    target: torch.Tensor
    delta_t: torch.Tensor
    buckets: list[str]


@dataclass(frozen=True)
class TensorSamples:
    seq: torch.Tensor
    target: torch.Tensor
    delta_t: torch.Tensor
    buckets: list[str]

    def __len__(self) -> int:
        return int(self.target.shape[0])


def make_encoder(args: argparse.Namespace) -> nn.Module:
    if args.encoder_type == "sasrec":
        return SharedSASRecEncoder(
            embedding_dim=args.embedding_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            max_seq_len=args.max_seq_len,
            dropout=args.dropout,
        )
    if args.encoder_type == "gru":
        return SharedGRUEncoder(
            embedding_dim=args.embedding_dim,
            num_layers=args.num_layers,
            max_seq_len=args.max_seq_len,
            dropout=args.dropout,
        )
    raise ValueError(f"unknown encoder_type={args.encoder_type}")


class TimeConcatTargetModel(nn.Module):
    """Small TimeConcat surface used only for target-domain finetuning."""

    def __init__(self, encoder: nn.Module, target_head: DomainItemHead, embedding_dim: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.target_head = target_head
        self.time_projection = nn.Sequential(
            nn.Linear(1, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, seq: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        padding_mask = seq.eq(0)
        hidden = self.encoder(self.target_head.embed_sequence(seq), padding_mask)
        hidden = hidden + self.time_projection(torch.log1p(delta_t).unsqueeze(-1))
        return self.target_head.logits(hidden)


class FrozenLoRATargetAdapter(nn.Module):
    """LoRA-style low-rank target-side residual over a frozen retained path."""

    def __init__(
        self,
        encoder: nn.Module,
        target_head: DomainItemHead,
        embedding_dim: int,
        rank: int = 8,
        alpha: float = 8.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.encoder = encoder
        self.target_head = target_head
        self.embedding_dim = embedding_dim
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.lora_down = nn.Linear(embedding_dim + 1, self.rank, bias=False)
        self.lora_up = nn.Linear(self.rank, embedding_dim, bias=False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, seq: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        padding_mask = seq.eq(0)
        hidden = self.encoder(self.target_head.embed_sequence(seq), padding_mask).detach()
        adapter_input = torch.cat([hidden, torch.log1p(delta_t).unsqueeze(-1)], dim=-1)
        residual = self.lora_up(self.lora_down(adapter_input)) * (self.alpha / self.rank)
        return self.target_head.logits(hidden + residual)


def round2_contract_tags(args: argparse.Namespace) -> list[str]:
    """Return executable round-2 contract tags for AutoDL runner audits."""

    if not args.round2_contract_checks:
        return []
    contracts: list[str] = []
    # R2-E1_GRU_S6 hard fail: GRU diagnostic must include the frozen target-side TC surface.
    if args.encoder_type == "gru":
        if not args.include_frozen_target_timeconcat:
            raise RuntimeError("R2-E1_GRU_S6 requires --include-frozen-target-timeconcat")
        contracts.append("R2-E1_GRU_S6")
    # R2-E2_LORA_PEFT_SURFACE hard fail: LoRA PEFT surface must run with the explicit flag.
    if args.include_lora_target_adapter:
        contracts.append("R2-E2_LORA_PEFT_SURFACE")
    return contracts


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return data


def load_processed(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a dict")
    return data


def left_pad(seq: list[int], max_seq_len: int) -> list[int]:
    trimmed = list(seq)[-max_seq_len:]
    return [0] * (max_seq_len - len(trimmed)) + trimmed


def target_slice_label(config_bucket: str, threshold_days: float | None, explicit_label: str | None = None) -> str:
    if explicit_label:
        return explicit_label
    if threshold_days is None:
        return config_bucket
    if float(threshold_days).is_integer():
        return f">{int(threshold_days)}d"
    return f">{threshold_days:g}d"


def sample_matches_target_slice(
    sample: dict[str, Any],
    bucket: str | None = None,
    gap_threshold_days: float | None = None,
) -> bool:
    if gap_threshold_days is not None:
        return float(sample.get("gap_days", 0.0)) > float(gap_threshold_days)
    if bucket:
        return sample.get("bucket") == bucket
    return True


def capped_samples(
    samples: list[dict[str, Any]],
    cap: int,
    rng: random.Random,
    bucket: str | None = None,
    gap_threshold_days: float | None = None,
) -> list[dict[str, Any]]:
    if bucket or gap_threshold_days is not None:
        selected = [
            sample
            for sample in samples
            if sample_matches_target_slice(sample, bucket=bucket, gap_threshold_days=gap_threshold_days)
        ]
        fallback = [
            sample
            for sample in samples
            if not sample_matches_target_slice(sample, bucket=bucket, gap_threshold_days=gap_threshold_days)
        ]
        selected = selected + fallback
    else:
        selected = list(samples)
    if cap > 0:
        selected = selected[:cap]
    rng.shuffle(selected)
    return selected


def make_batch(samples: list[dict[str, Any]], max_seq_len: int, device: torch.device) -> Batch:
    seq = torch.tensor([left_pad(sample["seq_idx"], max_seq_len) for sample in samples], dtype=torch.long, device=device)
    target = torch.tensor([int(sample["target_idx"]) - 1 for sample in samples], dtype=torch.long, device=device)
    delta_t = torch.tensor([float(sample.get("gap_days", 0.0)) for sample in samples], dtype=torch.float32, device=device)
    buckets = [str(sample.get("bucket", "missing")) for sample in samples]
    return Batch(seq=seq, target=target, delta_t=delta_t, buckets=buckets)


def tensorize_samples(samples: list[dict[str, Any]], max_seq_len: int, pin_memory: bool = False) -> TensorSamples:
    seq = torch.tensor([left_pad(sample["seq_idx"], max_seq_len) for sample in samples], dtype=torch.long)
    target = torch.tensor([int(sample["target_idx"]) - 1 for sample in samples], dtype=torch.long)
    delta_t = torch.tensor([float(sample.get("gap_days", 0.0)) for sample in samples], dtype=torch.float32)
    buckets = [str(sample.get("bucket", "missing")) for sample in samples]
    if pin_memory:
        try:
            seq = seq.pin_memory()
            target = target.pin_memory()
            delta_t = delta_t.pin_memory()
        except RuntimeError:
            pass
    return TensorSamples(seq=seq, target=target, delta_t=delta_t, buckets=buckets)


def batch_slices(length: int, batch_size: int) -> Iterable[slice]:
    for start in range(0, length, batch_size):
        yield slice(start, min(start + batch_size, length))


def tensor_batch(data: TensorSamples, index: slice | torch.Tensor, device: torch.device) -> Batch:
    non_blocking = device.type == "cuda"
    return Batch(
        seq=data.seq[index].to(device, non_blocking=non_blocking),
        target=data.target[index].to(device, non_blocking=non_blocking),
        delta_t=data.delta_t[index].to(device, non_blocking=non_blocking),
        buckets=[],
    )


def bucket_indices(
    data: TensorSamples,
    bucket: str | None,
    gap_threshold_days: float | None = None,
) -> torch.Tensor | None:
    if bucket is None and gap_threshold_days is None:
        return None
    if gap_threshold_days is not None:
        selected = [
            idx
            for idx, value in enumerate(data.delta_t.tolist())
            if float(value) > float(gap_threshold_days)
        ]
    else:
        selected = [idx for idx, value in enumerate(data.buckets) if value == bucket]
    if not selected:
        return torch.empty(0, dtype=torch.long)
    return torch.tensor(selected, dtype=torch.long)


def autocast_context(device: torch.device, enabled: bool, dtype_name: str):
    if not enabled or device.type != "cuda":
        return nullcontext()
    dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def domain_logits(encoder: nn.Module, head: DomainItemHead, seq: torch.Tensor) -> torch.Tensor:
    padding_mask = seq.eq(0)
    hidden = encoder(head.embed_sequence(seq), padding_mask)
    return head.logits(hidden)


def train_epoch(
    train_samples: dict[str, TensorSamples],
    encoder: SharedSASRecEncoder,
    heads: dict[str, DomainItemHead],
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    device: torch.device,
    amp: bool,
    amp_dtype: str,
    domain_schedule: str,
) -> list[float]:
    encoder.train()
    for head in heads.values():
        head.train()
    domain_batches = {domain: list(batch_slices(len(samples), batch_size)) for domain, samples in train_samples.items()}
    losses = []
    if domain_schedule == "once":
        for domain in sorted(domain_batches):
            items = domain_batches[domain]
            for item in items:
                batch = tensor_batch(train_samples[domain], item, device)
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(device, amp, amp_dtype):
                    logits = domain_logits(encoder, heads[domain], batch.seq)
                    loss = F.cross_entropy(logits, batch.target)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
    else:
        max_steps = max(len(items) for items in domain_batches.values())
        for step in range(max_steps):
            for domain in sorted(domain_batches):
                items = domain_batches[domain]
                if not items:
                    continue
                batch = tensor_batch(train_samples[domain], items[step % len(items)], device)
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(device, amp, amp_dtype):
                    logits = domain_logits(encoder, heads[domain], batch.seq)
                    loss = F.cross_entropy(logits, batch.target)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
    return losses


def train_target_model(
    samples: TensorSamples,
    model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    parameters: list[torch.nn.Parameter],
    epochs: int,
    batch_size: int,
    device: torch.device,
    lr: float,
    amp: bool,
    amp_dtype: str,
    phase_name: str,
    regularization_fn: Callable[[], torch.Tensor] | None = None,
    eval_modules_during_training: list[nn.Module] | None = None,
    activity_parameters: dict[str, torch.nn.Parameter] | None = None,
    parameter_activity: dict[str, dict[str, float | int]] | None = None,
    optimizer_weight_decay: float = 0.01,
) -> list[float]:
    if parameter_activity is not None and activity_parameters is None:
        raise ValueError("parameter_activity requires activity_parameters")
    if isinstance(model, nn.Module):
        model.train()
    for module in eval_modules_during_training or []:
        module.eval()
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=optimizer_weight_decay)
    if activity_parameters is not None and parameter_activity is not None:
        for name in activity_parameters:
            parameter_activity.setdefault(
                name,
                {
                    "optimizer_steps": 0,
                    "steps_with_nonzero_grad": 0,
                    "steps_with_update": 0,
                    "grad_l2_sum": 0.0,
                    "grad_max_abs": 0.0,
                    "update_l2_sum": 0.0,
                    "update_max_abs": 0.0,
                },
            )
    losses = []
    sample_batches = list(batch_slices(len(samples), batch_size))
    phase_started = time.time()
    for epoch in range(epochs):
        for module in eval_modules_during_training or []:
            module.eval()
        epoch_losses = []
        for sample_slice in sample_batches:
            batch = tensor_batch(samples, sample_slice, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, amp, amp_dtype):
                logits = model(batch.seq, batch.delta_t)
                loss = F.cross_entropy(logits, batch.target)
                if regularization_fn is not None:
                    loss = loss + regularization_fn()
            before_step = None
            if activity_parameters is not None:
                before_step = {name: parameter.detach().clone() for name, parameter in activity_parameters.items()}
            loss.backward()
            if activity_parameters is not None and parameter_activity is not None:
                for name, parameter in activity_parameters.items():
                    activity = parameter_activity[name]
                    activity["optimizer_steps"] = int(activity["optimizer_steps"]) + 1
                    grad = parameter.grad
                    if grad is None:
                        continue
                    grad_tensor = grad.detach().float()
                    grad_l2 = float(grad_tensor.norm().item())
                    grad_max = float(grad_tensor.abs().max().item())
                    activity["grad_l2_sum"] = float(activity["grad_l2_sum"]) + grad_l2
                    activity["grad_max_abs"] = max(float(activity["grad_max_abs"]), grad_max)
                    if grad_l2 > 0.0:
                        activity["steps_with_nonzero_grad"] = int(activity["steps_with_nonzero_grad"]) + 1
            optimizer.step()
            if activity_parameters is not None and parameter_activity is not None and before_step is not None:
                for name, parameter in activity_parameters.items():
                    activity = parameter_activity[name]
                    update_tensor = (parameter.detach() - before_step[name]).float()
                    update_l2 = float(update_tensor.norm().item())
                    update_max = float(update_tensor.abs().max().item())
                    activity["update_l2_sum"] = float(activity["update_l2_sum"]) + update_l2
                    activity["update_max_abs"] = max(float(activity["update_max_abs"]), update_max)
                    if update_l2 > 0.0:
                        activity["steps_with_update"] = int(activity["steps_with_update"]) + 1
            loss_value = float(loss.detach().cpu())
            losses.append(loss_value)
            epoch_losses.append(loss_value)
        print(
            "PROGRESS "
            f"phase={phase_name} epoch={epoch + 1}/{epochs} "
            f"steps={len(epoch_losses)} loss_mean={sum(epoch_losses) / max(len(epoch_losses), 1):.6f} "
            f"elapsed_seconds={time.time() - phase_started:.1f}",
            flush=True,
        )
    return losses


def evaluate_logits(
    samples: TensorSamples,
    logits_fn: Callable[[Batch], torch.Tensor],
    batch_size: int,
    device: torch.device,
    bucket: str | None = None,
    gap_threshold_days: float | None = None,
    amp: bool = False,
    amp_dtype: str = "bf16",
) -> dict[str, float | int]:
    selected_indices = bucket_indices(samples, bucket, gap_threshold_days=gap_threshold_days)
    count_total = len(samples) if selected_indices is None else int(selected_indices.numel())
    if count_total == 0:
        return {"count": 0, "hit10": 0.0, "ndcg10": 0.0}

    hits = 0.0
    ndcg = 0.0
    count = 0
    with torch.no_grad():
        if selected_indices is None:
            iterator: Iterable[slice | torch.Tensor] = batch_slices(len(samples), batch_size)
        else:
            iterator = (selected_indices[slc] for slc in batch_slices(count_total, batch_size))
        for index in iterator:
            batch = tensor_batch(samples, index, device)
            with autocast_context(device, amp, amp_dtype):
                logits = logits_fn(batch)
            topk = torch.topk(logits, k=min(10, logits.shape[-1]), dim=-1).indices
            for row, target in zip(topk.cpu().tolist(), batch.target.cpu().tolist()):
                count += 1
                if target in row:
                    rank = row.index(target) + 1
                    hits += 1.0
                    ndcg += 1.0 / math.log2(rank + 1)
    return {"count": count, "hit10": hits / count, "ndcg10": ndcg / count}


def evaluate_all(
    eval_samples: dict[str, TensorSamples],
    encoder: SharedSASRecEncoder,
    heads: dict[str, DomainItemHead],
    batch_size: int,
    device: torch.device,
    target_domain: str,
    target_time_model: nn.Module | None = None,
    target_adapter: SharedBackboneDeltaGateAdapter | None = None,
    target_bucket: str = ">365d",
    target_gap_threshold_days: float | None = None,
    amp: bool = False,
    amp_dtype: str = "bf16",
) -> dict[str, dict[str, float | int]]:
    results = {}
    for domain, samples in sorted(eval_samples.items()):
        if domain == target_domain and target_time_model is not None:
            logits_fn = lambda batch, model=target_time_model: model(batch.seq, batch.delta_t)
        elif domain == target_domain and target_adapter is not None:
            logits_fn = lambda batch, model=target_adapter: model(batch.seq, batch.delta_t)[0]
        else:
            logits_fn = lambda batch, d=domain: domain_logits(encoder, heads[d], batch.seq)
        results[f"{domain}_all"] = evaluate_logits(samples, logits_fn, batch_size, device, amp=amp, amp_dtype=amp_dtype)
        if domain == target_domain:
            results[f"{domain}_{target_bucket}"] = evaluate_logits(
                samples,
                logits_fn,
                batch_size,
                device,
                bucket=None if target_gap_threshold_days is not None else target_bucket,
                gap_threshold_days=target_gap_threshold_days,
                amp=amp,
                amp_dtype=amp_dtype,
            )
    return results


def metric_delta(current: dict[str, dict[str, float | int]], base: dict[str, dict[str, float | int]]) -> dict[str, float]:
    out = {}
    for key, values in current.items():
        if key not in base:
            continue
        for metric in ("hit10", "ndcg10"):
            out[f"{key}_{metric}_delta_vs_m0"] = float(values[metric]) - float(base[key][metric])
    return out


def state_snapshot(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def max_abs_drift(before: dict[str, torch.Tensor], module: nn.Module) -> float:
    drift = 0.0
    for name, value in module.state_dict().items():
        if name in before:
            drift = max(drift, float((value.detach().cpu() - before[name]).abs().max().item()))
    return drift


def empirical_hidden_prior(
    samples: TensorSamples,
    encoder: nn.Module,
    head: DomainItemHead,
    batch_size: int,
    device: torch.device,
    amp: bool,
    amp_dtype: str,
) -> torch.Tensor:
    total = torch.zeros(head.embedding_dim, device=device)
    count = 0
    encoder.eval()
    head.eval()
    with torch.no_grad():
        for sample_slice in batch_slices(len(samples), batch_size):
            batch = tensor_batch(samples, sample_slice, device)
            padding_mask = batch.seq.eq(0)
            with autocast_context(device, amp, amp_dtype):
                hidden = encoder(head.embed_sequence(batch.seq), padding_mask)
            total = total + hidden.float().sum(dim=0)
            count += int(hidden.shape[0])
    if count == 0:
        return torch.zeros(head.embedding_dim, device=device)
    return total / count


def l2_to_snapshot(
    modules: list[nn.Module],
    snapshots: list[dict[str, torch.Tensor]],
    device: torch.device,
) -> torch.Tensor:
    penalty = torch.zeros((), device=device)
    for module, snapshot in zip(modules, snapshots):
        for name, parameter in module.named_parameters():
            if name not in snapshot:
                continue
            reference = snapshot[name].to(device=device, dtype=parameter.dtype)
            penalty = penalty + torch.sum((parameter - reference) ** 2)
    return penalty


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def run_suite(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if args.matmul_precision:
        torch.set_float32_matmul_precision(args.matmul_precision)
    round2_contracts = round2_contract_tags(args)

    dataset = config["dataset"]
    processed_root = Path(dataset["processed_root"])
    target_domain = dataset["target_domain"]
    config_target_bucket = dataset["domains"][target_domain].get("primary_bucket", ">365d")
    target_gap_threshold_days = args.target_gap_threshold_days
    target_bucket = target_slice_label(config_target_bucket, target_gap_threshold_days, args.target_bucket_label)
    run_root = Path(config["run_root"])

    loaded = {}
    train_samples = {}
    eval_samples = {}
    domain_summary = {}
    for domain, domain_config in sorted(dataset["domains"].items()):
        data = load_processed(processed_root / domain_config["processed_file"])
        loaded[domain] = data
        train_bucket = config_target_bucket if domain == target_domain and args.prioritize_target_bucket else None
        train_gap_threshold = (
            target_gap_threshold_days
            if domain == target_domain and args.prioritize_target_bucket
            else None
        )
        selected_train = capped_samples(
            data["train_samples"],
            args.max_train_samples_per_domain,
            rng,
            train_bucket,
            gap_threshold_days=train_gap_threshold,
        )
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
            "target_slice_label": target_bucket if domain == target_domain else None,
            "target_gap_threshold_days": target_gap_threshold_days if domain == target_domain else None,
        }

    encoder = make_encoder(args).to(device)
    heads = {
        domain: DomainItemHead(int(data["item_vocab_size"]), embedding_dim=args.embedding_dim).to(device)
        for domain, data in loaded.items()
    }

    m0_optimizer = torch.optim.AdamW(
        [parameter for module in [encoder, *heads.values()] for parameter in module.parameters() if parameter.requires_grad],
        lr=args.lr,
    )
    m0_losses = []
    m0_started = time.time()
    for epoch in range(args.m0_epochs):
        epoch_losses = train_epoch(
            train_samples,
            encoder,
            heads,
            m0_optimizer,
            args.batch_size,
            device,
            args.amp,
            args.amp_dtype,
            args.m0_domain_schedule,
        )
        m0_losses.extend(epoch_losses)
        print(
            "PROGRESS "
            f"phase=m0 epoch={epoch + 1}/{args.m0_epochs} "
            f"schedule={args.m0_domain_schedule} steps={len(epoch_losses)} "
            f"loss_mean={sum(epoch_losses) / max(len(epoch_losses), 1):.6f} "
            f"elapsed_seconds={time.time() - m0_started:.1f}",
            flush=True,
        )

    encoder.eval()
    for head in heads.values():
        head.eval()
    if not m0_losses or not all(math.isfinite(value) for value in m0_losses):
        raise RuntimeError("M0 training produced missing or non-finite losses")
    m0_metrics = evaluate_all(
        eval_samples,
        encoder,
        heads,
        args.eval_batch_size,
        device,
        target_domain,
        target_bucket=target_bucket,
        target_gap_threshold_days=target_gap_threshold_days,
        amp=args.amp,
        amp_dtype=args.amp_dtype,
    )
    m0_param_report = trainable_parameter_report(nn.ModuleDict({"encoder": encoder, "heads": nn.ModuleDict(heads)}))

    if args.save_checkpoints:
        save_checkpoint(
            run_root / "checkpoints" / "m0_shared_sasrec_base.pt",
            {"encoder": encoder.state_dict(), "heads": {domain: head.state_dict() for domain, head in heads.items()}},
        )

    if args.stop_after_m0:
        return {
            "status": "pass",
            "errors": [],
            "run_type": args.run_type,
            "seed": args.seed,
            "device": str(device),
            "encoder_type": args.encoder_type,
            "target_domain": target_domain,
            "target_bucket": target_bucket,
            "retained_domains": [domain for domain in sorted(loaded) if domain != target_domain],
            "elapsed_seconds": time.time() - started,
            "stop_after_m0": True,
            "domain_summary": domain_summary,
            "m0_shared_sasrec_base": {
                "losses": m0_losses,
                "metrics": m0_metrics,
                "parameter_report": m0_param_report.__dict__,
                "checkpoint": str(run_root / "checkpoints" / "m0_shared_sasrec_base.pt")
                if args.save_checkpoints
                else None,
            },
        }

    m1_encoder = copy.deepcopy(encoder).to(device)
    m1_heads = {domain: copy.deepcopy(head).to(device) for domain, head in heads.items()}
    m1_encoder_before = state_snapshot(m1_encoder)
    time_model = TimeConcatTargetModel(m1_encoder, m1_heads[target_domain], args.embedding_dim).to(device)
    time_model.train()
    m1_losses = train_target_model(
        train_samples[target_domain],
        time_model,
        [parameter for parameter in time_model.parameters() if parameter.requires_grad],
        args.m1_epochs,
        args.batch_size,
        device,
        args.lr,
        args.amp,
        args.amp_dtype,
        "m1_timeconcat",
    )
    m1_encoder.eval()
    for head in m1_heads.values():
        head.eval()
    time_model.eval()
    m1_metrics = evaluate_all(
        eval_samples,
        m1_encoder,
        m1_heads,
        args.eval_batch_size,
        device,
        target_domain,
        target_time_model=time_model,
        target_bucket=target_bucket,
        target_gap_threshold_days=target_gap_threshold_days,
        amp=args.amp,
        amp_dtype=args.amp_dtype,
    )
    m1_param_report = trainable_parameter_report(time_model)
    m1_encoder_drift = max_abs_drift(m1_encoder_before, m1_encoder)

    if args.save_checkpoints:
        save_checkpoint(
            run_root / "checkpoints" / "m1_timeconcat_e2e_target_finetune.pt",
            {
                "encoder": m1_encoder.state_dict(),
                "heads": {domain: head.state_dict() for domain, head in m1_heads.items()},
                "time_model": time_model.state_dict(),
            },
        )

    m1_reg_losses: list[float] = []
    m1_reg_metrics: dict[str, dict[str, float | int]] | None = None
    m1_reg_param_report = None
    m1_reg_encoder_drift: float | None = None
    m1_reg_target_head_drift: float | None = None
    if args.include_regularized_timeconcat:
        m1_reg_encoder = copy.deepcopy(encoder).to(device)
        m1_reg_heads = {domain: copy.deepcopy(head).to(device) for domain, head in heads.items()}
        m1_reg_encoder_before = state_snapshot(m1_reg_encoder)
        m1_reg_target_head_before = state_snapshot(m1_reg_heads[target_domain])
        m1_reg_model = TimeConcatTargetModel(m1_reg_encoder, m1_reg_heads[target_domain], args.embedding_dim).to(device)
        m1_reg_model.train()
        m1_reg_epochs = args.m1_regularized_epochs if args.m1_regularized_epochs is not None else args.m1_epochs
        m1_reg_losses = train_target_model(
            train_samples[target_domain],
            m1_reg_model,
            [parameter for parameter in m1_reg_model.parameters() if parameter.requires_grad],
            m1_reg_epochs,
            args.batch_size,
            device,
            args.lr,
            args.amp,
            args.amp_dtype,
            "m1_l2sp_timeconcat",
            regularization_fn=lambda: args.m1_regularization_lambda
            * l2_to_snapshot(
                [m1_reg_encoder, m1_reg_heads[target_domain]],
                [m1_reg_encoder_before, m1_reg_target_head_before],
                device,
            ),
        )
        m1_reg_encoder.eval()
        for head in m1_reg_heads.values():
            head.eval()
        m1_reg_model.eval()
        m1_reg_metrics = evaluate_all(
            eval_samples,
            m1_reg_encoder,
            m1_reg_heads,
            args.eval_batch_size,
            device,
            target_domain,
            target_time_model=m1_reg_model,
            target_bucket=target_bucket,
            target_gap_threshold_days=target_gap_threshold_days,
            amp=args.amp,
            amp_dtype=args.amp_dtype,
        )
        m1_reg_param_report = trainable_parameter_report(m1_reg_model)
        m1_reg_encoder_drift = max_abs_drift(m1_reg_encoder_before, m1_reg_encoder)
        m1_reg_target_head_drift = max_abs_drift(m1_reg_target_head_before, m1_reg_heads[target_domain])

        if args.save_checkpoints:
            save_checkpoint(
                run_root / "checkpoints" / "m1_l2sp_timeconcat_target_finetune.pt",
                {
                    "encoder": m1_reg_encoder.state_dict(),
                    "heads": {domain: head.state_dict() for domain, head in m1_reg_heads.items()},
                    "time_model": m1_reg_model.state_dict(),
                    "regularization": {
                        "type": "l2_to_m0_encoder_and_target_head",
                        "lambda": args.m1_regularization_lambda,
                        "epochs": m1_reg_epochs,
                    },
                },
            )

    m1_frozen_losses: list[float] = []
    m1_frozen_metrics: dict[str, dict[str, float | int]] | None = None
    m1_frozen_param_report = None
    m1_frozen_encoder_drift: float | None = None
    m1_frozen_target_head_drift: float | None = None
    m1_frozen_retained_head_drift: dict[str, float] = {}
    if args.include_frozen_target_timeconcat:
        m1_frozen_encoder = copy.deepcopy(encoder).to(device)
        m1_frozen_heads = {domain: copy.deepcopy(head).to(device) for domain, head in heads.items()}
        m1_frozen_encoder_before = state_snapshot(m1_frozen_encoder)
        m1_frozen_target_head_before = state_snapshot(m1_frozen_heads[target_domain])
        m1_frozen_retained_head_before = {
            domain: state_snapshot(head)
            for domain, head in m1_frozen_heads.items()
            if domain != target_domain
        }
        freeze_module(m1_frozen_encoder)
        for domain, head in m1_frozen_heads.items():
            if domain != target_domain:
                freeze_module(head)
        m1_frozen_model = TimeConcatTargetModel(
            m1_frozen_encoder,
            m1_frozen_heads[target_domain],
            args.embedding_dim,
        ).to(device)
        m1_frozen_model.train()
        m1_frozen_encoder.eval()
        for domain, head in m1_frozen_heads.items():
            if domain != target_domain:
                head.eval()
        m1_frozen_eval_modules = [] if args.encoder_type == "gru" else [m1_frozen_encoder]
        m1_frozen_losses = train_target_model(
            train_samples[target_domain],
            m1_frozen_model,
            [parameter for parameter in m1_frozen_model.parameters() if parameter.requires_grad],
            args.m1_frozen_target_epochs if args.m1_frozen_target_epochs is not None else args.m1_epochs,
            args.batch_size,
            device,
            args.lr,
            args.amp,
            args.amp_dtype,
            "m1_frozen_target_timeconcat",
            eval_modules_during_training=m1_frozen_eval_modules,
        )
        m1_frozen_encoder.eval()
        for head in m1_frozen_heads.values():
            head.eval()
        m1_frozen_model.eval()
        m1_frozen_metrics = evaluate_all(
            eval_samples,
            m1_frozen_encoder,
            m1_frozen_heads,
            args.eval_batch_size,
            device,
            target_domain,
            target_time_model=m1_frozen_model,
            target_bucket=target_bucket,
            target_gap_threshold_days=target_gap_threshold_days,
            amp=args.amp,
            amp_dtype=args.amp_dtype,
        )
        m1_frozen_param_report = trainable_parameter_report(m1_frozen_model)
        m1_frozen_encoder_drift = max_abs_drift(m1_frozen_encoder_before, m1_frozen_encoder)
        m1_frozen_target_head_drift = max_abs_drift(m1_frozen_target_head_before, m1_frozen_heads[target_domain])
        m1_frozen_retained_head_drift = {
            domain: max_abs_drift(snapshot, m1_frozen_heads[domain])
            for domain, snapshot in m1_frozen_retained_head_before.items()
        }

        if args.save_checkpoints:
            save_checkpoint(
                run_root / "checkpoints" / "m1_frozen_target_timeconcat.pt",
                {
                    "encoder": m1_frozen_encoder.state_dict(),
                    "heads": {domain: head.state_dict() for domain, head in m1_frozen_heads.items()},
                    "time_model": m1_frozen_model.state_dict(),
                    "surface": {
                        "type": "frozen_shared_encoder_target_head_timeconcat",
                        "trainable": "target_domain_head_and_time_projection",
                        "frozen": "shared_encoder_and_retained_domain_heads",
                    },
                },
            )

    m1_lora_losses: list[float] = []
    m1_lora_metrics: dict[str, dict[str, float | int]] | None = None
    m1_lora_param_report = None
    m1_lora_activity: dict[str, dict[str, float | int]] = {}
    m1_lora_encoder_drift: float | None = None
    m1_lora_target_head_drift: float | None = None
    m1_lora_retained_head_drift: dict[str, float] = {}
    if args.include_lora_target_adapter:
        m1_lora_encoder = copy.deepcopy(encoder).to(device)
        m1_lora_heads = {domain: copy.deepcopy(head).to(device) for domain, head in heads.items()}
        m1_lora_encoder_before = state_snapshot(m1_lora_encoder)
        m1_lora_target_head_before = state_snapshot(m1_lora_heads[target_domain])
        m1_lora_retained_head_before = {
            domain: state_snapshot(head)
            for domain, head in m1_lora_heads.items()
            if domain != target_domain
        }
        freeze_module(m1_lora_encoder)
        for head in m1_lora_heads.values():
            freeze_module(head)
        m1_lora_model = FrozenLoRATargetAdapter(
            m1_lora_encoder,
            m1_lora_heads[target_domain],
            args.embedding_dim,
            rank=args.m1_lora_rank,
            alpha=args.m1_lora_alpha,
        ).to(device)
        m1_lora_model.train()
        m1_lora_encoder.eval()
        for head in m1_lora_heads.values():
            head.eval()
        m1_lora_activity_parameters = {
            name: parameter
            for name, parameter in m1_lora_model.named_parameters()
            if parameter.requires_grad
        }
        m1_lora_losses = train_target_model(
            train_samples[target_domain],
            m1_lora_model,
            [parameter for parameter in m1_lora_model.parameters() if parameter.requires_grad],
            args.m1_lora_epochs if args.m1_lora_epochs is not None else args.m1_epochs,
            args.batch_size,
            device,
            args.lr,
            args.amp,
            args.amp_dtype,
            "m1_frozen_lora_target_adapter",
            eval_modules_during_training=[m1_lora_encoder, *m1_lora_heads.values()],
            activity_parameters=m1_lora_activity_parameters,
            parameter_activity=m1_lora_activity,
        )
        m1_lora_encoder.eval()
        for head in m1_lora_heads.values():
            head.eval()
        m1_lora_model.eval()
        m1_lora_metrics = evaluate_all(
            eval_samples,
            m1_lora_encoder,
            m1_lora_heads,
            args.eval_batch_size,
            device,
            target_domain,
            target_time_model=m1_lora_model,
            target_bucket=target_bucket,
            target_gap_threshold_days=target_gap_threshold_days,
            amp=args.amp,
            amp_dtype=args.amp_dtype,
        )
        m1_lora_param_report = trainable_parameter_report(m1_lora_model)
        m1_lora_encoder_drift = max_abs_drift(m1_lora_encoder_before, m1_lora_encoder)
        m1_lora_target_head_drift = max_abs_drift(m1_lora_target_head_before, m1_lora_heads[target_domain])
        m1_lora_retained_head_drift = {
            domain: max_abs_drift(snapshot, m1_lora_heads[domain])
            for domain, snapshot in m1_lora_retained_head_before.items()
        }

        if args.save_checkpoints:
            save_checkpoint(
                run_root / "checkpoints" / "m1_frozen_lora_target_adapter.pt",
                {
                    "encoder": m1_lora_encoder.state_dict(),
                    "heads": {domain: head.state_dict() for domain, head in m1_lora_heads.items()},
                    "lora_adapter": m1_lora_model.state_dict(),
                    "surface": {
                        "type": "frozen_target_side_lora_residual",
                        "trainable": "low_rank_target_side_residual",
                        "frozen": "shared_encoder_and_all_domain_heads",
                        "rank": args.m1_lora_rank,
                        "alpha": args.m1_lora_alpha,
                    },
                },
            )

    m2_encoder = copy.deepcopy(encoder).to(device)
    m2_heads = {domain: copy.deepcopy(head).to(device) for domain, head in heads.items()}
    m2_encoder_before = state_snapshot(m2_encoder)
    m2_target_head_before = state_snapshot(m2_heads[target_domain])
    freeze_module(m2_encoder)
    for head in m2_heads.values():
        freeze_module(head)
    adapter = SharedBackboneDeltaGateAdapter(m2_encoder, m2_heads[target_domain], embedding_dim=args.embedding_dim).to(device)
    adapter.train()
    m2_encoder.eval()
    for head in m2_heads.values():
        head.eval()
    m2_frozen_modules_eval_during_training = not m2_encoder.training and all(not head.training for head in m2_heads.values())
    m2_param_report_before = trainable_parameter_report(adapter)
    m2_losses = train_target_model(
        train_samples[target_domain],
        lambda seq, delta_t: adapter(seq, delta_t)[0],
        [parameter for parameter in adapter.parameters() if parameter.requires_grad],
        args.m2_epochs,
        args.batch_size,
        device,
        args.lr,
        args.amp,
        args.amp_dtype,
        "m2_deltagate_adapter",
    )
    m2_encoder.eval()
    for head in m2_heads.values():
        head.eval()
    adapter.eval()
    m2_metrics = evaluate_all(
        eval_samples,
        m2_encoder,
        m2_heads,
        args.eval_batch_size,
        device,
        target_domain,
        target_adapter=adapter,
        target_bucket=target_bucket,
        target_gap_threshold_days=target_gap_threshold_days,
        amp=args.amp,
        amp_dtype=args.amp_dtype,
    )
    m2_param_report_after = trainable_parameter_report(adapter)
    m2_encoder_drift = max_abs_drift(m2_encoder_before, m2_encoder)
    m2_target_head_drift = max_abs_drift(m2_target_head_before, m2_heads[target_domain])

    if args.save_checkpoints:
        save_checkpoint(
            run_root / "checkpoints" / "m2_frozen_deltagate_adapter.pt",
            {
                "encoder": m2_encoder.state_dict(),
                "heads": {domain: head.state_dict() for domain, head in m2_heads.items()},
                "adapter": adapter.state_dict(),
            },
        )

    m2_trainable_ok = all(
        name == "global_prior" or name.startswith("gate_mlp.")
        for name in m2_param_report_after.trainable_names
    )

    m2_emp_losses: list[float] = []
    m2_emp_metrics: dict[str, dict[str, float | int]] | None = None
    m2_emp_param_report_before = None
    m2_emp_param_report_after = None
    m2_emp_trainable_ok: bool | None = None
    m2_emp_encoder_drift: float | None = None
    m2_emp_target_head_drift: float | None = None
    empirical_prior_norm: float | None = None
    if args.include_empirical_prior_deltagate:
        m2_emp_encoder = copy.deepcopy(encoder).to(device)
        m2_emp_heads = {domain: copy.deepcopy(head).to(device) for domain, head in heads.items()}
        m2_emp_encoder_before = state_snapshot(m2_emp_encoder)
        m2_emp_target_head_before = state_snapshot(m2_emp_heads[target_domain])
        freeze_module(m2_emp_encoder)
        for head in m2_emp_heads.values():
            freeze_module(head)
        empirical_adapter = SharedBackboneDeltaGateAdapter(
            m2_emp_encoder,
            m2_emp_heads[target_domain],
            embedding_dim=args.embedding_dim,
        ).to(device)
        with torch.no_grad():
            prior = empirical_hidden_prior(
                train_samples[target_domain],
                m2_emp_encoder,
                m2_emp_heads[target_domain],
                args.eval_batch_size,
                device,
                args.amp,
                args.amp_dtype,
            )
            empirical_adapter.global_prior.copy_(prior.to(device=device, dtype=empirical_adapter.global_prior.dtype))
            empirical_prior_norm = float(empirical_adapter.global_prior.detach().float().norm().cpu())
        empirical_adapter.train()
        m2_emp_encoder.eval()
        for head in m2_emp_heads.values():
            head.eval()
        m2_emp_param_report_before = trainable_parameter_report(empirical_adapter)
        m2_emp_losses = train_target_model(
            train_samples[target_domain],
            lambda seq, delta_t: empirical_adapter(seq, delta_t)[0],
            [parameter for parameter in empirical_adapter.parameters() if parameter.requires_grad],
            args.m2_empirical_prior_epochs if args.m2_empirical_prior_epochs is not None else args.m2_epochs,
            args.batch_size,
            device,
            args.lr,
            args.amp,
            args.amp_dtype,
            "m2_empirical_prior_deltagate_adapter",
        )
        m2_emp_encoder.eval()
        for head in m2_emp_heads.values():
            head.eval()
        empirical_adapter.eval()
        m2_emp_metrics = evaluate_all(
            eval_samples,
            m2_emp_encoder,
            m2_emp_heads,
            args.eval_batch_size,
            device,
            target_domain,
            target_adapter=empirical_adapter,
            target_bucket=target_bucket,
            target_gap_threshold_days=target_gap_threshold_days,
            amp=args.amp,
            amp_dtype=args.amp_dtype,
        )
        m2_emp_param_report_after = trainable_parameter_report(empirical_adapter)
        m2_emp_encoder_drift = max_abs_drift(m2_emp_encoder_before, m2_emp_encoder)
        m2_emp_target_head_drift = max_abs_drift(m2_emp_target_head_before, m2_emp_heads[target_domain])
        m2_emp_trainable_ok = all(
            name == "global_prior" or name.startswith("gate_mlp.")
            for name in m2_emp_param_report_after.trainable_names
        )
    finite_losses = all(
        math.isfinite(loss)
        for loss in [
            *m0_losses,
            *m1_losses,
            *m1_reg_losses,
            *m1_frozen_losses,
            *m1_lora_losses,
            *m2_losses,
            *m2_emp_losses,
        ]
    )

    status = "pass"
    errors = []
    if not finite_losses:
        status = "fail"
        errors.append("non-finite loss observed")
    if not m2_trainable_ok:
        status = "fail"
        errors.append(f"M2 unexpected trainable names: {m2_param_report_after.trainable_names}")
    if m2_encoder_drift != 0.0:
        status = "fail"
        errors.append(f"M2 encoder drifted: {m2_encoder_drift}")
    if m2_target_head_drift != 0.0:
        status = "fail"
        errors.append(f"M2 target head drifted: {m2_target_head_drift}")
    if args.include_frozen_target_timeconcat:
        if m1_frozen_encoder_drift != 0.0:
            status = "fail"
            errors.append(f"M1 frozen TimeConcat encoder drifted: {m1_frozen_encoder_drift}")
        nonzero_retained = {
            domain: drift
            for domain, drift in m1_frozen_retained_head_drift.items()
            if drift != 0.0
        }
        if nonzero_retained:
            status = "fail"
            errors.append(f"M1 frozen TimeConcat retained heads drifted: {nonzero_retained}")
    if args.include_lora_target_adapter:
        lora_trainable_ok = (
            m1_lora_param_report is not None
            and all(
                name.startswith("lora_down.") or name.startswith("lora_up.")
                for name in m1_lora_param_report.trainable_names
            )
        )
        if not lora_trainable_ok:
            status = "fail"
            errors.append(
                "M1 frozen LoRA unexpected trainable names: "
                f"{m1_lora_param_report.trainable_names if m1_lora_param_report else None}"
            )
        if m1_lora_encoder_drift != 0.0:
            status = "fail"
            errors.append(f"M1 frozen LoRA encoder drifted: {m1_lora_encoder_drift}")
        if m1_lora_target_head_drift != 0.0:
            status = "fail"
            errors.append(f"M1 frozen LoRA target head drifted: {m1_lora_target_head_drift}")
        nonzero_lora_retained = {
            domain: drift
            for domain, drift in m1_lora_retained_head_drift.items()
            if drift != 0.0
        }
        if nonzero_lora_retained:
            status = "fail"
            errors.append(f"M1 frozen LoRA retained heads drifted: {nonzero_lora_retained}")
        if args.round2_contract_checks:
            for name in ("lora_down.weight", "lora_up.weight"):
                activity = m1_lora_activity.get(name)
                if activity is None:
                    status = "fail"
                    errors.append(f"M1 frozen LoRA missing activity audit for {name}")
                    continue
                if float(activity.get("grad_l2_sum", 0.0)) <= 0.0 or int(activity.get("steps_with_nonzero_grad", 0)) <= 0:
                    status = "fail"
                    errors.append(f"M1 frozen LoRA {name} received no nonzero gradient: {activity}")
                if float(activity.get("update_l2_sum", 0.0)) <= 0.0 or int(activity.get("steps_with_update", 0)) <= 0:
                    status = "fail"
                    errors.append(f"M1 frozen LoRA {name} received no parameter update: {activity}")
    if args.include_empirical_prior_deltagate:
        if not m2_emp_trainable_ok:
            status = "fail"
            errors.append(
                "M2 empirical-prior unexpected trainable names: "
                f"{m2_emp_param_report_after.trainable_names if m2_emp_param_report_after else None}"
            )
        if m2_emp_encoder_drift != 0.0:
            status = "fail"
            errors.append(f"M2 empirical-prior encoder drifted: {m2_emp_encoder_drift}")
        if m2_emp_target_head_drift != 0.0:
            status = "fail"
            errors.append(f"M2 empirical-prior target head drifted: {m2_emp_target_head_drift}")

    report = {
        "status": status,
        "errors": errors,
        "run_type": args.run_type,
        "seed": args.seed,
        "device": str(device),
        "encoder_type": args.encoder_type,
        "target_domain": target_domain,
        "target_bucket": target_bucket,
        "target_gap_threshold_days": target_gap_threshold_days,
        "config_target_bucket": config_target_bucket,
        "retained_domains": [domain for domain in sorted(loaded) if domain != target_domain],
        "elapsed_seconds": time.time() - started,
        "limits": {
            "max_train_samples_per_domain": args.max_train_samples_per_domain,
            "max_eval_samples_per_domain": args.max_eval_samples_per_domain,
            "m0_epochs": args.m0_epochs,
            "m1_epochs": args.m1_epochs,
            "m1_regularized_epochs": args.m1_regularized_epochs,
            "m1_regularization_lambda": args.m1_regularization_lambda,
            "include_regularized_timeconcat": args.include_regularized_timeconcat,
            "include_frozen_target_timeconcat": args.include_frozen_target_timeconcat,
            "m1_frozen_target_epochs": args.m1_frozen_target_epochs,
            "include_lora_target_adapter": args.include_lora_target_adapter,
            "m1_lora_epochs": args.m1_lora_epochs,
            "m1_lora_rank": args.m1_lora_rank,
            "m1_lora_alpha": args.m1_lora_alpha,
            "round2_contract_checks": args.round2_contract_checks,
            "m2_epochs": args.m2_epochs,
            "include_empirical_prior_deltagate": args.include_empirical_prior_deltagate,
            "m2_empirical_prior_epochs": args.m2_empirical_prior_epochs,
            "target_bucket_label": args.target_bucket_label,
            "target_gap_threshold_days": args.target_gap_threshold_days,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "m0_domain_schedule": args.m0_domain_schedule,
            "encoder_type": args.encoder_type,
            "embedding_dim": args.embedding_dim,
            "num_layers": args.num_layers,
            "max_seq_len": args.max_seq_len,
            "amp": args.amp,
            "amp_dtype": args.amp_dtype,
            "pin_memory": args.pin_memory,
            "matmul_precision": args.matmul_precision,
        },
        "experiment_contracts": round2_contracts,
        "domain_summary": domain_summary,
        "m0_shared_sasrec_base": {
            "losses": m0_losses,
            "metrics": m0_metrics,
            "parameter_report": m0_param_report.__dict__,
        },
        "m1_timeconcat_e2e_target_finetune": {
            "losses": m1_losses,
            "metrics": m1_metrics,
            "delta_vs_m0": metric_delta(m1_metrics, m0_metrics),
            "parameter_report": m1_param_report.__dict__,
            "encoder_max_abs_drift": m1_encoder_drift,
        },
        "m2_frozen_deltagate_adapter": {
            "losses": m2_losses,
            "metrics": m2_metrics,
            "delta_vs_m0": metric_delta(m2_metrics, m0_metrics),
            "parameter_report_before": m2_param_report_before.__dict__,
            "parameter_report_after": m2_param_report_after.__dict__,
            "trainable_name_ok": m2_trainable_ok,
            "frozen_modules_eval_during_training": m2_frozen_modules_eval_during_training,
            "encoder_max_abs_drift": m2_encoder_drift,
            "target_head_max_abs_drift": m2_target_head_drift,
        },
    }
    if args.include_empirical_prior_deltagate:
        report["m2_empirical_prior_deltagate_adapter"] = {
            "losses": m2_emp_losses,
            "metrics": m2_emp_metrics,
            "delta_vs_m0": metric_delta(m2_emp_metrics or {}, m0_metrics),
            "delta_vs_zero_init_deltagate": metric_delta(m2_emp_metrics or {}, m2_metrics),
            "parameter_report_before": m2_emp_param_report_before.__dict__ if m2_emp_param_report_before else {},
            "parameter_report_after": m2_emp_param_report_after.__dict__ if m2_emp_param_report_after else {},
            "trainable_name_ok": m2_emp_trainable_ok,
            "encoder_max_abs_drift": m2_emp_encoder_drift,
            "target_head_max_abs_drift": m2_emp_target_head_drift,
            "empirical_prior_norm": empirical_prior_norm,
            "surface": {
                "type": "empirical_average_prior_deltagate_ablation",
                "prior_init": "mean frozen target hidden state over target training slice",
                "trainable": "gate_mlp_and_global_prior_after_empirical_initialization",
            },
        }
    if args.include_regularized_timeconcat:
        report["m1_l2sp_timeconcat_target_finetune"] = {
            "losses": m1_reg_losses,
            "metrics": m1_reg_metrics,
            "delta_vs_m0": metric_delta(m1_reg_metrics or {}, m0_metrics),
            "parameter_report": m1_reg_param_report.__dict__ if m1_reg_param_report is not None else {},
            "encoder_max_abs_drift": m1_reg_encoder_drift,
            "target_head_max_abs_drift": m1_reg_target_head_drift,
            "regularization": {
                "type": "l2_to_m0_encoder_and_target_head",
                "lambda": args.m1_regularization_lambda,
                "epochs": args.m1_regularized_epochs if args.m1_regularized_epochs is not None else args.m1_epochs,
            },
        }
    if args.include_frozen_target_timeconcat:
        report["m1_frozen_target_timeconcat"] = {
            "losses": m1_frozen_losses,
            "metrics": m1_frozen_metrics,
            "delta_vs_m0": metric_delta(m1_frozen_metrics or {}, m0_metrics),
            "parameter_report": m1_frozen_param_report.__dict__ if m1_frozen_param_report is not None else {},
            "encoder_max_abs_drift": m1_frozen_encoder_drift,
            "target_head_max_abs_drift": m1_frozen_target_head_drift,
            "retained_head_max_abs_drift": m1_frozen_retained_head_drift,
            "surface": {
                "type": "frozen_shared_encoder_target_head_timeconcat",
                "trainable": "target_domain_head_and_time_projection",
                "frozen": "shared_encoder_and_retained_domain_heads",
            },
        }
    if args.include_lora_target_adapter:
        report["m1_frozen_lora_target_adapter"] = {
            "losses": m1_lora_losses,
            "metrics": m1_lora_metrics,
            "delta_vs_m0": metric_delta(m1_lora_metrics or {}, m0_metrics),
            "parameter_report": m1_lora_param_report.__dict__ if m1_lora_param_report is not None else {},
            "parameter_activity": m1_lora_activity,
            "encoder_max_abs_drift": m1_lora_encoder_drift,
            "target_head_max_abs_drift": m1_lora_target_head_drift,
            "retained_head_max_abs_drift": m1_lora_retained_head_drift,
            "surface": {
                "type": "frozen_target_side_lora_residual",
                "rank": args.m1_lora_rank,
                "alpha": args.m1_lora_alpha,
                "trainable": "lora_down_and_lora_up_only",
                "frozen": "shared_encoder_and_all_domain_heads",
            },
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/shared_backbone_seed42_protocol.json"))
    parser.add_argument("--json-out", type=Path, default=Path("reports/shared_backbone_seed42_runner_smoke.json"))
    parser.add_argument("--run-type", default="bounded_smoke", choices=["bounded_smoke", "seed42", "capped_evidence"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--encoder-type", choices=["sasrec", "gru"], default="sasrec")
    parser.add_argument("--max-train-samples-per-domain", type=int, default=32)
    parser.add_argument("--max-eval-samples-per-domain", type=int, default=32)
    parser.add_argument("--prioritize-target-bucket", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--m0-epochs", type=int, default=1)
    parser.add_argument("--m1-epochs", type=int, default=1)
    parser.add_argument("--include-regularized-timeconcat", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--m1-regularization-lambda", type=float, default=1e-4)
    parser.add_argument("--m1-regularized-epochs", type=int, default=None)
    parser.add_argument("--include-frozen-target-timeconcat", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--m1-frozen-target-epochs", type=int, default=None)
    parser.add_argument("--include-lora-target-adapter", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--m1-lora-epochs", type=int, default=None)
    parser.add_argument("--m1-lora-rank", type=int, default=8)
    parser.add_argument("--m1-lora-alpha", type=float, default=8.0)
    parser.add_argument("--round2-contract-checks", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--m2-epochs", type=int, default=1)
    parser.add_argument("--include-empirical-prior-deltagate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--m2-empirical-prior-epochs", type=int, default=None)
    parser.add_argument("--target-gap-threshold-days", type=float, default=None)
    parser.add_argument("--target-bucket-label", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--m0-domain-schedule", choices=["max_cycle", "once"], default="max_cycle")
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
    parser.add_argument("--save-checkpoints", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stop-after-m0", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    if args.run_type == "seed42" and (
        args.max_train_samples_per_domain > 0 or args.max_eval_samples_per_domain > 0
    ):
        parser.error(
            "--run-type seed42 requires --max-train-samples-per-domain 0 "
            "and --max-eval-samples-per-domain 0; use bounded_smoke for capped runs"
        )

    config = load_json(args.config)
    report = run_suite(config, args)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(text + "\n", encoding="utf-8")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
