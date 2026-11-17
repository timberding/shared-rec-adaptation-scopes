#!/usr/bin/env python3
"""Fit a bounded Replay comparison against the frozen CDs & Vinyl S8 run."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE))
import train

PLAN_PATH = HERE / "PLAN.json"
PLAN = json.loads(PLAN_PATH.read_text())
SEEDS = tuple(PLAN["seeds"])
TARGET = PLAN["target"]
RETAINED = tuple(PLAN["retained_domains"])
DOMAINS = (TARGET, *RETAINED)
RESULTS = HERE / "results"
DEVELOPMENT = BASE / "data/development"
SEALED = BASE / "data/test_sealed"
ORIGINAL_RUNS = BASE / "runs"


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def bind(path, value):
    if path.is_file():
        require(read(path) == value, f"Receipt changed: {path}")
    else:
        train.save_json(path, value)


def source_identity():
    return hashlib.sha256((train.code_fingerprint() + train.sha(PLAN_PATH)
                           + train.sha(Path(__file__))).encode()).hexdigest()


def provenance():
    freeze_path = ORIGINAL_RUNS / "freeze.json"
    require(train.sha(freeze_path) == PLAN["parent_freeze_sha256"], "Original freeze changed")
    freeze = read(freeze_path)
    recipe_path = ORIGINAL_RUNS / str(SEEDS[0]) / "recipe.json"
    require(train.sha(recipe_path) == freeze["recipe_sha"], "Original recipe changed")
    recipe = read(recipe_path)
    require(set(SEEDS) <= set(freeze["seeds"]), "Seed mismatch")
    train.TARGET = TARGET
    require(train.code_fingerprint() == recipe["source_sha"], "Original executable changed")
    require(recipe["lrs"]["S8"] == PLAN["learning_rate"], "S8 learning rate changed")
    require(recipe["development_sha"] == PLAN["development_manifest_sha256"],
            "Original development split changed")
    require(freeze["test_manifest_sha"] == PLAN["test_manifest_sha256"],
            "Original sealed split changed")
    return freeze, recipe


def source_data(root, windows, digest, recipe, limit=None):
    require(train.sha(root / "manifest.json") == digest, f"Manifest changed: {root}")
    counts, data, manifest = train.load_data(root, windows, limit)
    require(counts == recipe["catalog_sizes"]
            and all(manifest[k] == recipe[k] for k in ("b0_sha256", "cutoff_days_inclusive")),
            "Global split identity differs")
    return counts, data


def training_epoch(model, optimizer, rows, seed, epoch, size, device):
    model.train()
    target_n = len(rows[TARGET]["target"])
    target_order = np.random.default_rng(seed + epoch*8191).permutation(target_n)
    retained_orders = []
    for index, domain in enumerate(RETAINED):
        n = (target_n + 1 - index)//2
        rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, index, 44019]))
        pool = len(rows[domain]["target"])
        whole, remainder = divmod(n, pool)
        parts = [rng.permutation(pool) for _ in range(whole)]
        if remainder:
            parts.append(rng.choice(pool, remainder, replace=False))
        retained_orders.append(np.concatenate(parts) if parts else np.empty(0, dtype=np.int64))

    losses = torch.zeros(len(DOMAINS), device=device)
    offsets = [0, 0]
    steps = 0
    for index in train.slices(target_n, size):
        target_ids = target_order[index]
        optimizer.zero_grad(set_to_none=True)
        batch = train.batch_from(rows[TARGET], target_ids, device)
        with train.amp(device):
            loss = F.cross_entropy(model(TARGET, batch), batch["target"])
        loss.backward()
        losses[0].add_(loss.detach())
        del loss, batch

        for i, domain in enumerate(RETAINED):
            n = (len(target_ids) + 1 - i)//2
            if n == 0:
                continue
            ids = retained_orders[i][offsets[i]:offsets[i]+n]
            offsets[i] += n
            batch = train.batch_from(rows[domain], ids, device)
            with train.amp(device):
                loss = F.cross_entropy(model(domain, batch), batch["target"])
            (PLAN["replay_loss_weight"]*n/len(target_ids)*loss).backward()
            losses[i+1].add_(loss.detach())
            del loss, batch
        optimizer.step()
        steps += 1

    values = (losses / steps).tolist()
    require(all(math.isfinite(x) for x in values)
            and offsets == [len(x) for x in retained_orders], "Invalid Replay epoch")
    return {"epoch": epoch+1, "steps": steps, "losses": dict(zip(DOMAINS, values)),
            "exposures": dict(zip(DOMAINS, [target_n, *offsets])),
            "unique_retained_examples": {d: int(np.unique(ids).size)
                                         for d, ids in zip(RETAINED, retained_orders)}}


def trainable_state(model):
    return sorted(name for name, parameter in model.named_parameters() if parameter.requires_grad)


def fit(root, seed, mode, original, recipe):
    start = time.monotonic()
    out = root / str(seed)
    out.mkdir(parents=True, exist_ok=True)
    smoke = mode != "fit"
    parent = ORIGINAL_RUNS / str(seed) / "S0.pt"
    parent_sha = original["checkpoints"][str(seed)]["S0"]
    require(train.sha(parent) == parent_sha, f"Wrong parent for {seed}")
    epochs = 2 if smoke else PLAN["training_epochs"]
    limit = 2048 if smoke else None
    size = PLAN["batch_size"]
    digest = PLAN["development_manifest_sha256"]
    contract = {"seed": seed, "parent_sha": parent_sha, "source_sha": source_identity(),
                "development_sha": digest, "smoke": smoke, "epochs": epochs,
                "batch_size": size, "lr": PLAN["learning_rate"], "limit": limit}
    saved_model = out / "replay.pt"
    receipt = out / "FIT_COMPLETE.json"
    if receipt.exists():
        value = read(receipt)
        require(value["contract"] == contract and value["checkpoint_sha"] == train.sha(saved_model),
                "Completed model identity changed")
        return

    counts, data = source_data(DEVELOPMENT, ("train", "validation"), digest, recipe, limit)
    if not smoke:
        require({d: len(data["train"][d]["target"]) for d in DOMAINS} == PLAN["training_counts"],
                "Training population differs")
    device = torch.device("cuda")
    model = train.model_for("S8", counts, seed, recipe["cutoff_days_inclusive"][0], parent).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=contract["lr"])
    torch.cuda.reset_peak_memory_stats()
    state_path = out / "resume.pt"
    rows = []
    best = None
    training_seconds = validation_seconds = 0.0
    if state_path.is_file():
        state = torch.load(state_path, map_location=device, weights_only=True)
        require(state["contract"] == contract, "Resume contract changed")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["cpu_rng"].cpu())
        torch.cuda.set_rng_state(state["cuda_rng"].cpu(), device=device)
        rows, best = state["rows"], state["best"]
        training_seconds, validation_seconds = state["training_seconds"], state["validation_seconds"]
    require(mode != "resume_smoke" or len(rows) == 1, "Missing epoch-one canary checkpoint")
    for epoch in range(len(rows), epochs):
        tick = time.monotonic()
        row = training_epoch(model, optimizer, data["train"], seed, epoch, size, device)
        training_seconds += time.monotonic() - tick
        if smoke or (epoch+1) % 2 == 0:
            tick = time.monotonic()
            metrics, *_ = train.evaluate(model, TARGET, data["validation"][TARGET], device, size)
            validation_seconds += time.monotonic() - tick
            score = metrics["long_known"]["ndcg10"]
            if smoke and score is None:
                score = metrics["all_known"]["ndcg10"]
            require(score is not None and math.isfinite(score), "Invalid validation selection score")
            row["validation_ndcg10"] = score
            if best is None or score > best["score"]:
                best = {"score": score, "epoch": epoch+1, "metrics": metrics,
                        "model": {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}}
        rows.append(row)
        state = {"contract": contract, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(device),
                 "rows": rows, "best": best, "training_seconds": training_seconds,
                 "validation_seconds": validation_seconds}
        train.atomic(state_path, lambda tmp: torch.save(state, tmp))
        print(json.dumps(row), flush=True)
        if mode == "interrupt" and epoch == 0:
            raise SystemExit(75)

    train.atomic(saved_model, lambda tmp: torch.save({"model": best["model"]}, tmp))
    frozen_names = trainable_state(model)
    require(all(not name.startswith(tuple(f"base.heads.{d}." for d in RETAINED))
                for name in frozen_names), "Non-target head is trainable")
    train.save_json(receipt, {"contract": contract, "checkpoint_sha": train.sha(saved_model),
                            "selected_epoch": best["epoch"], "validation": best["metrics"],
                            "training_seconds": training_seconds, "validation_seconds": validation_seconds,
                            "worker_seconds": time.monotonic()-start,
                            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
                            "trainable_state": frozen_names, "epochs": rows,
                            "peak_allocated_bytes": torch.cuda.max_memory_allocated()})


def freeze(root, original):
    checkpoints = {}
    fits = {}
    for seed in SEEDS:
        out = root / str(seed)
        receipt = out / "FIT_COMPLETE.json"
        v = read(receipt)
        contract = v["contract"]
        expected = {"seed": seed, "source_sha": source_identity(), "smoke": False,
                    "parent_sha": original["checkpoints"][str(seed)]["S0"],
                    "development_sha": PLAN["development_manifest_sha256"],
                    "epochs": PLAN["training_epochs"], "batch_size": PLAN["batch_size"],
                    "lr": PLAN["learning_rate"], "limit": None}
        require(contract == expected and v["checkpoint_sha"] == train.sha(out/"replay.pt"),
                f"Unbound formal model for {seed}")
        require(v["selected_epoch"] in (2,4,6,8,10) and v["validation"]["long_known"]["n"] == PLAN["validation_long_known"],
                "Wrong validation population or schedule")
        require(len(v["epochs"]) == PLAN["training_epochs"] and all(
            row["steps"] == math.ceil(row["exposures"][TARGET]/PLAN["batch_size"])
            and row["exposures"][TARGET] == PLAN["training_counts"][TARGET]
            and all(row["exposures"][d] == (row["exposures"][TARGET] + 1 - i)//2
                    and row["unique_retained_examples"][d] == min(row["exposures"][d], PLAN["training_counts"][d])
                    for i,d in enumerate(RETAINED))
            for row in v["epochs"]), "Target/replay exposure certificate failed")
        checkpoints[str(seed)] = v["checkpoint_sha"]
        fits[str(seed)] = train.sha(receipt)
    bind(root/"freeze.json", {"seeds": list(SEEDS), "source_sha": source_identity(),
                             "test_manifest_sha": PLAN["test_manifest_sha256"],
                             "checkpoints": checkpoints, "fit_receipts": fits})


def test(root, seed, original, recipe):
    frozen_path = root/"freeze.json"
    require(frozen_path.is_file(), "Formal validation freeze required before test")
    freeze(root, original)
    frozen = read(frozen_path)
    out = root/str(seed)
    done = out/"TEST_COMPLETE.json"
    if done.is_file():
        require(read(done) == {"seed": seed, "freeze_sha": train.sha(frozen_path)},
                "Test completion freeze differs")
        return
    counts, data = source_data(SEALED, ("test",), frozen["test_manifest_sha"], recipe)
    parent = ORIGINAL_RUNS/str(seed)/"S0.pt"
    require(train.sha(parent) == original["checkpoints"][str(seed)]["S0"], "Test parent changed")
    model = train.model_for("S8", counts, seed, recipe["cutoff_days_inclusive"][0], parent)
    model.load_state_dict(torch.load(out/"replay.pt", map_location="cpu", weights_only=True)["model"])
    device = torch.device("cuda")
    model.to(device)
    metrics = {}
    for domain, rows in data["test"].items():
        metrics[domain], rank, _, top10 = train.evaluate(model, domain, rows, device, 512, save_top=True)
        path = out/f"test_replay_{domain}.npz"
        train.atomic(path, lambda tmp: save_predictions(tmp, rows, rank, top10))
    train.save_json(out/"test_replay.json", {"metrics": metrics,
                    "checkpoint_sha": frozen["checkpoints"][str(seed)], "test_data_sha": frozen["test_manifest_sha"]})
    train.save_json(done, {"seed": seed, "freeze_sha": train.sha(frozen_path)})


def save_predictions(path, rows, rank, top10):
    with path.open("wb") as stream:
        np.savez_compressed(stream, rank=rank, top10=top10, user=rows["user"].numpy(),
                            gap=rows["gap"].numpy(), known=rows["target"].numpy() >= 0,
                            untied=rows["untied"].numpy())


def verify_smoke(root, seed):
    clean, resumed = (root/folder/str(seed) for folder in ("clean", "resume"))
    for name in ("replay.pt", "resume.pt"):
        a, b = (torch.load(p/name, map_location="cpu", weights_only=True)["model"] for p in (clean, resumed))
        require(a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a),
                "Continuous and resumed model states differ")
    first, second = (read(p/"FIT_COMPLETE.json") for p in (clean, resumed))
    require(all(first[k] == second[k] for k in ("epochs", "validation", "selected_epoch"))
            and len(first["epochs"]) == 2, "Canary metrics differ")
    require(max(v["peak_allocated_bytes"] for v in (first, second)) <= 18*2**30,
            "Canary exceeded memory admission")
    bind(root/"PASS.json", {"source_sha": source_identity(), "seed": seed,
                           "resume_equal": True, "interrupted_epoch": 1, "finished_epoch": 2,
                           "peak_allocated_bytes": max(v["peak_allocated_bytes"] for v in (first, second))})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("sanity", "interrupt", "resume_smoke", "verify_smoke", "fit", "freeze", "test"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    torch.backends.mha.set_fastpath_enabled(False)
    torch.set_float32_matmul_precision("high")
    original, recipe = provenance()
    if args.phase == "freeze":
        freeze(args.root, original)
    elif args.phase == "test":
        test(args.root, args.seed, original, recipe)
    elif args.phase == "verify_smoke":
        verify_smoke(args.root, args.seed)
    else:
        fit(args.root, args.seed, args.phase, original, recipe)


if __name__ == "__main__":
    main()
