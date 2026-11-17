"""Fit the original objectives and seal each horizon's validation-selected state."""
import argparse
import io
import math
from pathlib import Path
import time

import numpy as np
from torch.nn import functional as F

from paths import (CORE, FOLLOWUP, GROUPS, HERE, INPUTS_PATH, PLAN, PLAN_PATH, RUNS,
                   SEEDS, YELP, RESTAURANTS, methods, read, require, sha)
from runtime import DEVICE, configure, inventory, module, parent_path, rows_for, torch, verified_file

HORIZONS = PLAN["horizons"]
STRICT = set(PLAN["exact_methods"]) - {"S0"}


def is_replay(method):
    return method == "Replay" or method.startswith("Distill_")


def make_model(train, group, method, counts, seed, cutoff, parent, for_training=False):
    target = GROUPS[group]["target"]
    backbone_variant = "S8" if method == "S7" or is_replay(method) else method
    replay = None
    factory = train.model_for
    if for_training and is_replay(method):
        directory = "replay_vg_20261111" if target == "video_games" else "cds_replay_s7_20261112"
        replay = module("archived_replay_epoch", CORE / directory / "replay.py")
        if GROUPS[group]["source"] == "yelp":
            plan = read((YELP if target == "shopping" else RESTAURANTS) / "PLAN.json")
            replay.TARGET, replay.RETAINED = target, tuple(plan["retained"])
            replay.DOMAINS = (replay.TARGET, *replay.RETAINED)
            require(sha(Path(replay.__file__)) == plan["replay_source_sha256"], "Yelp's replay kernel changed")
        require(replay.TARGET == target and replay.PLAN["batch_size"] == PLAN["batch_size"]
                and replay.PLAN["replay_loss_weight"] == 1.0, "Unexpected replay configuration")
        if method.startswith("Distill_"):
            followup = module("archived_distillation", FOLLOWUP / "run.py")
            followup.install_distillation(replay, float(method.split("_")[1]))
    try:
        model = train.model_for(backbone_variant, counts, seed, cutoff, parent).to(DEVICE)
    finally:
        train.model_for = factory
    if for_training:
        return model, replay
    return model


def standard_epoch(train, group, seed, method, rows, model, optimizer, epoch, batch_size, penalty):
    target = GROUPS[group]["target"]
    model.train()
    if method in STRICT:
        model.base.eval()
    order = np.random.default_rng(seed + (epoch - 1) * 8191).permutation(len(rows["target"]))
    losses = torch.zeros((), device=DEVICE)
    steps = 0
    for index in train.slices(len(order), batch_size):
        batch = train.batch_from(rows, order[index], DEVICE)
        optimizer.zero_grad(set_to_none=True)
        with train.amp(DEVICE):
            loss = F.cross_entropy(model(target, batch), batch["target"])
            if penalty is not None:
                loss = loss + penalty()
        loss.backward()
        optimizer.step()
        losses.add_(loss.detach())
        steps += 1
    return {"epoch": epoch, "steps": steps, "loss": float(losses) / steps}


def equals(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a.cpu(), b.cpu())
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equals(v, b[k]) for k, v in a.items())
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(equals(x, y) for x, y in zip(a, b))
    return a == b


def milestone(best, path):
    return {"epoch": best["epoch"], "val": best["metrics"],
            "weights": {"path": str(path), "sha256": sha(path)}}


def store_best(train, best, folder, saved):
    require(best is not None, "No checkpoint selected by validation")
    for record in saved.values():
        if record["epoch"] == best["epoch"]:
            return record
    path = folder / f"best-{best['epoch']}.pt"
    train.atomic(path, lambda tmp: torch.save({"model": best["model"]}, tmp))
    return milestone(best, path)


def import_state(train, source, folder, method, runtime):
    start = source["start"]
    if not start:
        return None, None, [], {}, 0.0, 0.0
    old = read(verified_file(source["fit10"]))
    selected10 = verified_file(source["best10"])
    state = torch.load(verified_file(source["state"]), map_location="cpu", weights_only=True)
    current_best = state["best"]
    selected = {"10": {"epoch": old["selected_epoch"], "val": old["validation"], "weights": source["best10"]}}
    history = old["epochs"]
    trained = old.get("train_seconds", old.get("training_seconds"))
    val_seconds = old["validation_seconds"]
    if start == 10:
        contract = state["contract"]
        require(contract == old["contract"] and (contract.get("torch_version") in (None, str(torch.__version__)))
                and state.get("epoch", len(state.get("rows", ()))) == 10
                and state.get("epoch_log", state.get("rows")) == history
                and current_best["epoch"] == old["selected_epoch"]
                and current_best["metrics"] == old["validation"], "Original run does not end at the bound epoch")
        require(equals(current_best["model"], torch.load(selected10, map_location="cpu", weights_only=True)["model"]),
                "Archive's selected weights differ from its resume state")
        if contract.get("runtime") and "runtime" in runtime:
            require(contract["runtime"] == runtime["runtime"], "Cross-backbone optimizer runtime changed")
    else:
        last = read(verified_file(source["fit50"]))
        require(start == 50 and state["epoch"] == start and state["binding"] == last["binding"]
                and current_best["epoch"] == last["selected_epoch"]
                and current_best["metrics"] == last["validation"] and last["curve"] == state["curve"],
                "Bound 50-epoch resume state changed")
        history = [p for p in history if p["epoch"] < 10] + state["curve"]
        trained += state["extension_train_seconds"]
        val_seconds += state["extension_validation_seconds"]
        selected["50"] = store_best(train, current_best, folder, selected)
    require(state["cuda_rng"] is not None and len(history) == start, "Original checkpoint lacks RNG or epochs")
    return state, current_best, history, selected, trained, val_seconds


def fit(group, seed, method, phase):
    train, runtime = configure(group)
    recipe = GROUPS[group]
    target = recipe["target"]
    parent, parent_sha = parent_path(group, seed)
    tick = time.monotonic()
    probe = phase != "fit"
    source = {"start": 0} if probe else inventory()["fits"][group][str(seed)][method]
    scopes = (HERE / "canary" / ("resumed" if phase in ("interrupt", "resume") else phase)) if probe else RUNS
    folder = scopes / group / str(seed) / method
    folder.mkdir(parents=True, exist_ok=True)
    receipt, state_path = folder / "FIT.json", folder / "resume.pt"
    milestones = [2] if probe else HORIZONS
    cap = max(milestones)
    size = 128 if probe else PLAN["batch_size"]
    bound = {"group": group, "seed": seed, "method": method, "runtime": runtime, "probe": probe,
             "plan": sha(PLAN_PATH), "inputs": sha(INPUTS_PATH), "parent": parent_sha,
             "source": source, "worker": sha(Path(__file__)), "horizons": milestones}
    if receipt.exists():
        previous = read(receipt)
        require(previous["binding"] == bound and previous["resume_sha"] == sha(state_path),
                "Completed adaptation result changed")
        return
    rows_needed = None if is_replay(method) else {target}
    counts, data, cutoff, _ = rows_for(group, "fit", rows_needed, small=probe)
    model, replay = make_model(train, group, method, counts, seed, cutoff, parent, for_training=True)
    penalty = None
    if method == "S7":
        from scripts.train_shared_backbone_seed42 import l2_to_snapshot, state_snapshot
        modules = [model.base.encoder, model.base.heads[target]]
        snapshots = [{k: v.to(DEVICE) for k, v in state_snapshot(part).items()} for part in modules]

        def penalty():
            return PLAN["reg_lambda"] * l2_to_snapshot(modules, snapshots, DEVICE)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=PLAN["learning_rates"][method])
    if state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        require(state["binding"] == bound, "Adaptation checkpoint binding changed")
        best, history, selected = state["best"], state["history"], state["selected"]
        trained, val_seconds = state["train_seconds"], state["validation_seconds"]
    else:
        state, best, history, selected, trained, val_seconds = import_state(train, source, folder, method, runtime)
    done = len(history)
    require(phase != "resume" or done == 1, "Canary did not stop at epoch one")
    if state is not None:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["cpu_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"], device=DEVICE)
        if probe:
            require(equals(model.state_dict(), state["model"])
                    and equals(optimizer.state_dict(), state["optimizer"])
                    and equals(torch.get_rng_state(), state["cpu_rng"])
                    and equals(torch.cuda.get_rng_state(DEVICE), state["cuda_rng"]),
                    "CUDA interruption did not recover weights, optimizer and RNG")
    previous_save = time.monotonic()
    setup = previous_save - tick
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(done + 1, cap + 1):
        tick = time.monotonic()
        if replay is not None:
            row = replay.training_epoch(model, optimizer, data["train"], seed, epoch - 1, size, DEVICE)
        else:
            row = standard_epoch(train, group, seed, method, data["train"][target],
                                 model, optimizer, epoch, size, penalty)
            require(math.isfinite(row["loss"]), f"Nonfinite objective at epoch {epoch}")
        row["seconds"] = time.monotonic() - tick
        trained += row["seconds"]
        require(row["epoch"] == epoch and (probe or row["steps"] == recipe["steps"]),
                f"Wrong update count at epoch {epoch}")
        if epoch % PLAN["validation_interval"] == 0:
            tick = time.monotonic()
            metrics, *_ = train.evaluate(model, target, data["validation"][target], DEVICE, size)
            val_seconds += time.monotonic() - tick
            value = metrics["long_known"]
            require(value["ndcg10"] is not None and math.isfinite(value["ndcg10"])
                    and (probe or value["n"] == PLAN["data"][recipe["source"]]["long_covered_validation"][target]),
                    "Validation cohort or metric differs")
            row["ndcg10"] = value["ndcg10"]
            if best is None or row["ndcg10"] > best["score"]:
                best = {"epoch": epoch, "score": row["ndcg10"], "metrics": metrics,
                        "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        history.append(row)
        if epoch in milestones:
            selected[str(epoch)] = store_best(train, best, folder, selected)
        if epoch == done + 1 or epoch in milestones or time.monotonic() - previous_save >= PLAN["snapshot_seconds"]:
            checkpoint = {"binding": bound, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                          "best": best, "history": history, "selected": selected,
                          "train_seconds": trained, "validation_seconds": val_seconds,
                          "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(DEVICE)}
            train.atomic(state_path, lambda tmp: torch.save(checkpoint, tmp))
            previous_save = time.monotonic()
        print(json_line(row), flush=True)
        if phase == "interrupt":
            raise SystemExit(75)
    require(set(selected) == {str(h) for h in milestones}, "Missing horizon selection")
    update = io.BytesIO()
    torch.save({key: value.detach().cpu() for key, value in model.named_parameters() if value.requires_grad}, update)
    train.save_json(receipt, {"binding": bound, "epochs": len(history), "history": history, "selected": selected,
                             "resume_sha": sha(state_path), "update_bytes": update.tell(), "setup_seconds": setup,
                             "train_seconds": trained, "validation_seconds": val_seconds,
                             "peak_allocated_bytes": torch.cuda.max_memory_allocated()})


def json_line(row):
    import json
    return json.dumps(row, allow_nan=False)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("fit", "clean", "repeat", "interrupt", "resume"))
    cli.add_argument("group", choices=GROUPS)
    cli.add_argument("seed", type=int, choices=SEEDS)
    cli.add_argument("method")
    args = cli.parse_args()
    require(args.method in methods(args.group), "Method is outside the committed matrix")
    fit(args.group, args.seed, args.method, args.phase)


if __name__ == "__main__":
    main()
