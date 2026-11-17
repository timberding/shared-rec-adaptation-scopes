"""Score the two original CDs shared objectives after exactly 220 updates."""

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch.optim.optimizer import register_optimizer_step_post_hook

import launch
from run import (CORE, HERE, PLAN, build_model, configure, load_data, read,
                 replay_module, require)


TARGET = "cds_vinyl"
DOMAIN = "movies_tv"
METHODS = ("S8", "Replay")
SEEDS = (9308, 42096, 169953)
STEPS = 220
RESULTS = HERE / "aligned"


class FixedHorizonReached(Exception):
    pass


def folder(seed, method):
    return RESULTS / str(seed) / method


def fit(seed, method):
    ctx = configure(TARGET, seed)
    train = ctx.train
    out = folder(seed, method)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "selected.pt"
    receipt = out / "FIT.json"
    signed = {"source": train.sha(Path(__file__)), "core": train.code_fingerprint(),
              "plan": train.sha(HERE / "PLAN.json"), "data_sha": PLAN["amazon_development"],
              "parent_sha": ctx.parent_sha, "steps": STEPS, "seed": seed, "method": method}
    if receipt.is_file():
        require(read(receipt) == {**signed, "weight_sha": train.sha(path)},
                "Prior matched-horizon fit changed")
        return
    counts, data, cutoff, digest = load_data(ctx, ("train", "validation"))
    model, variant = build_model(ctx, method, counts, seed, cutoff)
    device = torch.device("cuda")
    step = 0

    def checkpoint(_optimizer, _args, _kwargs):
        nonlocal step
        step += 1
        if step == STEPS:
            train.atomic(path, lambda p: torch.save({"model": model.state_dict()}, p))
            train.save_json(receipt, {**signed, "weight_sha": train.sha(path)})
            raise FixedHorizonReached

    hook = register_optimizer_step_post_hook(checkpoint)
    try:
        lr, batch = PLAN["learning_rates"]["S8"], PLAN["batch"]
        if method == "Replay":
            replay = replay_module(TARGET)
            require(replay.PLAN["batch_size"] == batch and replay.PLAN["learning_rate"] == lr,
                    "Replay schedule changed")
            model.to(device)
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
            replay.training_epoch(model, optimizer, data["train"], seed, 0, batch, device)
        else:
            args = SimpleNamespace(seed=seed, device=str(device), parent_epochs=20,
                                   adapt_epochs=PLAN["epochs"], batch_size=batch, mode="fit", limit=None)
            train.fit_trial(variant, model, data, args, lr, out / "trial", ctx.parent_sha, digest)
    except FixedHorizonReached:
        return
    finally:
        hook.remove()
    raise RuntimeError("The training loop did not reach the committed horizon")


def freeze():
    ctx = configure(TARGET, SEEDS[0])
    sha = ctx.train.sha
    archive = read(CORE / "runs/freeze.json")
    vg_replay = CORE / "replay_vg_20261111/results"
    vg_epochs = {}
    fits = {}
    for seed in SEEDS:
        base = CORE / "runs" / str(seed)
        vg_label = read(vg_replay / str(seed) / "FIT_COMPLETE.json")
        vg_ft = read(base / "S8.json")
        require(vg_ft["checkpoint_sha"] == archive["checkpoints"][str(seed)]["S8"]
                and vg_label["contract"]["source_sha"] == PLAN["archived_replay_source"]["video_games"]
                and vg_label["contract"]["parent_sha"] == vg_ft["contract"]["parent_sha"]
                == archive["checkpoints"][str(seed)]["S0"]
                and vg_ft["selected_epoch"] == vg_label["selected_epoch"] == 2
                and vg_label["epochs"][0]["steps"] == 110, "VG selected steps differ from 220")
        vg_epochs[str(seed)] = {"S8": sha(base / "S8.json"),
                                "Replay": sha(vg_replay / str(seed) / "FIT_COMPLETE.json")}
        for method in METHODS:
            out = folder(seed, method)
            record = read(out / "FIT.json")
            require(record["seed"] == seed and record["method"] == method and record["steps"] == STEPS
                    and record["source"] == sha(Path(__file__))
                    and record["parent_sha"] == archive["checkpoints"][str(seed)]["S0"]
                    and record["weight_sha"] == sha(out / "selected.pt"), "Aligned fit identity changed")
            fits[f"{seed}_{method}"] = sha(out / "FIT.json")
    frozen = {"steps": STEPS, "fits": fits, "vg_archive_220_step": vg_epochs,
              "test_manifest_sha": PLAN["amazon_test"]}
    path = RESULTS / "FREEZE.json"
    if path.is_file():
        require(read(path) == frozen, "Matched-step freeze changed")
    else:
        ctx.train.save_json(path, frozen)


def test(seed):
    ctx = configure(TARGET, seed)
    train = ctx.train
    locked = RESULTS / "FREEZE.json"
    frozen = read(locked)
    counts, data, cutoff, digest = load_data(ctx, ("test",))
    require(digest == frozen["test_manifest_sha"], "Aligned test split changed")
    rows = data["test"][DOMAIN]
    require(len(rows["target"]) == 81849, "Movies & TV request population changed")
    device = torch.device("cuda")
    for method in METHODS:
        out = folder(seed, method)
        require(train.sha(out / "FIT.json") == frozen["fits"][f"{seed}_{method}"]
                and train.sha(out / "selected.pt") == read(out / "FIT.json")["weight_sha"],
                "Aligned model differs from freeze")
        done = out / "TEST.json"
        arrays = out / "movies.npz"
        if done.is_file():
            require(read(done)["freeze"] == train.sha(locked)
                    and read(done)["predictions"] == train.sha(arrays), "Aligned test differs")
            continue
        model, _ = build_model(ctx, method, counts, seed, cutoff)
        model.load_state_dict(torch.load(out / "selected.pt", map_location="cpu", weights_only=True)["model"])
        _, ranks, _, top = train.evaluate(model.to(device), DOMAIN, rows, device, 512, save_top=True)

        def save(path):
            with path.open("wb") as stream:
                np.savez_compressed(stream, user=rows["user"].numpy(), gap=rows["gap"].numpy(),
                                    known=rows["target"].numpy() >= 0, untied=rows["untied"].numpy(),
                                    rank=ranks, top10=top)

        train.atomic(arrays, save)
        train.save_json(done, {"freeze": train.sha(locked), "predictions": train.sha(arrays)})


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action", choices=("fit", "test", "launch"))
    cli.add_argument("--seed", type=int, choices=SEEDS)
    cli.add_argument("--method", choices=METHODS)
    args = cli.parse_args()
    if args.action == "fit":
        fit(args.seed, args.method)
    elif args.action == "test":
        test(args.seed)
    else:
        launch.LOGS.mkdir(exist_ok=True)
        launch.WORKER = [sys.executable, "-B", str(Path(__file__).resolve())]
        launch.wave([(f"aligned_fit_{method}_{seed}",
                      ["fit", "--seed", str(seed), "--method", method])
                     for seed in SEEDS for method in METHODS])
        freeze()
        launch.wave([(f"aligned_test_{seed}", ["test", "--seed", str(seed)])
                     for seed in SEEDS])
        print("MATCHED_STEP_TEST_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
