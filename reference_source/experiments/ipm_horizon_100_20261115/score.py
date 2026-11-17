"""Apply the original validation decision rules and score only locked checkpoints."""
import argparse
from pathlib import Path
from statistics import mean

import numpy as np

from fit_adaptation import make_model
from paths import (BACKBONE_SRC, GROUPS, PLAN, PLAN_PATH, RUNS, SEEDS,
                   methods, read, require, sha, write)
from runtime import DEVICE, configure, module, parent_path, rows_for, torch, verified_file

HORIZONS = tuple(str(e) for e in PLAN["horizons"])
EXACT = tuple(PLAN["exact_methods"])
SCOPE_CANDIDATES = ("S0", *PLAN["methods"])


def policy(by_seed):
    means = {}
    for method in SCOPE_CANDIDATES:
        values = [row[method] for row in by_seed]
        means[method] = {
            "ndcg": mean(row["selection_ndcg"] for row in values),
            "bytes": max(row["update_bytes"] for row in values),
            "retained": {domain: {
                name: mean(row["retained"][domain][name] for row in values)
                for name in ("delta_hit", "deep")
            } for domain in values[0]["retained"]},
        }
    ordered = sorted(means, key=lambda m: (-means[m]["ndcg"],
                                           means[m]["bytes"] if m in EXACT else float("inf"), m))
    floor, limit, cap = (PLAN["thresholds"][k] for k in ("delta_hit", "deep", "bytes"))
    feasible = [m for m in ordered if all(v["delta_hit"] >= floor and v["deep"] <= limit
                                           for v in means[m]["retained"].values())]
    exact = [m for m in ordered if m in EXACT]
    return {"unrestricted": ordered[0], "joint": feasible[0], "exact": exact[0],
            "capped": next(m for m in exact if means[m]["bytes"] <= cap),
            "joint_feasible": feasible, "validation_means": means}


def freeze(group):
    rows = {}
    for seed in SEEDS:
        path = RUNS / group / str(seed) / "validation/COMPLETE.json"
        row = read(path)
        require(row["plan_sha"] == sha(PLAN_PATH)
                and row["data_sha"] == PLAN["data"][GROUPS[group]["source"]]["development"]
                and row["group"] == group and row["seed"] == seed, "Validation receipt changed")
        for method, entry in row["fits"].items():
            require(entry == sha(RUNS / group / str(seed) / method / "FIT.json"), "Fit changed before lock")
        require(set(row["fits"]) == set(methods(group)), "Incomplete validation receipt")
        rows[str(seed)] = {"sha256": sha(path), "validation": row}
    output = {"group": group, "plan_sha": sha(PLAN_PATH), "seeds": rows,
              "decisions": {h: policy([row["validation"]["horizons"][h] for row in rows.values()])
                            for h in HORIZONS}}
    path = RUNS / group / "FREEZE.json"
    if path.exists():
        require(read(path) == output, "Would change an existing validation lock")
    else:
        write(path, output)


def prediction(train, path, rank, top):
    def save(temporary):
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, rank=rank, top10=top)
    train.atomic(path, save)
    return {"file": path.name, "sha256": sha(path)}


def score(group, seed, phase):
    folder = RUNS / group / str(seed) / phase
    folder.mkdir(parents=True, exist_ok=True)
    done = folder / "COMPLETE.json"
    if done.exists():
        return
    plan_sha = sha(PLAN_PATH)
    locked = None
    if phase == "test":
        lock = RUNS / group / "FREEZE.json"
        locked = read(lock)
        require(locked["group"] == group and locked["plan_sha"] == plan_sha
                and set(locked["seeds"]) == {str(s) for s in SEEDS}, "Validation must be frozen before Test")
    train, runtime = configure(group)
    retained = module("matched_retention_metrics", BACKBONE_SRC / "run.py").retained
    target = GROUPS[group]["target"]
    parent, parent_sha = parent_path(group, seed)
    counts, data, cutoff, digest = rows_for(group, phase)
    data = data[phase]
    baseline = train.model_for("S0", counts, seed, cutoff).to(DEVICE)
    baseline.load_state_dict(torch.load(parent, map_location="cpu", weights_only=True)["model"])
    if phase == "validation":
        initial_metrics, *_ = train.evaluate(baseline, target, data[target], DEVICE, PLAN["batch_size"])
    fits, result, reference = {}, {}, {}
    for method in ("S0", *methods(group)):
        fit_path = folder.parent / method / "FIT.json"
        if method != "S0":
            fit = read(fit_path)
            binding = fit["binding"]
            require(binding["runtime"] == runtime and binding["group"] == group
                    and binding["parent"] == parent_sha and binding["seed"] == seed
                    and binding["method"] == method and binding["plan"] == plan_sha
                    and not binding["probe"] and fit["epochs"] == 100, "Unbound adaptation")
            fits[method] = sha(fit_path)
            if locked is not None:
                old = locked["seeds"][str(seed)]["validation"]
                require(old["fits"][method] == fits[method], "Fit differs from validation lock")
        states = {}
        for h in HORIZONS:
            verified = fit["selected"][h] if method != "S0" else None
            if verified is None:
                epoch, identity = 0, parent_sha
            else:
                epoch, identity = verified["epoch"], verified["weights"]["sha256"]
            if identity not in states:
                if method == "S0":
                    model = baseline
                else:
                    model = make_model(train, group, method, counts, seed, cutoff, parent)
                    path = verified_file(verified["weights"])
                    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model"])
                row = {"epoch": epoch, "weights_sha": identity, "domains": {}, "retained": {},
                       "update_bytes": 0 if method == "S0" else fit["update_bytes"]}
                for domain, cohort in data.items():
                    exact = method in EXACT[1:] and domain != target
                    scores, rank, equal, top = train.evaluate(
                        model, domain, cohort, DEVICE, PLAN["eval_batch_size"],
                        reference=baseline if exact else None, save_top=True)
                    if method == "S0":
                        reference[domain] = (rank, top)
                    if domain != target:
                        if method != "S0":
                            require(all(torch.equal(w, baseline.heads[domain].state_dict()[k])
                                        for k, w in model.base.heads[domain].state_dict().items()),
                                    "Non-target head changed")
                        if exact:
                            require(equal["compared"] and not equal["mismatch_rows"] and equal["max_abs"] == 0
                                    and np.array_equal(rank, reference[domain][0])
                                    and np.array_equal(top, reference[domain][1]), "Exact preservation failed")
                        row["retained"][domain] = retained(*reference[domain], rank, top)
                        row["retained"][domain]["top1_changed"] = float((top[:, 0] != reference[domain][1][:, 0]).mean())
                    else:
                        expected = PLAN["data"][GROUPS[group]["source"]][f"long_covered_{phase}"][target]
                        require(scores["long_known"]["n"] == expected, "Target request count differs")
                    pred = prediction(train, folder / f"{method}-e{epoch}-{domain}.npz", rank, top)
                    row["domains"][domain] = {"metrics": scores, "prediction": pred, "identity": equal}
                if phase == "validation":
                    selected_metrics = initial_metrics if method == "S0" else verified["val"]
                    row["selection_ndcg"] = selected_metrics["long_known"]["ndcg10"]
                    require(row["selection_ndcg"] is not None, "Missing validation score")
                states[identity] = row
                del model
            result.setdefault(h, {})[method] = states[identity]
    write(done, {"group": group, "seed": seed, "phase": phase, "data_sha": digest,
                 "plan_sha": plan_sha, "fits": fits, "horizons": result,
                 "lock_sha": sha(lock) if locked is not None else None})
    print(f"SCORED {group} {seed} {phase}", flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("validation", "freeze", "test"))
    cli.add_argument("group", choices=GROUPS)
    cli.add_argument("seed", type=int, nargs="?", choices=SEEDS)
    args = cli.parse_args()
    if args.phase == "freeze":
        freeze(args.group)
    else:
        require(args.seed is not None, "Specify a paired seed")
        score(args.group, args.seed, args.phase)


if __name__ == "__main__":
    main()
