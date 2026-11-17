"""Recompute policy and method contrasts from the returned five-seed receipts."""
import argparse
from pathlib import Path
from statistics import mean, stdev

from scipy.stats import t

from paths import GROUPS, PLAN, PLAN_PATH, SEEDS, methods, read, require, sha, write

HORIZONS = tuple(str(n) for n in PLAN["horizons"])
POLICIES = ("unrestricted", "joint", "exact", "capped")
LIST_METRICS = {"delta_hit": 100, "deep": 100, "mean_replaced": 1, "rbo": 1, "top1_changed": 100}


def paired(values):
    require(len(values) == len(SEEDS), "Wrong number of paired training seeds")
    average = mean(values)
    half_width = t.ppf(.975, len(values) - 1) * stdev(values) / len(values)**.5
    return {"mean": average, "ci95": [average - half_width, average + half_width],
            "by_seed": dict(zip(SEEDS, values))}


def paired_change(after, before):
    return paired([after["by_seed"][seed] - before["by_seed"][seed] for seed in SEEDS])


def group_summary(root, group):
    recipe = GROUPS[group]
    target = recipe["target"]
    lock_path = root / "runs" / group / "FREEZE.json"
    frozen = read(lock_path)
    lock_sha = sha(lock_path)
    require(frozen["plan_sha"] == sha(PLAN_PATH) and frozen["group"] == group
            and tuple(map(int, frozen["seeds"])) == SEEDS, "Unbound five-seed selection")
    results = {}
    for seed in SEEDS:
        folder = lock_path.parent / str(seed)
        report = read(folder / "test/COMPLETE.json")
        old = frozen["seeds"][str(seed)]
        require(old["sha256"] == sha(folder / "validation/COMPLETE.json")
                and report["group"] == group and report["seed"] == seed
                and report["data_sha"] == PLAN["data"][recipe["source"]]["test"]
                and report["lock_sha"] == lock_sha and report["fits"] == old["validation"]["fits"],
                "Returned Test is not paired to the validation lock")
        for method in methods(group):
            path = folder / method / "FIT.json"
            require(report["fits"][method] == sha(path), "Fit hash differs")
            record = read(path)
            require(record["epochs"] == 100 and set(record["selected"]) == set(HORIZONS), "Incomplete fit")
            for h in HORIZONS:
                scored = report["horizons"][h][method]
                chosen = record["selected"][h]
                require(scored["epoch"] == chosen["epoch"] <= int(h)
                        and scored["weights_sha"] == chosen["weights"]["sha256"], "Unexpected selected checkpoint")
                if method in PLAN["exact_methods"]:
                    for domain in scored["retained"]:
                        compared = scored["domains"][domain]["identity"]
                        require(compared["compared"] and compared["mismatch_rows"] == 0 and compared["max_abs"] == 0,
                                "Missing path-preservation evidence")
        results[seed] = report["horizons"]

    def measure(h, method, key="ndcg10"):
        return [100 * results[s][h][method]["domains"][target]["metrics"]["long_known"][key]
                for s in SEEDS]

    def delta(a, b):
        return paired([x - y for x, y in zip(a, b)])

    all_methods = ("S0", *methods(group))
    horizons = {}
    for h in HORIZONS:
        decisions = frozen["decisions"][h]
        outcomes = {
            m: {
                "test_ndcg_pct": mean(measure(h, m)),
                "test_hit_pct": mean(measure(h, m, "hit10")),
                "ndcg_minus_base_pp": delta(measure(h, m), measure(h, "S0")),
                "selected_epochs": [results[s][h][m]["epoch"] for s in SEEDS],
            } for m in all_methods
        }
        comparisons = {}
        pairs = [("Replay", "S8")]
        if "Distill_4" in all_methods:
            pairs += [("Distill_1", "Replay"), ("Distill_4", "Replay"), ("Distill_4", "S6")]
        for a, b in pairs:
            comparisons[f"{a}_minus_{b}"] = {
                "ndcg_pp": delta(measure(h, a), measure(h, b)),
                "retained": {
                    d: {metric: paired([
                        scale * (results[s][h][a]["retained"][d][metric]
                                 - results[s][h][b]["retained"][d][metric])
                        for s in SEEDS]) for metric, scale in LIST_METRICS.items()}
                    for d in results[SEEDS[0]][h][a]["retained"]
                },
            }
        horizons[h] = {
            "decisions": {p: decisions[p] for p in POLICIES},
            "joint_feasible": decisions["joint_feasible"],
            "scopes": outcomes,
            "gated_minus_additive_pp": delta(measure(h, "S5"), measure(h, "S4a")),
            "capped_minus_exact_pp": delta(measure(h, decisions["capped"]), measure(h, decisions["exact"])),
            "replay_and_distillation": comparisons,
        }
    shifts = {
        f"{after}_minus_{before}": {
            "methods_pp": {m: delta(measure(after, m), measure(before, m)) for m in methods(group)},
            "policy_outcomes_pp": {
                p: delta(measure(after, horizons[after]["decisions"][p]),
                         measure(before, horizons[before]["decisions"][p]))
                for p in POLICIES
            },
        } for after, before in (("50", "10"), ("100", "10"), ("100", "50"))
    }
    after = horizons["100"]["replay_and_distillation"]["Replay_minus_S8"]["retained"]
    before = horizons["50"]["replay_and_distillation"]["Replay_minus_S8"]["retained"]
    shifts["100_minus_50"]["replay_minus_s8_retained"] = {
        domain: {metric: paired_change(after[domain][metric], before[domain][metric])
                 for metric in LIST_METRICS}
        for domain in after
    }
    return {"validation_lock": lock_sha,
            "n": PLAN["data"][recipe["source"]]["long_covered_test"][target],
            "horizons": horizons, "shifts": shifts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("returned", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    groups = {group: group_summary(args.returned, group) for group in GROUPS}
    reversals = {}
    for backbone in ("sasrec", "gru"):
        rot = groups[f"{backbone}-cds_vinyl"]["horizons"]
        vg = groups[f"{backbone}-video_games"]["horizons"]
        reversals[backbone] = {}
        for h in HORIZONS:
            def movies(row):
                return row[h]["replay_and_distillation"]["Replay_minus_S8"]["retained"]["movies_tv"]["rbo"]
            a, b = movies(vg), movies(rot)
            reversals[backbone][h] = {"VG": a, "CDs": b, "CDs_minus_VG": paired_change(b, a)}
    write(args.output, {"seeds": SEEDS, "groups": groups, "movies_rbo": reversals,
                        "status": "verified", "plan_sha": sha(PLAN_PATH)})
    for group, row in groups.items():
        print(group, "1 MiB choice", [v["decisions"]["capped"] for v in row["horizons"].values()],
              "100-minus-50", row["shifts"]["100_minus_50"]["policy_outcomes_pp"]["capped"])


if __name__ == "__main__":
    main()
