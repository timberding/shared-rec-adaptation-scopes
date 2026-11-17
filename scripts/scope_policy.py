"""Validation policy, copied without mathematical changes from the scoring kernel."""
from statistics import mean
import json
from pathlib import Path
PLAN=json.loads((Path(__file__).resolve().parents[1]/"configs/experiment.json").read_text())
EXACT=tuple(PLAN["exact_methods"])
SCOPE_CANDIDATES=("S0", *PLAN["methods"])

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
