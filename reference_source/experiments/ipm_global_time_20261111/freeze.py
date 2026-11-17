#!/usr/bin/env python3
"""Bind all fitted checkpoints to the validation-selected recipe before test."""
import argparse
import json
from pathlib import Path
from statistics import mean
from train import SEEDS, VARIANTS, PARENTS, sha, save_json

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("runs", type=Path)
p.add_argument("--test-manifest-sha", required=True)
p.add_argument("--lock", action="store_true")
args = p.parse_args()
recipe = args.runs / "9308/recipe.json"
selected = json.loads(recipe.read_text())
checks = {}
validation = {name:[] for name in VARIANTS}
for seed in SEEDS:
    directory = args.runs / str(seed)
    receipt = json.loads((directory / "FIT_COMPLETE.json").read_text())
    if receipt["seed"] != seed or receipt["development_sha"] != selected["development_sha"] or receipt["lrs"] != selected["lrs"]:
        raise RuntimeError(f"Fitted recipe mismatch: {seed}")
    hashes = {}
    for variant in VARIANTS:
        row = json.loads((directory / f"{variant}.json").read_text())
        found = sha(directory / f"{variant}.pt")
        contract = row["contract"]
        parent_sha = hashes[PARENTS[variant]] if variant in PARENTS else None
        if (row["checkpoint_sha"] != found or contract["source_sha"] != selected["source_sha"]
            or contract["seed"] != seed or contract["data_sha"] != selected["development_sha"]
            or contract["parent_sha"] != parent_sha or contract["lr"] != selected["lrs"][variant]
            or contract["limit"] is not None or contract["device"] != "cuda"
            or contract["epochs"] != (20 if variant in ("S0","time_start","meantime") else 10)
            or contract["batch"] != (256 if variant == "meantime" else 2048)
            or contract["mode"] != ("tune" if seed==9308 else "fit")):
            raise RuntimeError(f"Fitted weight mismatch: {seed} {variant}")
        hashes[variant] = found
        validation[variant].append(row["validation"]["long_known"]["ndcg10"])
    checks[str(seed)] = hashes
path = args.runs / ("freeze.json" if args.lock else "fit_weights.json")
value = {"seeds":list(SEEDS), "recipe_sha":sha(recipe), "checkpoints":checks,
         "test_manifest_sha":args.test_manifest_sha}
if args.lock:
    costs = {}
    for surface in ("S4a","S5","S6","S8"):
        files = [args.runs/"cost"/f"{surface}_rep{i}.json" for i in (1,2,3)]
        costs[surface] = [json.loads(file.read_text()) for file in files]
        for i,cost in enumerate(costs[surface],1):
            contract = cost["contract"]
            if (cost["replicate"] != i or contract["variant"] != surface
                or contract["seed"] != 42096 or contract["mode"] != "cost"
                or contract["data_sha"] != selected["development_sha"]
                or contract["source_sha"] != selected["source_sha"]
                or contract["parent_sha"] != checks["42096"]["S0"]
                or contract["lr"] != selected["lrs"][surface]
                or contract["epochs"] != 10 or contract["batch"] != 2048):
                raise RuntimeError(f"Cost receipt mismatch: {surface} replicate {i}")
        value.setdefault("cost_sha256",{}).update({file.name:sha(file) for file in files})
    storage = {s:max(x["serialized_update_bytes"] for x in costs[s]) for s in costs}
    storage["S0"] = 0
    storage["S8_copy"] = max(x["target_copy_file_bytes"] for x in costs["S8"])
    storage["meantime"] = (args.runs/"42096/meantime.pt").stat().st_size
    quality = {k:mean(v) for k,v in validation.items()}
    quality["S8_copy"] = quality["S8"]
    strict = ("S0","S4a","S5","S6","S8_copy","meantime")
    choices = {"strict_invariant":strict,
               "strict_invariant_within_1MiB":tuple(s for s in strict if storage[s]<=1024**2),
               "unrestricted":("S0","S4a","S5","S6","S8","meantime")}
    value["validation_decisions"] = {
        k:{"selected":max(pool,key=lambda s:(quality[s],-storage[s])),
           "eligible":list(pool)} for k,pool in choices.items()}
    value["validated_storage_bytes"] = storage
    value["validation_long_known_ndcg10"] = quality
if path.exists() and json.loads(path.read_text()) != value:
    raise RuntimeError("Existing freeze manifest differs")
save_json(path,value)
print(sha(path),flush=True)
