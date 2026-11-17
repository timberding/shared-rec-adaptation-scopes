"""Paths and identities shared by the inventory and workers."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
CORE = WORKSPACE / "deltagate_ipm_global_time_20261111_r1/experiments/ipm_global_time_20261111"
YELP = WORKSPACE / "deltagate_ipm_yelp_20261112_r1"
RESTAURANTS = WORKSPACE / "deltagate_ipm_yelp_restaurants_20261112"
FOLLOWUP = WORKSPACE / "deltagate_ipm_review_closure_20261115"
HORIZON50 = WORKSPACE / "deltagate_ipm_horizon_50_20261115"
BACKBONES = WORKSPACE / "001_rebuttal_package/ipm_backbones_20261114"
BACKBONE_SRC = BACKBONES / "07_resubmission_work/deltagate_resubmission/experiments/ipm_shared_gru_20261113"
RUNS = HERE / "runs"
PLAN_PATH = HERE / "PLAN.json"
INPUTS_PATH = HERE / "INPUTS.json"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


PLAN = read(PLAN_PATH)
SEEDS = tuple(PLAN["seeds"])
GROUPS = PLAN["groups"]


def methods(group):
    return (*PLAN["methods"], *GROUPS[group].get("extra_comparisons", []))


def parent_family(group):
    recipe = GROUPS[group]
    return f"{recipe['source']}-{recipe['backbone']}"


def base_paths(family, seed):
    parent = RUNS / "base" / family / str(seed)
    return parent / "S0.pt", parent / "FIT.json"


def development(source):
    return ((CORE if source == "amazon" else YELP) / "data/development")
