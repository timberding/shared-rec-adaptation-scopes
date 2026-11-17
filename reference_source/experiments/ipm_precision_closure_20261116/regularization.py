"""Bounded L2-SP check using the archived training and resume implementation."""
import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "deltagate_ipm_horizon_100_20261115"))
import fit_adaptation as fit
from paths import BACKBONE_SRC, read, write, sha, require
from runtime import DEVICE, configure, module, parent_path, rows_for, torch, verified_file

GROUP = "sasrec-video_games"


def setup(args):
    require(sha(Path(__file__)) == read(HERE / "PLAN.json")["source_sha256"]["regularization.py"],
            "Wrapper changed after plan was frozen")
    require(args.strength in ("1e-5", "1e-3", "1e-2"), "Original 1e-4 must be reused")
    base = HERE / "l2" / args.strength
    fit.RUNS = base / "runs"
    fit.HERE = base
    fit.PLAN = {**fit.PLAN, "reg_lambda": float(args.strength)}
    fit.HORIZONS = [50]
    fit.PLAN_PATH = HERE / "PLAN.json"
    fit.inventory = lambda: {"fits": {GROUP: {str(args.seed): {"S7": {"start": 0}}}}}
    return base


def evaluate(args, base):
    train, runtime = configure(GROUP)
    metrics_fn = module("l2_retention", BACKBONE_SRC / "run.py").retained
    parent, parent_sha = parent_path(GROUP, args.seed)
    counts, data, cutoff, data_sha = rows_for(GROUP, args.phase)
    folder = base / "runs" / GROUP / str(args.seed) / "S7"
    result = read(folder / "FIT.json")
    require(result["epochs"] == 50 and result["binding"]["parent"] == parent_sha
            and result["binding"]["plan"] == sha(HERE / "PLAN.json"), "Fit binding mismatch")
    chosen = result["selected"]["50"]
    if args.phase == "test":
        lock = read(HERE / "L2_SELECTION.json")
        require(lock["selected_lambda"] == args.strength and lock["plan_sha"] == sha(HERE / "PLAN.json"),
                "No validation-only lambda lock")
    path = verified_file(chosen["weights"])
    baseline = train.model_for("S0", counts, args.seed, cutoff).to(DEVICE)
    baseline.load_state_dict(torch.load(parent, map_location="cpu", weights_only=True)["model"])
    model = fit.make_model(train, GROUP, "S7", counts, args.seed, cutoff, parent)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model"])
    out = {"phase": args.phase, "seed": args.seed, "lambda": args.strength,
           "plan_sha": sha(HERE / "PLAN.json"), "fit_sha": sha(folder / "FIT.json"),
           "data_sha": data_sha, "selected": chosen, "domains": {}}
    for domain, rows in data[args.phase].items():
        scores, rank, _, top = train.evaluate(model, domain, rows, DEVICE, 512, save_top=True)
        entry = {"scores": scores}
        if domain != "video_games":
            _, ref_rank, _, ref_top = train.evaluate(baseline, domain, rows, DEVICE, 512, save_top=True)
            entry["retained"] = metrics_fn(ref_rank, ref_top, rank, top)
        out["domains"][domain] = entry
    write(folder / (args.phase + ".json"), out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("fit", "clean", "repeat", "interrupt", "resume", "validation", "test"))
    parser.add_argument("strength")
    parser.add_argument("seed", type=int)
    args = parser.parse_args()
    base = setup(args)
    if args.phase in ("validation", "test"):
        evaluate(args, base)
    else:
        fit.fit(GROUP, args.seed, "S7", args.phase)
