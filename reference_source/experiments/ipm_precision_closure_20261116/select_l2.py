"""Lock the bounded lambda comparison from calibration validation only."""
import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "deltagate_ipm_horizon_100_20261115"))
from paths import RUNS, PLAN_PATH, read, write, sha, require


def select(rows):
    eligible = [r for r in rows if all(v["delta_hit"] >= -.001 and v["deep"] <= .25
                                      for v in r["retained"].values())]
    return min(eligible or rows, key=lambda r: (-r["ndcg"], float(r["lambda"]))), eligible


def lock():
    rows, inputs = [], {}
    plan = HERE / "PLAN.json"
    for strength in ("1e-5", "1e-4", "1e-3", "1e-2"):
        if strength == "1e-4":
            path = RUNS / "sasrec-video_games/9308/validation/COMPLETE.json"
            receipt = read(path)
            require(receipt["phase"] == "validation" and receipt["plan_sha"] == sha(PLAN_PATH),
                    "Original calibration receipt changed")
            fit_path = RUNS / "sasrec-video_games/9308/S7/FIT.json"
            require(receipt["fits"]["S7"] == sha(fit_path), "Original fit changed")
            row = receipt["horizons"]["50"]["S7"]
            score, retained = row["selection_ndcg"], row["retained"]
        else:
            path = HERE / "l2" / strength / "runs/sasrec-video_games/9308/S7/validation.json"
            receipt = read(path)
            fit_path = path.with_name("FIT.json")
            require(receipt["phase"] == "validation" and receipt["seed"] == 9308
                    and receipt["lambda"] == strength and receipt["plan_sha"] == sha(plan)
                    and receipt["fit_sha"] == sha(fit_path), "New calibration receipt changed")
            score = receipt["selected"]["val"]["long_known"]["ndcg10"]
            retained = {d: r["retained"] for d, r in receipt["domains"].items() if d != "video_games"}
        require(set(retained) == {"cds_vinyl", "movies_tv"}, "Incomplete non-target comparison")
        inputs[str(path)] = sha(path)
        inputs[str(fit_path)] = sha(fit_path)
        rows.append({"lambda": strength, "ndcg": score, "retained": retained})
    selected, eligible = select(rows)
    return {"selected_lambda": selected["lambda"], "eligible": [r["lambda"] for r in eligible],
            "calibration_seed": 9308, "plan_sha": sha(plan), "selector_sha": sha(Path(__file__)),
            "inputs": inputs, "validation": rows}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(); cli.add_argument("--verify", action="store_true")
    args = cli.parse_args()
    a = {"lambda": "1e-4", "ndcg": 1., "retained": {"d": {"delta_hit": 0., "deep": 0.}}}
    b = {**a, "lambda": "1e-5"}
    assert select([a,b])[0]["lambda"] == "1e-5"
    assert select([{**a, "retained": {"d": {"delta_hit": -.1, "deep": 1.}}}])[1] == []
    result = lock(); path = HERE / "L2_SELECTION.json"
    if args.verify or path.exists():
        require(read(path) == result, "Validation lock differs")
    else:
        write(path, result)
    print({k: result[k] for k in ("selected_lambda", "eligible", "calibration_seed")})
