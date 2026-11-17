"""Fixed-checkpoint inference precision check; never changes model selection."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "deltagate_ipm_horizon_100_20261115"
sys.path.insert(0, str(OLD))
from paths import PLAN, PLAN_PATH, RUNS, sha, read, require, write
from runtime import DEVICE, configure, rows_for, parent_path, verified_file
from fit_adaptation import make_model

GROUP = "sasrec-video_games"
METHODS = ("S0", "S8", "Replay", "Distill_1", "Distill_4")


def list_metrics(base_rank, base_top, rank, top):
    hit0, hit = (base_rank > 0) & (base_rank <= 10), (rank > 0) & (rank <= 10)
    replaced = 10 - (base_top[:, :, None] == top[:, None, :]).any(2).sum(1)
    deep, both_miss = replaced >= 5, ~hit0 & ~hit
    overlap = np.array([(base_top[:, :k, None] == top[:, None, :k]).any(2).sum(1) / k
                        for k in range(1, 11)])
    rbo = .1 * (overlap * (.9 ** np.arange(10))[:, None]).sum(0) + .9**10 * overlap[-1]
    return {"n": len(rank), "hit": float(hit.mean()), "delta_hit": float(hit.mean()-hit0.mean()),
            "ndcg": float(np.where(hit, 1/np.log2(np.maximum(rank, 1)+1), 0).mean()),
            "deep": float(deep.mean()), "replaced": float(replaced.mean()), "rbo": float(rbo.mean()),
            "top1": float((base_top[:, 0] != top[:, 0]).mean()),
            "both_miss_n": int(both_miss.sum()), "deep_both_miss_n": int((deep & both_miss).sum()),
            "deep_both_miss_all": float((deep & both_miss).mean()),
            "deep_given_both_miss": float(deep[both_miss].mean()) if both_miss.any() else None}, replaced


def self_check():
    base = np.arange(10)[None, :]
    a, r = list_metrics(np.array([-1]), base, np.array([11]), base)
    assert a["rbo"] > 1-1e-12 and a["deep_both_miss_n"] == 0 and r[0] == 0
    b, r = list_metrics(np.array([11]), base, np.array([-1]), base+10)
    assert b["rbo"] == 0 and b["deep_both_miss_n"] == 1 and r[0] == 10
    c, _ = list_metrics(np.array([1]), base, np.array([-1]), base+10)
    assert c["deep_both_miss_n"] == 0 and c["both_miss_n"] == 0


@torch.inference_mode()
def run(args):
    self_check()
    train, runtime = configure(GROUP)
    # Highest also disables TF32; explicitly record both flags below.
    if args.precision == "fp32":
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    parent, parent_sha = parent_path(GROUP, args.seed)
    counts, data, cutoff, data_sha = rows_for(GROUP, args.phase, domains={"cds_vinyl", "movies_tv"})
    old_dir = RUNS / GROUP / str(args.seed) / args.phase
    receipt = read(old_dir / "COMPLETE.json")
    require(receipt["plan_sha"] == sha(PLAN_PATH) and receipt["data_sha"] == data_sha,
            "Original prediction receipt differs")
    folder = HERE / ("smoke" if args.smoke else "results") / str(args.seed) / args.phase
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / (args.precision + ".json")
    binding = {"source": sha(Path(__file__)), "old_plan": sha(PLAN_PATH), "data": data_sha,
               "original_receipt": sha(old_dir / "COMPLETE.json"), "parent": parent_sha,
               "seed": args.seed, "phase": args.phase, "precision": args.precision, "smoke": args.smoke}
    if output.exists():
        prior = read(output)
        require(prior["binding"] == binding and sha(output.with_suffix(".npz")) == prior["arrays_sha"],
                "Completed output changed")
        return
    models, weights = {}, {}
    for method in METHODS:
        original = receipt["horizons"]["50"][method]
        if method == "S0":
            path = parent
            model = train.model_for(method, counts, args.seed, cutoff).to(DEVICE)
        else:
            fit_path = RUNS / GROUP / str(args.seed) / method / "FIT.json"
            require(sha(fit_path) == receipt["fits"][method], "Fit differs from original receipt")
            fit = read(fit_path)
            path = verified_file(fit["selected"]["50"]["weights"])
            model = make_model(train, GROUP, method, counts, args.seed, cutoff, parent)
        require(sha(path) == original["weights_sha"], "Selected checkpoint changed")
        state = torch.load(path, map_location="cpu", weights_only=True)["model"]
        dtypes = sorted({str(v.dtype) for v in state.values() if v.is_floating_point()})
        require(dtypes == ["torch.float32"], "Checkpoint has lost FP32 weight precision")
        model.load_state_dict(state)
        models[method] = model.float().eval()
        weights[method] = {"path": str(path), "sha256": sha(path), "dtypes": dtypes,
                           "epoch": original["epoch"]}
    torch.cuda.reset_peak_memory_stats()
    arrays, results = {}, {}
    tick = time.monotonic()
    for domain, cohort in data[args.phase].items():
        n = min(256, len(cohort["target"])) if args.smoke else len(cohort["target"])
        rows = {m: {k: [] for k in ("rank", "top10", "ties", "kl", "top20")} for m in METHODS}
        for index in train.slices(n, 512):
            batch = train.batch_from(cohort, index, DEVICE)
            teacher_logits = teacher_lz = teacher_prob = None
            for method, model in models.items():
                context = train.amp(DEVICE) if args.precision == "bf16" else nullcontext()
                with context:
                    logits = model(domain, batch)
                require(logits.dtype == (torch.bfloat16 if args.precision == "bf16" else torch.float32),
                        "Unexpected scoring precision")
                require(torch.isfinite(logits).all().item(), "Nonfinite scores")
                ids = torch.arange(logits.shape[1], device=DEVICE)
                target = batch["target"]
                hit_score = logits.gather(1, target.clamp_min(0)[:, None])
                rank = 1 + ((logits > hit_score) | ((logits == hit_score) & (ids < target[:, None]))).sum(1)
                rows[method]["rank"].append(torch.where(target >= 0, rank, -1).cpu().numpy())
                rows[method]["top10"].append(train.stable_top10(logits).cpu().numpy())
                boundary = logits.topk(11, dim=1).values
                rows[method]["ties"].append((boundary[:, 9] == boundary[:, 10]).cpu().numpy())
                if args.phase == "validation":
                    scores = logits.to(torch.float64) / 2
                    lz = scores.logsumexp(1)
                    if method == "S0":
                        teacher_logits, teacher_lz = scores, lz
                        teacher_prob = (scores-lz[:, None]).exp()
                        rows[method]["top20"].append((scores.topk(20, dim=1).values-lz[:, None]).exp().cpu().numpy())
                    else:
                        kl = ((teacher_logits-scores)*teacher_prob).sum(1)+lz-teacher_lz
                        require((kl >= -1e-12).all().item(), "Negative KL")
                        rows[method]["kl"].append(kl.cpu().numpy())
        for method in METHODS:
            for key, parts in rows[method].items():
                if parts:
                    arrays[f"{domain}/{method}/{key}"] = np.concatenate(parts)
            rank, top = (arrays[f"{domain}/{method}/{k}"] for k in ("rank", "top10"))
            metrics, replaced = list_metrics(arrays[f"{domain}/S0/rank"], arrays[f"{domain}/S0/top10"], rank, top)
            arrays[f"{domain}/{method}/replaced"] = replaced.astype(np.uint8)
            pred = receipt["horizons"]["50"][method]["domains"][domain]["prediction"]
            original_path = old_dir / pred["file"]
            require(sha(original_path) == pred["sha256"], "Original predictions changed")
            with np.load(original_path, allow_pickle=False) as saved:
                metrics["original_rank_mismatch"] = int((rank != saved["rank"][:n]).sum())
                metrics["original_top10_mismatch"] = int((top != saved["top10"][:n]).any(1).sum())
            if args.precision == "bf16":
                require(metrics["original_rank_mismatch"] == metrics["original_top10_mismatch"] == 0,
                        "BF16 no longer reproduces original inference")
            metrics["boundary_tie_rate"] = float(arrays[f"{domain}/{method}/ties"].mean())
            results.setdefault(domain, {})[method] = metrics
        print(domain, results[domain], flush=True)
    torch.cuda.synchronize()
    temp = output.with_suffix(".npz.tmp")
    with temp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    temp.replace(output.with_suffix(".npz"))
    write(output, {"binding": binding, "weights": weights, "metrics": results, "runtime": runtime,
                   "arrays_sha": sha(output.with_suffix(".npz")), "seconds": time.monotonic()-tick,
                   "peak_allocated": torch.cuda.max_memory_allocated(),
                   "matmul_precision": torch.get_float32_matmul_precision(),
                   "cuda_tf32": torch.backends.cuda.matmul.allow_tf32,
                   "cudnn_tf32": torch.backends.cudnn.allow_tf32})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("seed", type=int)
    p.add_argument("phase", choices=("validation", "test"))
    p.add_argument("precision", choices=("bf16", "fp32"))
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())
