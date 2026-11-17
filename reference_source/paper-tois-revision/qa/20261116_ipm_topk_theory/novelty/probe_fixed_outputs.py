"""Score fixed validation checkpoints using the original BF16 inference protocol."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch


def radius_bounds(top, k, depth):
    """Bracket the clipping threshold. Evaluate each term at its cheapest endpoint."""
    inside, outside = top[:, k-depth:k], top[:, k:k+depth]
    low, high = outside[:, -1], inside[:, 0]
    for _ in range(48):
        mid = (low + high) / 2
        excess = np.maximum(inside - mid[:, None], 0).sum(1)
        deficit = np.maximum(mid[:, None] - outside, 0).sum(1)
        low, high = np.where(excess > deficit, mid, low), np.where(excess > deficit, high, mid)
    p = np.concatenate([inside, outside], axis=1)
    q_lower = np.concatenate([np.minimum(inside, high[:, None]),
                              np.maximum(outside, low[:, None])], axis=1)
    q_upper = np.concatenate([np.minimum(inside, low[:, None]),
                              np.maximum(outside, high[:, None])], axis=1)

    def cost(q):
        delta = (p - q) / q
        return (q * ((1 + delta) * np.log1p(delta) - delta)).sum(1)

    # Each replaced insider pairs with an entrant. The sum of these teacher
    # mass gaps is at least the gap between the closest two depth-sized blocks.
    # The disjoint pairs require TV >= half of this gap. Pinsker gives KL >= 2 TV².
    pinsker = (inside.sum(1) - outside.sum(1)) ** 2 / 2
    return cost(q_lower), cost(q_upper), pinsker


def self_check():
    p = np.array([[.60, .30, .05, .049, .001]])
    lo, hi, pinsker = radius_bounds(p, 3, 2)
    assert abs(lo[0] - .23607163396890865) < 1e-14
    assert abs(hi[0] - lo[0]) < 1e-14
    assert pinsker[0] < lo[0] < .2553670128804517


@torch.inference_mode()
def run(args):
    sys.path.insert(0, str(args.experiment))
    from fit_adaptation import make_model
    from paths import PLAN, PLAN_PATH, RUNS, require, sha
    from runtime import DEVICE, configure, rows_for, verified_file

    binding = json.loads(args.binding.read_text())
    group, k, temp = binding["group"], binding["K"], binding["temperature"]
    require(sha(PLAN_PATH) == binding["plan_sha256"], "Plan identity differs")
    require(args.seed in binding["paired_seeds"], "Unbound seed")
    train, runtime = configure(group)
    torch.cuda.set_per_process_memory_fraction(8 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    counts, data, cutoff, data_sha = rows_for(
        group, "validation", domains=binding["non_target_validation_counts"])
    require(binding["development_sha256"] == data_sha, "Validation identity differs")
    models, registrations = {}, {}
    archived = RUNS / group / str(args.seed) / "validation"
    for entry in binding["bindings"]:
        if entry["seed"] != args.seed:
            continue
        method = entry["method"]
        require(sha(archived / "COMPLETE.json") == entry["validation_receipt_sha256"],
                f"Validation receipt differs: {method}")
        if method == "S0":
            expected = entry["weights"]["input_inventory_matches"][0]
            require(expected["sha256"] == entry["weights"]["sha256"], "Base identity differs")
            base_path = verified_file(expected)
            model = train.model_for(method, counts, args.seed, cutoff).to(DEVICE)
            path = base_path
        else:
            require(sha(archived.parent / method / "FIT.json") == entry["fit_receipt_sha256"],
                    f"Fit receipt differs: {method}")
            path = verified_file(entry["weights"])
            model = make_model(train, group, method, counts, args.seed, cutoff, base_path)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model"])
        models[method] = model.eval()
        registrations[method] = entry
    require(list(models) == binding["methods"], "Checkpoint set differs")
    arrays, domain_reports = {}, {}
    for domain, rows in data["validation"].items():
        n = len(rows["target"])
        require(n == binding["non_target_validation_counts"][domain], "Request count differs")
        reports, reference = {}, {}
        for method, entry in registrations.items():
            pred = entry["domains"][domain]["archived_prediction"]
            with np.load(verified_file({"path": archived / pred["file"], "sha256": pred["sha256"]}),
                         allow_pickle=False) as saved:
                reference[method] = {name: saved[name] for name in ("rank", "top10")}
            reports[method] = {"rank_mismatch": 0, "ordered_top10_mismatch": 0, "membership_mismatch": 0}
        teacher_top = []
        lengths = {method: {"kl": [], "replaced": [], "student_tie": []} for method in models if method != "S0"}
        torch.cuda.synchronize()
        start = time.monotonic()
        for index in train.slices(n, PLAN["eval_batch_size"]):
            batch = train.batch_from(rows, index, DEVICE)
            p_logits, p_log_z, p_weights, teacher_ids = None, None, None, None
            for method, model in models.items():
                with train.amp(DEVICE):
                    logits = model(domain, batch)
                require(torch.isfinite(logits).all().item(), f"Nonfinite logits: {domain}/{method}")
                ids = torch.arange(logits.shape[1], device=DEVICE)
                target = batch["target"]
                hit_score = logits.gather(1, target.clamp_min(0)[:, None])
                rank = 1 + ((logits > hit_score) | ((logits == hit_score) & (ids < target[:, None]))).sum(1)
                rank = torch.where(target >= 0, rank, -1).cpu().numpy()
                top = train.stable_top10(logits)
                ranked = top.cpu().numpy()
                archived_top = reference[method]["top10"][index]
                report = reports[method]
                report["rank_mismatch"] += int((rank != reference[method]["rank"][index]).sum())
                report["ordered_top10_mismatch"] += int((ranked != archived_top).any(1).sum())
                report["membership_mismatch"] += int(
                    (np.sort(ranked, axis=1) != np.sort(archived_top, axis=1)).any(1).sum())
                scores = logits.to(torch.float64).div_(temp)
                # Use a full denominator, with the same item universe as rank evaluation.
                score_log_z = scores.logsumexp(1)
                if method == "S0":
                    p_logits, p_log_z, teacher_ids = scores, score_log_z, top
                    p_weights = (scores - score_log_z[:, None]).exp_()
                    values = scores.topk(2 * k, dim=1).values
                    teacher_top.append((values - p_log_z[:, None]).exp_().cpu().numpy())
                else:
                    scores.neg_().add_(p_logits).mul_(p_weights)
                    kl = scores.sum(1) + (score_log_z - p_log_z)
                    replaced = k - (teacher_ids[:, :, None] == top[:, None, :]).any(2).sum(1)
                    cutoff_values = logits.topk(k+1, dim=1).values
                    lengths[method]["kl"].append(kl.cpu().numpy())
                    lengths[method]["replaced"].append(replaced.cpu().numpy().astype(np.uint8))
                    lengths[method]["student_tie"].append((cutoff_values[:, k-1] == cutoff_values[:, k]).cpu().numpy())
        torch.cuda.synchronize()
        seen = time.monotonic() - start
        top = np.concatenate(teacher_top)
        arrays[f"{domain}/strict_boundary"] = top[:, k-1] > top[:, k]
        arrays[f"{domain}/top"] = top
        for method, values in lengths.items():
            for name, batch_values in values.items():
                arrays[f"{domain}/{method}/{name}"] = np.concatenate(batch_values)
        print(f"{args.seed} {domain} {n} requests times {len(models)} models in {seen:.2f}s; {reports}",
              file=sys.stderr, flush=True)
        domain_reports[domain] = {"n": n, "seconds_forward_and_statistics": seen, "archive_comparison": reports}
    result = {
        "seed": args.seed, "group": group, "protocol": "validation only, frozen 50-epoch selection",
        "binding_sha256": sha(args.binding), "script_sha256": sha(Path(__file__)),
        "development_sha256": data_sha, "plan_sha256": sha(PLAN_PATH),
        "weights_sha256": {method: row["weights"]["sha256"] for method, row in registrations.items()},
        "runtime": runtime, "GPU": torch.cuda.get_device_name(0),
        "batch_size": PLAN["eval_batch_size"], "temperature": temp, "K": k,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(), "domains": domain_reports,
    }
    # Keep boundary probabilities and scalar summaries, not dense logits or user IDs.
    with args.output.with_suffix(".tmp").open("wb") as stream:
        np.savez_compressed(stream, **arrays, receipt=np.array(json.dumps(result)))
    args.output.with_suffix(".tmp").replace(args.output)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("experiment", type=Path, nargs="?")
    cli.add_argument("binding", type=Path, nargs="?")
    cli.add_argument("seed", type=int, nargs="?")
    cli.add_argument("output", type=Path, nargs="?")
    cli.add_argument("--self-check", action="store_true")
    args = cli.parse_args()
    self_check()
    if not args.self_check:
        run(args)
