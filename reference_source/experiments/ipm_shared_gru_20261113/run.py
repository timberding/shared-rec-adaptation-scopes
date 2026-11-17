#!/usr/bin/env python3
"""Matched backbone checks: fit on development, freeze selections, then test."""
import argparse
from itertools import count
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from statistics import mean, stdev
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
CORE = HERE.parent / "ipm_global_time_20261111"
sys.path.insert(0, str(CORE))
import train
from scripts.train_shared_backbone_seed42 import l2_to_snapshot, state_snapshot
from src.shared_backbone import SharedGRUEncoder

EXACT = ("S0", "S4a", "S5", "S6")
METHODS = (*EXACT, "S7", "S8", "Replay")
ARCHIVED = CORE / "returned_core_verified_20261111/runs/9308/recipe.json"
PLAN = json.loads((HERE / "PLAN.json").read_text())
SEEDS, TARGETS = tuple(PLAN["seeds"]), tuple(PLAN["targets"])
LRS = PLAN["learning_rates"]
BACKBONE = "shared_gru"
RUNTIME = {}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    return json.loads(path.read_text())


def bound_json(path, value):
    if path.exists():
        require(read(path) == value, f"Changed receipt: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        train.save_json(path, value)


def replay_for(target):
    path = CORE / PLAN["replay_sources"][target]
    spec = importlib.util.spec_from_file_location("replay_epoch", path)
    replay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay)
    require(replay.TARGET == target and set(SEEDS) <= set(replay.SEEDS)
            and replay.PLAN["learning_rate"] == LRS["Replay"]
            and replay.PLAN["development_manifest_sha256"] == PLAN["manifests"]["development"],
            "Replay protocol differs from the matched rotation")
    return replay


class SharedGRU(SharedGRUEncoder):
    def forward(self, embeddings, padding_mask):
        # S6 learns input embeddings through an otherwise frozen, eval-mode GRU.
        if not self.gru.training and torch.is_grad_enabled() and embeddings.requires_grad:
            self.gru.train()
            try:
                return super().forward(embeddings, padding_mask)
            finally:
                self.gru.eval()
        return super().forward(embeddings, padding_mask)


def configure(backbone="gru", device=torch.device("cpu"), summary_only=False):
    global BACKBONE, TARGETS, RUNTIME
    original = train.code_fingerprint()
    require(original == PLAN["sasrec_source_sha256"], "Archived base executable changed")
    parts = [original, train.sha(Path(__file__)), train.sha(HERE / "PLAN.json"), train.sha(ARCHIVED)]
    RUNTIME = {"kind": device.type, "torch": str(torch.__version__)}
    if backbone == "ssd4rec":
        import shared_ssd4rec as ssd
        spec = HERE / "SSD4REC_PLAN.json"
        config = read(spec)
        BACKBONE, TARGETS = config["backbone"], tuple(config["targets"])
        parts.extend(train.sha(p) for p in (spec, Path(ssd.__file__), *ssd.PINNED))
        require(all(train.sha(p) == expected for p, expected in ssd.PINNED.items()), "SSD4Rec source pin mismatch")
        if not summary_only:
            factory, registers, RUNTIME = ssd.load_factory(device)
            train.SharedSASRecEncoder = lambda width, heads, layers, length, dropout: ssd.SharedSSD4Rec(
                width, length, config["encoder"], factory, registers)
    else:
        train.SharedSASRecEncoder = lambda width, heads, layers, length, dropout: SharedGRU(
            width, layers, length, dropout)
    for path in PLAN["replay_sources"].values():
        parts.extend(train.sha(p) for p in (CORE / path, (CORE / path).with_name("PLAN.json")))
    source = hashlib.sha256("".join(parts).encode()).hexdigest()
    train.code_fingerprint = lambda: source
    torch.set_float32_matmul_precision("high")
    return source


def settings(seed, device, smoke):
    return SimpleNamespace(seed=seed, device=str(device), batch_size=16 if smoke else 2048,
                           parent_epochs=2 if smoke else 20, adapt_epochs=2 if smoke else 10,
                           mode="sanity" if smoke else BACKBONE, limit=64 if smoke else None)


def bundle(phase, smoke=False, limit=None):
    if phase == "test" and smoke:
        phase = "freeze"
    root = CORE / ("data/development" if phase != "test" else "data/test_sealed")
    digest = train.sha(root / "manifest.json")
    require(digest == PLAN["manifests"]["test" if phase == "test" else "development"], "Data manifest changed")
    windows = {"fit": ("train", "validation"), "freeze": ("validation",), "test": ("test",)}
    counts, data, manifest = train.load_data(root, windows[phase], limit if limit is not None else (64 if smoke else None))
    if BACKBONE == "shared_ssd4rec":
        for datasets in data.values():
            for domain, rows in datasets.items():
                # The bidirectional scan must never see a hole or an empty user segment.
                for seq in rows["seq"].split(65536):
                    valid = seq.ne(0)
                    require(bool(valid[:, -1].all()) and not bool((valid[:, :-1] & ~valid[:, 1:]).any()),
                            f"Noncontiguous or empty observed prefix: {domain}")
    return counts, data, manifest["cutoff_days_inclusive"][0], digest


def variant(method):
    return "S8" if method in ("S7", "Replay") else method


def update_bytes(model, method):
    if method == "S0":
        return 0
    f = io.BytesIO()
    torch.save({k: p.detach().cpu() for k, p in model.named_parameters() if p.requires_grad}, f)
    return f.tell()


def fit_one(method, out, counts, data, cutoff, digest, args, parent=None):
    out.mkdir(parents=True, exist_ok=True)
    path, receipt = out / f"{method}.pt", out / f"{method}.json"
    parent_sha = train.sha(parent) if parent else None
    base_contract = train.trial_contract(variant(method), args, LRS[method], parent_sha, digest)
    contract = {**base_contract, "scope": method, "target": train.TARGET, "backbone": BACKBONE, "runtime": RUNTIME}
    if receipt.exists():
        row = read(receipt)
        require(row["bound_contract"] == contract and row["checkpoint_sha"] == train.sha(path), f"Fit mismatch: {path}")
        return
    model = train.model_for(variant(method), counts, args.seed, cutoff, parent)
    original_function = train.F
    if method == "S7":
        modules = [model.base.encoder, model.base.heads[train.TARGET]]
        snapshots = [{k: v.to(args.device) for k, v in state_snapshot(module).items()} for module in modules]
        train.F = SimpleNamespace(cross_entropy=lambda logits, y: original_function.cross_entropy(logits, y)
                                  + PLAN["reg_lambda"] * l2_to_snapshot(modules, snapshots, logits.device))
    try:
        trial = out / f"{method}.trial"
        trial_contract = train.trial_contract
        train.trial_contract = lambda *a, **kw: {**trial_contract(*a, **kw), "scope": method, "runtime": RUNTIME}
        model, result = train.fit_trial(variant(method), model, data, args, LRS[method], trial, parent_sha, digest)
    finally:
        train.F = original_function
        train.trial_contract = trial_contract
    peak = result["peak_allocated_bytes"]
    require(peak is None or peak <= 18 * 2**30, f"{method} exceeded memory ceiling")
    train.atomic(path, lambda tmp: torch.save({"model": model.state_dict()}, tmp))
    train.save_json(receipt, {**result, "bound_contract": contract, "checkpoint_sha": train.sha(path),
                              "serialized_update_bytes": update_bytes(model, method)})


def fit_replay(root, parent_dir, counts, data, cutoff, digest, args, smoke, replay_mode=None):
    target, seed = train.TARGET, args.seed
    out = root / str(seed)
    replay = replay_for(target)
    parent = parent_dir / "S0.pt"
    replay.ORIGINAL_RUNS = parent_dir.parent
    replay.source_identity = runtime_source
    original = {"checkpoints": {str(seed): {"S0": train.sha(parent)}}}
    recipe = read(ARCHIVED)
    if not smoke or args.device == "cuda":
        replay.fit(root, seed, replay_mode or ("sanity" if smoke else "fit"), original, recipe)
        report = read(out / "FIT_COMPLETE.json")
        path = out / "replay.pt"
        require(report["checkpoint_sha"] == train.sha(path)
                and report["contract"]["source_sha"] == runtime_source(), "Replay receipt mismatch")
        require(report["peak_allocated_bytes"] <= 18 * 2**30, "Replay exceeded memory ceiling")
        return

    # shortcut: CPU smoke exercises the archived epoch kernel; CUDA smoke uses the formal fitter.
    model = train.model_for("S8", counts, seed, cutoff, parent)
    device = torch.device(args.device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=LRS["Replay"])
    epochs = [replay.training_epoch(model, optimizer, data["train"], seed, i, args.batch_size, device)
              for i in range(args.adapt_epochs)]
    metrics, *_ = train.evaluate(model, target, data["validation"][target], device, args.batch_size)
    path = out / "replay.pt"
    train.atomic(path, lambda tmp: torch.save({"model": model.state_dict()}, tmp))
    train.save_json(out / "FIT_COMPLETE.json", {"validation": metrics, "checkpoint_sha": train.sha(path),
                    "contract": {"source_sha": runtime_source(), "parent_sha": train.sha(parent),
                                 "development_sha": digest, "smoke": True}, "epochs": epochs})


def runtime_source():
    return hashlib.sha256((train.code_fingerprint() + json.dumps(RUNTIME, sort_keys=True)).encode()).hexdigest()


def fit(root, seed, device, smoke):
    tick = time.monotonic()
    counts, data, cutoff, digest = bundle("fit", smoke)
    setup_seconds = time.monotonic() - tick
    a = settings(seed, device, smoke)
    parent_dir = root / "base" / str(seed)
    train.TARGET = "video_games"
    fit_one("S0", parent_dir, counts, data, cutoff, digest, a)
    for target in TARGETS:
        train.TARGET = target
        out, parent = root / target / str(seed), parent_dir / "S0.pt"
        for method in METHODS[1:-1]:
            fit_one(method, out, counts, data, cutoff, digest, a, parent)
        fit_replay(root / target, parent_dir, counts, data, cutoff, digest, a, smoke)
    frozen = {t: {m: train.sha(fitted_artifact(root, t, seed, m)[1]) for m in METHODS} for t in TARGETS}
    receipt = root / f"FIT-{seed}.json"
    if receipt.exists():
        setup_seconds = read(receipt)["setup_seconds"]
    bound_json(receipt, {"backbone": BACKBONE, "runtime": RUNTIME, "source": train.code_fingerprint(), "seed": seed,
                         "smoke": smoke, "fits": frozen, "setup_seconds": setup_seconds})


def retained(base_rank, base_list, rank, top):
    shared = (base_list[:, :, None] == top[:, None, :]).any(axis=2).sum(axis=1)
    hit = lambda r: (r > 0) & (r <= 10)
    weights = 0.1 * 0.9 ** np.arange(10)
    matches = base_list[:, :, None] == top[:, None, :]
    agreement = np.column_stack([matches[:, :k, :k].sum(axis=(1, 2))/k for k in range(1, 11)])
    return {"delta_hit": float(hit(rank).mean() - hit(base_rank).mean()),
            "deep": float((shared <= 5).mean()), "mean_replaced": float((10 - shared).mean()),
            "rbo": float((agreement @ weights + 0.9**10 * agreement[:, -1]).mean())}


def select(by_seed):
    validation = {m: {"ndcg": mean(row[m]["ndcg"] for row in by_seed),
                      "bytes": max(row[m]["bytes"] for row in by_seed),
                      "retained": {d: {k: mean(row[m]["retained"][d][k] for row in by_seed)
                                       for k in ("delta_hit", "deep")}
                                   for d in by_seed[0][m]["retained"]}}
                  for m in METHODS}
    ordered = sorted(METHODS, key=lambda m: (-validation[m]["ndcg"], validation[m]["bytes"] if m in EXACT else float("inf"), m))
    policy = PLAN["policy"]
    safe = [m for m in ordered if all(r["delta_hit"] >= policy["min_hit"] and r["deep"] <= policy["max_deep"]
                                     for r in validation[m]["retained"].values())]
    exact = [m for m in ordered if m in EXACT]
    return {"unrestricted": ordered[0], "joint": safe[0], "exact": exact[0],
            "capped": next(m for m in exact if validation[m]["bytes"] <= policy["cap_bytes"])}, validation


def interval(deltas):
    center = mean(deltas)
    half = 4.302652729911275 * stdev(deltas) / 3**.5
    return {"mean": center, "CI95": [center - half, center + half], "by_seed": deltas}


def fitted_artifact(root, target, seed, method):
    folder = root / ("base" if method == "S0" else target) / str(seed)
    name, receipt = ("replay", "FIT_COMPLETE") if method == "Replay" else (method, method)
    return folder / f"{name}.pt", folder / f"{receipt}.json"


def score(root, target, seed, counts, rows, cutoff, device, smoke, phase):
    train.TARGET = target
    parent, _ = fitted_artifact(root, target, seed, "S0")
    base, results = None, {}
    scores = {}
    for m in METHODS:
        path, receipt = fitted_artifact(root, target, seed, m)
        report = read(receipt)
        contract = report.get("bound_contract", report["contract"])
        digest = train.sha(path)
        source = runtime_source() if m == "Replay" else train.code_fingerprint()
        require(digest == report["checkpoint_sha"] and contract["source_sha"] == source
                and contract.get("data_sha", contract.get("development_sha")) == PLAN["manifests"]["development"]
                and contract["parent_sha"] == (train.sha(parent) if m != "S0" else None)
                and contract.get("smoke", contract.get("mode") == "sanity") == smoke, f"Unbound checkpoint: {path}")
        if m != "Replay":
            require(contract["scope"] == m and contract["backbone"] == BACKBONE and contract["runtime"] == RUNTIME
                    and contract["target"] == (TARGETS[0] if m == "S0" else target),
                    "Wrong checkpoint scope, backbone or target")
        if not smoke:
            require(contract["seed"] == seed and contract["lr"] == LRS[m], "Wrong seed or learning rate")
            require(report["selected_epoch"] in (range(4, 21, 4) if m == "S0" else range(2, 11, 2)),
                    "Wrong checkpoint selection schedule")
            require(contract["epochs"] == (20 if m == "S0" else 10)
                    and contract.get("batch", contract.get("batch_size")) == 2048
                    and contract["limit"] is None, "Wrong training budget")
        model = train.model_for(variant(m), counts, seed, cutoff, parent if m != "S0" else None)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model"])
        model.to(device)
        if m == "S0":
            base = model
        by_domain = {}
        for domain, cohort in rows.items():
            compare = domain != target
            metrics, rank, identity, top = train.evaluate(
                model, domain, cohort, device, 32 if smoke else 512,
                base if compare and m in EXACT[1:] else None, save_top=True)
            if compare and m in EXACT:
                require(identity["mismatch_rows"] == 0, "Non-target score identity violated")
            if compare and m == "S0":
                scores[domain] = rank, top
            if compare:
                pair = retained(*scores[domain], rank, top)
                if m in EXACT:
                    require(np.array_equal(top, scores[domain][1]) and np.array_equal(rank, scores[domain][0]),
                            "Non-target ranking identity violated")
                by_domain[domain] = {**pair, "n": metrics["all"]["n"], "identity": identity}
            else:
                key = "test_counts" if phase == "test" else "validation_counts"
                require(smoke or metrics["long_known"]["n"] == PLAN[key][target], "Target cohort count mismatch")
                target_metrics = metrics
            if phase == "test":
                predictions = root / "predictions" / target / str(seed)
                predictions.mkdir(parents=True, exist_ok=True)
                output = predictions / f"{m}-{domain}.npz"
                def archive(tmp):
                    with tmp.open("wb") as stream:
                        np.savez_compressed(stream, rank=rank, top10=top, user=cohort["user"].numpy(),
                                            gap=cohort["gap"].numpy(), known=cohort["target"].numpy() >= 0,
                                            untied=cohort["untied"].numpy())
                train.atomic(output, archive)
        chosen = target_metrics["long_known"]["ndcg10"]
        if smoke and chosen is None:
            chosen = target_metrics["all_known"]["ndcg10"]
        size = update_bytes(model, m)
        require(m == "Replay" or size == report["serialized_update_bytes"], "Serialized update changed")
        results[m] = {"ndcg": chosen, "bytes": size, "target": target_metrics, "retained": by_domain,
                      "checkpoint_sha256": digest, "fit_sha256": train.sha(receipt)}
    return results


def freeze(root, device, smoke):
    counts, data, cutoff, digest = bundle("freeze", smoke)
    seeds = SEEDS[:1] if smoke else SEEDS
    fit_receipts = {}
    for seed in seeds:
        path = root / f"FIT-{seed}.json"
        rec = read(path)
        require(rec["source"] == train.code_fingerprint() and rec["seed"] == seed
                and rec["smoke"] == smoke and rec["runtime"] == RUNTIME
                and set(rec["fits"]) == set(TARGETS)
                and all(set(fits) == set(METHODS) for fits in rec["fits"].values()), "Fit runtime or scope mismatch")
        for target, fits in rec["fits"].items():
            for method, fit_sha in fits.items():
                require(train.sha(fitted_artifact(root, target, seed, method)[1]) == fit_sha, "Fit changed since completion")
        fit_receipts[str(seed)] = train.sha(path)
    targets = {}
    for target in TARGETS:
        fits = {str(seed): score(root, target, seed, counts, data["validation"], cutoff, device, smoke, "freeze")
                for seed in seeds}
        choices, means = select(list(fits.values()))
        targets[target] = {"choices": choices, "means": means, "by_seed": fits}
    bound_json(root / "freeze.json", {"source": train.code_fingerprint(), "development": digest,
               "test": PLAN["manifests"]["test"], "seeds": list(seeds), "targets": targets, "smoke": smoke,
               "backbone": BACKBONE, "runtime": RUNTIME, "fit_receipts": fit_receipts})


def test(root, seed, device, smoke):
    freeze_path = root / "freeze.json"
    frozen = read(freeze_path)
    seeds = list(SEEDS[:1] if smoke else SEEDS)
    require(frozen["smoke"] == smoke and frozen["seeds"] == seeds and seed in seeds
            and frozen["source"] == train.code_fingerprint() and frozen["test"] == PLAN["manifests"]["test"]
            and frozen["development"] == PLAN["manifests"]["development"]
            and frozen["backbone"] == BACKBONE and frozen["runtime"] == RUNTIME,
            "No matching validation freeze")
    for target in TARGETS:
        for m in METHODS:
            path, receipt = fitted_artifact(root, target, seed, m)
            locked = frozen["targets"][target]["by_seed"][str(seed)][m]
            require(locked["checkpoint_sha256"] == train.sha(path) and locked["fit_sha256"] == train.sha(receipt),
                    f"Fit changed after selection freeze: {path}")
    counts, data, cutoff, _ = bundle("test", smoke)
    output = {}
    for target in TARGETS:
        output[target] = score(root, target, seed, counts, data["validation" if smoke else "test"], cutoff, device,
                               smoke, "test")
    bound_json(root / f"test-{seed}.json", {"seed": seed, "freeze_sha256": train.sha(freeze_path), "targets": output})


def summarize(root):
    path = root / "freeze.json"
    freeze = read(path)
    require(not freeze["smoke"] and freeze["source"] == train.code_fingerprint()
            and tuple(freeze["seeds"]) == SEEDS, "Formal freeze required")
    reports = [read(root / f"test-{seed}.json") for seed in SEEDS]
    require(all(r["freeze_sha256"] == train.sha(path) and r["seed"] == seed
                for r, seed in zip(reports, SEEDS)), "Unbound test receipts")
    output = {}
    for t in TARGETS:
        choices = freeze["targets"][t]["choices"]
        rows = [r["targets"][t] for r in reports]
        ndcg = lambda a, b: interval([100*(r[a]["ndcg"] - r[b]["ndcg"]) for r in rows])
        pair = {d: {m: interval([(r["Replay"]["retained"][d][m] - r["S8"]["retained"][d][m])
                                 * (100 if m != "rbo" else 1) for r in rows])
                    for m in ("delta_hit", "deep", "rbo")}
                for d in rows[0]["Replay"]["retained"]}
        output[t] = {"choices": choices, "mean_target_ndcg_pct": {m: 100*mean(r[m]["ndcg"] for r in rows) for m in METHODS},
                     "vs_base_pp": {m: ndcg(m, "S0") for m in dict.fromkeys(choices.values())},
                     "capped_minus_uncapped_pp": ndcg(choices["capped"], choices["exact"]),
                     "replay_minus_shared": pair}
    bound_json(root / "SUMMARY.json", {"freeze_sha256": train.sha(path), "backbone": BACKBONE,
                                       "runtime": freeze["runtime"], "targets": output})


def resume_smoke(root, device):
    counts, data, cutoff, digest = bundle("fit", True)
    args = settings(SEEDS[0], device, True)
    train.TARGET = TARGETS[0]
    paths = [root / p / str(args.seed) for p in ("clean", "interrupted")]
    require(not any(p.exists() for p in paths), "Use a fresh resume-smoke root")
    fit_one("S0", paths[0], counts, data, cutoff, digest, args)
    atomic, timer = train.atomic, train.time
    ticks = count(step=301)

    class ForcedInterruption(Exception):
        pass

    def first_epoch(path, writer):
        atomic(path, writer)
        if path.name.endswith(".resume.pt"):
            state = torch.load(path, map_location="cpu", weights_only=True)
            require(state["epoch"] == 1, "Wrong interruption boundary")
            raise ForcedInterruption()

    try:
        train.atomic = first_epoch
        train.time = SimpleNamespace(monotonic=lambda: next(ticks))
        fit_one("S0", paths[1], counts, data, cutoff, digest, args)
        raise RuntimeError("Canary did not stop at first save")
    except ForcedInterruption:
        pass
    finally:
        train.atomic, train.time = atomic, timer
    fit_one("S0", paths[1], counts, data, cutoff, digest, args)

    def tensors_equal(name):
        a, b = (torch.load(p / name, map_location="cpu", weights_only=True)["model"] for p in paths)
        return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)

    require(tensors_equal("S0.pt") and tensors_equal("S0.resume.pt"), "Interrupted training changed model tensors")
    replay_checks = {}
    if device.type == "cuda":
        parent = paths[0]
        for target in TARGETS:
            train.TARGET = target
            replay_root = root / target
            replay_args = (parent, counts, data, cutoff, digest, args, True)
            fit_replay(replay_root / "clean", *replay_args)
            try:
                fit_replay(replay_root / "resume", *replay_args, replay_mode="interrupt")
                raise RuntimeError("Replay canary did not interrupt")
            except SystemExit as exc:
                require(exc.code == 75, "Unexpected Replay interruption")
            fit_replay(replay_root / "resume", *replay_args, replay_mode="resume_smoke")
            # Use the archived verifier, which compares checkpoints and traces.
            replay = replay_for(target)
            replay.source_identity = runtime_source
            replay.verify_smoke(replay_root, args.seed)
            replay_checks[target] = train.sha(replay_root / "PASS.json")
    bound_json(root / "RESUME_SMOKE.json", {"device": str(device), "interrupted_epoch": 1,
               "resumed_through_epoch": 2, "model_tensors_equal": True, "source": train.code_fingerprint(),
               "runtime": RUNTIME, "replay": replay_checks})


def ssd_runtime_probe(root, device):
    """Exercise isolation and CUDA backward at the planned full batch size."""
    require(BACKBONE == "shared_ssd4rec" and device.type == "cuda", "Probe requires official SSD4Rec CUDA")
    counts, _, cutoff, _ = bundle("freeze", True)
    train.TARGET = TARGETS[0]
    base = train.model_for("S0", counts, SEEDS[0], cutoff).to(device)
    torch.manual_seed(SEEDS[0])
    x = torch.randn(3, 50, train.WIDTH, device=device)
    mask = torch.arange(50, device=device)[None] < torch.tensor([0, 40, 49], device=device)[:, None]
    with torch.inference_mode(), train.amp(device):
        base.eval()
        batch = base.encoder(x, mask)
        altered = x.clone()
        altered[0] += 2.0
        perturb = base.encoder(altered, mask)
        single = torch.cat([base.encoder(x[i:i+1], mask[i:i+1]) for i in range(len(x))])
        require(torch.isfinite(batch).all().item() and not torch.equal(batch[0], perturb[0]), "Degenerate SSD canary")
        torch.testing.assert_close(batch[1:].float(), perturb[1:].float(), atol=.008, rtol=.008)
        torch.testing.assert_close(batch.float(), single.float(), atol=.02, rtol=.02)
    # Include a 50-token maximum stress case on the largest catalog. Retain the
    # main recipe only if it fits; changing its batch would also change Replay.
    domain = max(counts, key=counts.get)
    train.TARGET = domain
    adapted = train.Adaptation("S6", base).to(device)
    adapted.train()
    adapted.base.eval()
    rows = {"seq": torch.randint(1, counts[domain]+1, (2048, 50), device=device),
            "target": torch.randint(0, counts[domain], (2048,), device=device),
            "gap": torch.zeros(2048, device=device)}
    torch.cuda.reset_peak_memory_stats(device)
    tick = time.monotonic()
    with train.amp(device):
        loss = train.F.cross_entropy(adapted(domain, rows), rows["target"])
    loss.backward()
    grad = adapted.base.heads[domain].item_embedding.weight.grad
    require(grad is not None and bool(grad.isfinite().all()) and bool(grad.ne(0).any())
            and all(p.grad is None for p in adapted.base.encoder.parameters()), "S6 update boundary failed")
    s6_peak = torch.cuda.max_memory_allocated(device)
    require(s6_peak <= 18 * 2**30, "S6 maximum batch exceeded memory ceiling")
    del adapted, base, loss, grad
    shared = train.Adaptation("S8", train.model_for("S0", counts, SEEDS[0], cutoff)).to(device)
    shared.train()
    torch.cuda.reset_peak_memory_stats(device)
    with train.amp(device):
        loss = train.F.cross_entropy(shared(domain, rows), rows["target"])
    loss.backward()
    require(all(p.grad is None or bool(p.grad.isfinite().all()) for p in shared.parameters())
            and any(p.grad is not None and bool(p.grad.ne(0).any()) for p in shared.base.encoder.parameters()),
            "Shared SSD backward failed")
    shared_peak = torch.cuda.max_memory_allocated(device)
    require(shared_peak <= 18 * 2**30, "Shared maximum batch exceeded memory ceiling")
    bound_json(root / "SSD_CUDA_PROBE.json", {"source": train.code_fingerprint(), "runtime": RUNTIME,
               "packed_user_isolation": True, "max_batch": 2048, "catalog": counts[domain],
               "s6_peak": s6_peak, "shared_peak": shared_peak, "seconds": time.monotonic() - tick})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("fit", "freeze", "test", "summarize", "resume-smoke", "ssd-cuda-probe"))
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--seed", type=int, choices=SEEDS, default=SEEDS[0])
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--backbone", choices=("gru", "ssd4rec"), default="gru")
    a = p.parse_args()
    require(a.smoke or a.device == "cuda" or a.phase == "summarize", "Formal training/evaluation uses CUDA")
    require(not a.smoke or a.seed == SEEDS[0], "Smoke uses the first seed")
    torch.set_num_threads(2)
    configure(a.backbone, torch.device(a.device), summary_only=a.phase == "summarize")
    if a.phase == "ssd-cuda-probe":
        ssd_runtime_probe(a.root, torch.device(a.device))
    elif a.phase == "resume-smoke":
        require(a.smoke, "Resume canary requires --smoke")
        resume_smoke(a.root, torch.device(a.device))
    elif a.phase == "summarize":
        summarize(a.root)
    elif a.phase == "freeze":
        freeze(a.root, torch.device(a.device), a.smoke)
    else:
        {"fit": fit, "test": test}[a.phase](a.root, a.seed, torch.device(a.device), a.smoke)


if __name__ == "__main__":
    main()
