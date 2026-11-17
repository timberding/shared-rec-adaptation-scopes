"""Matched IP&M follow-up fits using the frozen chronological training code."""

import argparse
from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F


HERE = Path(__file__).resolve().parent
PLAN_PATH = HERE / "PLAN.json"
PLAN = json.loads(PLAN_PATH.read_text())
RUNS = HERE / "runs"
WORKSPACE_ROOT = Path("/artifact")
CORE = WORKSPACE_ROOT / "deltagate_ipm_global_time_20261111_r1/experiments/ipm_global_time_20261111"
YELP = WORKSPACE_ROOT / "deltagate_ipm_yelp_20261112_r1"
AMAZON = ("video_games", "cds_vinyl")
DISTILL = PLAN["distillation"]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(path.read_text())


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def replay_module(target):
    directory = "replay_vg_20261111" if target == "video_games" else "cds_replay_s7_20261112"
    replay = load_module("archived_replay", CORE / directory / "replay.py")
    require(replay.source_identity() == PLAN["archived_replay_source"][target],
            f"Archived {target} replay executable differs from fitted runs")
    return replay


def configure(target, seed):
    sys.path.insert(0, str(CORE))
    import train
    require(train.code_fingerprint() == PLAN["core_source"], "Original training code changed")
    train.TARGET = target
    if target in AMAZON:
        frozen = CORE / "runs/freeze.json"
        require(train.sha(frozen) == PLAN["amazon_parent_freeze"], "Amazon freeze changed")
        expected = read(frozen)["checkpoints"][str(seed)]["S0"]
        parent = CORE / "runs" / str(seed) / "S0.pt"
        data_root, source = CORE / "data", "amazon"
    else:
        expected = PLAN["yelp_parent_weights"][str(seed)]
        parent = YELP / "runs" / str(seed) / "S0.pt"
        data_root, source = YELP / "data", "yelp"
    require(train.sha(parent) == expected, f"Base bytes changed for {target} {seed}")
    source_hash = sha256(
        (train.code_fingerprint() + train.sha(PLAN_PATH) + train.sha(Path(__file__))).encode()
    ).hexdigest()
    torch.backends.mha.set_fastpath_enabled(False)
    torch.set_float32_matmul_precision("high")
    return SimpleNamespace(train=train, parent=parent, parent_sha=expected, target=target,
                           data_root=data_root, source=source, source_hash=source_hash)


def load_data(ctx, windows, limit=None):
    window = "test" if windows == ("test",) else "development"
    root = ctx.data_root / ("test_sealed" if window == "test" else window)
    digest = PLAN[f"{ctx.source}_{window}"]
    require(ctx.train.sha(root / "manifest.json") == digest, f"Data manifest changed: {root}")
    counts, data, manifest = ctx.train.load_data(root, windows, limit)
    for split in data.values():
        for rows in split.values():
            rows["seq"] = rows["seq"].to(torch.int32)
    return counts, data, manifest["cutoff_days_inclusive"][0], digest


def build_model(ctx, method, counts, seed, cutoff):
    variant = {"AdditiveFixed": "S4a", "S7": "S8"}.get(method, method)
    if method == "Replay" or method.startswith("Distill_"):
        variant = "S8"
    train = ctx.train
    model = train.model_for(variant, counts, seed, cutoff, ctx.parent)
    if method == "S0":
        model.load_state_dict(torch.load(ctx.parent, map_location="cpu", weights_only=True)["model"])
    if method == "AdditiveFixed":
        def set_constant_gap(_module, inputs):
            sequence, gap = inputs
            return sequence, torch.full_like(gap, PLAN["constant_gap_days"])
        model.target.register_forward_pre_hook(set_constant_gap)
    return model, variant


def fit_standard(ctx, method, seed, counts, data, cutoff, digest, smoke=False):
    root = (HERE / "smoke" if smoke else RUNS) / ctx.target / str(seed)
    root.mkdir(parents=True, exist_ok=True)
    weights, receipt = root / f"{method}.pt", root / f"{method}.json"
    train = ctx.train
    args = SimpleNamespace(seed=seed, device="cuda", parent_epochs=20,
                           adapt_epochs=2 if smoke else PLAN["epochs"],
                           batch_size=PLAN["batch"], mode="sanity" if smoke else "fit",
                           limit=2048 if smoke else None)
    model, variant = build_model(ctx, method, counts, seed, cutoff)
    lr = PLAN["learning_rates"]["S4a" if method == "AdditiveFixed" else method]
    original_fingerprint = train.code_fingerprint
    train.code_fingerprint = lambda: ctx.source_hash
    original_f = train.F
    try:
        contract = train.trial_contract(variant, args, lr, ctx.parent_sha, digest)
        if receipt.exists():
            prior = read(receipt)
            require(prior["method"] == method and prior["contract"] == contract
                    and prior["checkpoint_sha"] == train.sha(weights), "Standard fit changed")
            return
        if method == "S7":
            from scripts.train_shared_backbone_seed42 import l2_to_snapshot, state_snapshot
            modules = [model.base.encoder, model.base.heads[ctx.target]]
            device = torch.device(args.device)
            snapshots = [{k: v.to(device) for k, v in state_snapshot(module).items()}
                         for module in modules]

            def regularized_loss(logits, labels):
                return F.cross_entropy(logits, labels) + PLAN["regularization"] * l2_to_snapshot(
                    modules, snapshots, device
                )
            train.F = SimpleNamespace(cross_entropy=regularized_loss)
        model, fitted = train.fit_trial(variant, model, data, args, lr, root / method,
                                        ctx.parent_sha, digest)
        train.atomic(weights, lambda tmp: torch.save({"model": model.state_dict()}, tmp))
        temp_update = root / f"{method}.update.pt"
        torch.save({k: p.detach().cpu() for k, p in model.named_parameters() if p.requires_grad},
                   temp_update)
        fitted.update(method=method, checkpoint_sha=train.sha(weights),
                      serialized_update_bytes=temp_update.stat().st_size)
        temp_update.unlink()
        train.save_json(receipt, fitted)
    finally:
        train.F = original_f
        train.code_fingerprint = original_fingerprint


def logit_kl(student, teacher, temperature):
    with torch.autocast(device_type=student.device.type, enabled=False):
        student_logp = F.log_softmax(student.float() / temperature, dim=1)
        teacher_logp = F.log_softmax(teacher.detach().float() / temperature, dim=1)
        return F.kl_div(student_logp, teacher_logp, log_target=True,
                        reduction="batchmean") * temperature**2


def install_distillation(replay, weight):
    """Keep the archived minibatch loop and replace only its non-target loss."""
    original_fit_model = replay.train.model_for

    def distill_model(*args, **kwargs):
        model = original_fit_model(*args, **kwargs)
        teacher = deepcopy(model.base).requires_grad_(False).to("cuda").eval()
        forward = model.forward
        teacher_logits = None

        def with_teacher(domain, batch):
            nonlocal teacher_logits
            if domain != replay.TARGET and model.training:
                with torch.no_grad():
                    teacher_logits = teacher(domain, batch)
            return forward(domain, batch)

        def loss(student, labels):
            nonlocal teacher_logits
            if teacher_logits is None:
                return F.cross_entropy(student, labels)
            result = logit_kl(student, teacher_logits, DISTILL["temperature"])
            teacher_logits = None
            return result

        model.forward = with_teacher
        replay.F = SimpleNamespace(cross_entropy=loss)
        return model

    replay.PLAN = {**replay.PLAN, "replay_loss_weight": weight}
    replay.train.model_for = distill_model


def distill_method(weight):
    return f"Distill_{weight:g}"


def replay_fingerprint(ctx, replay, weight):
    return sha256((ctx.source_hash + ctx.train.sha(Path(replay.__file__))
                   + ctx.train.sha(replay.PLAN_PATH) + str(weight)).encode()).hexdigest()


def fit_replay(ctx, seed, mode="fit", weight=None, name="clean"):
    replay = replay_module(ctx.target)
    expected_digest = PLAN["amazon_development"]
    recipe = read(CORE / "runs/9308/recipe.json")
    require(recipe["development_sha"] == expected_digest, "Archived replay split differs")
    require(replay.PLAN["learning_rate"] == PLAN["learning_rates"]["S8"]
            and replay.PLAN["training_epochs"] == PLAN["epochs"]
            and replay.PLAN["batch_size"] == PLAN["batch"], "Archived replay schedule differs")
    identity = replay_fingerprint(ctx, replay, weight)
    replay.source_identity = lambda: identity
    method = "Replay" if weight is None else distill_method(weight)
    root = (RUNS if mode == "fit" else HERE / "smoke" / name) / ctx.target / method
    saved_model_for = replay.train.model_for
    if weight is not None:
        require(ctx.target == DISTILL["target"], "Distillation sampling is fixed to Video Games")
        install_distillation(replay, weight)
    try:
        replay.fit(root, seed, mode, {"checkpoints": {str(seed): {"S0": ctx.parent_sha}}}, recipe)
    finally:
        replay.train.model_for = saved_model_for


def artifacts(target, seed, method):
    from_replay = method == "Replay" or method.startswith("Distill_")
    if from_replay:
        root = RUNS / target / method / str(seed)
        return root / "replay.pt", root / "FIT_COMPLETE.json"
    root = RUNS / target / str(seed)
    return root / f"{method}.pt", root / f"{method}.json"


def lock_distill():
    sys.path.insert(0, str(CORE))
    from train import save_json, sha
    target, seed = DISTILL["target"], DISTILL["calibration_seed"]
    ctx = configure(target, seed)
    replay = replay_module(target)
    records = {}
    scores = {}
    for weight in DISTILL["weights"]:
        p, receipt = artifacts(target, seed, distill_method(weight))
        record = read(receipt)
        require(record["checkpoint_sha"] == sha(p)
                and record["contract"]["source_sha"] == replay_fingerprint(ctx, replay, weight)
                and record["contract"]["seed"] == seed and record["contract"]["limit"] is None
                and record["contract"]["parent_sha"] == ctx.parent_sha, "Calibration weight changed")
        key = str(weight)
        records[key] = sha(receipt)
        scores[key] = record["validation"]["long_known"]["ndcg10"]
    require(all(math.isfinite(s) for s in scores.values()), "Calibration scores must be finite")
    best = min(DISTILL["weights"], key=lambda w: (-scores[str(w)], w != 1.0, w))
    value = {"seed": seed, "validation": scores, "fits": records, "chosen_weight": best}
    path = RUNS / "DISTILL_WEIGHTS.json"
    if path.exists():
        require(read(path) == value, "Locked distillation calibration changed")
    else:
        save_json(path, value)
    print(json.dumps(value), flush=True)


def scored_methods(target):
    methods = PLAN["standard_fits"][target]
    if target == "video_games":
        chosen = read(RUNS / "DISTILL_WEIGHTS.json")["chosen_weight"]
        methods = methods + sorted({distill_method(DISTILL["primary_weight"]), distill_method(chosen)})
    return methods


def freeze():
    sys.path.insert(0, str(CORE))
    from train import save_json, sha
    lock_distill()
    checkpoints = {}
    for target, seeds in PLAN["seeds"].items():
        for seed in seeds:
            ctx = configure(target, seed)
            group = checkpoints.setdefault(target, {})
            expected_dev = PLAN[f"{ctx.source}_development"]
            group[str(seed)] = {}
            for method in scored_methods(target):
                weights, receipt = artifacts(target, seed, method)
                fitted = read(receipt)
                contract = fitted["contract"]
                expected_source = ctx.source_hash
                if method == "Replay" or method.startswith("Distill_"):
                    weight = None if method == "Replay" else float(method.split("_")[-1])
                    expected_source = replay_fingerprint(ctx, replay_module(target), weight)
                require(contract["seed"] == seed and contract["parent_sha"] == ctx.parent_sha
                        and contract["source_sha"] == expected_source
                        and (contract.get("data_sha") or contract.get("development_sha")) == expected_dev
                        and contract["epochs"] == PLAN["epochs"] and contract["limit"] is None
                        and fitted["selected_epoch"] in (2, 4, 6, 8, 10)
                        and fitted["checkpoint_sha"] == sha(weights), f"Fit identity: {method} {seed}")
                group[str(seed)][method] = {"weights_sha": fitted["checkpoint_sha"],
                                            "fit_sha": sha(receipt)}
    frozen = {"checkpoints": checkpoints, "source_sha": sha(Path(__file__)),
              "plan_sha": sha(PLAN_PATH), "distill_choice": sha(RUNS / "DISTILL_WEIGHTS.json")}
    path = RUNS / "FREEZE.json"
    if path.exists():
        require(read(path) == frozen, "Held-out lock changed")
    else:
        save_json(path, frozen)


def score(ctx, seed, window):
    train = ctx.train
    locked = RUNS / "FREEZE.json"
    frozen = read(locked)
    require(frozen["source_sha"] == train.sha(Path(__file__)) and frozen["plan_sha"] == train.sha(PLAN_PATH)
            and frozen["distill_choice"] == train.sha(RUNS / "DISTILL_WEIGHTS.json"),
            "Held-out lock changed")
    if window == "test":
        policy = read(RUNS / "POLICY.json")
        require(policy["model_freeze_sha"] == train.sha(locked), "Policy freeze changed")
    out = RUNS / ctx.target / str(seed) / window
    out.mkdir(parents=True, exist_ok=True)
    completion = out / "COMPLETE.json"
    if completion.exists():
        prior = read(completion)
        require(prior["freeze_sha"] == train.sha(locked) and prior["seed"] == seed,
                f"Prior {window} differs from the freeze")
        if window == "test":
            for method, row in prior["methods"].items():
                for domain, record in row["domains"].items():
                    require(train.sha(out/f"{method}_{domain}.npz") == record["sha"],
                            f"Prior {method} {domain} predictions changed")
        return
    counts, data, cutoff, digest = load_data(ctx, (window,))
    device = torch.device("cuda")
    base, _ = build_model(ctx, "S0", counts, seed, cutoff)
    base.to(device).eval()
    methods = scored_methods(ctx.target)
    if window == "validation" and ctx.target == "video_games":
        methods = [name for name in methods if name.startswith("Distill_")]
    if window == "validation" or ctx.target == "cds_vinyl":
        methods = ["S0", *methods]
    receipt = {}
    base_outputs = {}
    for method in methods:
        identity = frozen["checkpoints"][ctx.target][str(seed)].get(method, {})
        if method == "S0":
            model, weight_sha = base, ctx.parent_sha
        else:
            weights, fit = artifacts(ctx.target, seed, method)
            require(identity == {"weights_sha": train.sha(weights), "fit_sha": train.sha(fit)},
                    f"Frozen {method} differs")
            model, _ = build_model(ctx, method, counts, seed, cutoff)
            model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True)["model"])
            model.to(device)
            weight_sha = identity["weights_sha"]
        compact = method in ("AdditiveFixed", "history_only", "gap_only", "S4a", "S5", "S6")
        records = {}
        for domain, rows in data[window].items():
            non_target = domain != ctx.target
            if non_target and model is not base:
                require(all(torch.equal(value, model.base.heads[domain].state_dict()[k])
                            for k, value in base.heads[domain].state_dict().items()),
                        f"Frozen {domain} head changed")
            metrics, ranks, fixed, top10 = train.evaluate(
                model, domain, rows, device, PLAN["eval_batch"],
                reference=base if compact and non_target else None, save_top=True
            )
            require(not fixed["mismatch_rows"] and fixed["max_abs"] == 0, f"{method} changed {domain}")
            records[domain] = {"metrics": metrics, "identity": fixed}
            if window == "validation":
                if model is base:
                    base_outputs[domain] = top10
                original_hit = receipt.get("S0", {}).get("domains", {}).get(domain)
                if non_target and original_hit is not None:
                    shared = (top10[:, :, None] == base_outputs[domain][:, None, :]).any(axis=2)
                    records[domain]["deep"] = float((shared.sum(axis=1) <= 5).mean())
                    records[domain]["delta_hit10"] = (metrics["all"]["hit10"]
                                                      - original_hit["metrics"]["all"]["hit10"])
                continue
            path = out / f"{method}_{domain}.npz"
            def write(tmp):
                with tmp.open("wb") as stream:
                    np.savez_compressed(stream, rank=ranks, top10=top10, user=rows["user"].numpy(),
                                        gap=rows["gap"].numpy(), known=rows["target"].numpy() >= 0,
                                        untied=rows["untied"].numpy())
            train.atomic(path, write)
            records[domain]["sha"] = train.sha(path)
        receipt[method] = {"weights_sha": weight_sha, "domains": records}
        del model
        print(f"{window.upper()} {ctx.target} {seed} {method}", flush=True)
    train.save_json(completion, {"seed": seed, "data_sha": digest,
                                 "freeze_sha": train.sha(locked), "methods": receipt})


def verify_resume(ctx, seed):
    method = distill_method(DISTILL["primary_weight"])
    clean = HERE / "smoke" / "clean" / ctx.target / method / str(seed)
    resumed = HERE / "smoke" / "resume" / ctx.target / method / str(seed)
    a = torch.load(clean / "replay.pt", map_location="cpu", weights_only=True)["model"]
    b = torch.load(resumed / "replay.pt", map_location="cpu", weights_only=True)["model"]
    require(a.keys() == b.keys() and all(torch.equal(v, b[k]) for k, v in a.items()),
            "Resumed selected weights are different")
    lhs, rhs = read(clean / "FIT_COMPLETE.json"), read(resumed / "FIT_COMPLETE.json")
    for key in ("contract", "validation", "selected_epoch", "epochs"):
        require(lhs[key] == rhs[key], f"Resumed {key} changed")
    ctx.train.save_json(HERE / "CANARY.json", {"seed": seed, "resume_equal": True,
                                             "source_sha": ctx.source_hash,
                                             "checkpoint_sha": lhs["checkpoint_sha"]})


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action", choices=("fit", "smoke", "distill", "interrupt", "resume_smoke",
                                        "verify_resume", "calibrate", "freeze", "validation", "test"))
    cli.add_argument("--target", choices=PLAN["seeds"], default="video_games")
    cli.add_argument("--seed", type=int, default=DISTILL["calibration_seed"])
    cli.add_argument("--weight", type=float, choices=DISTILL["weights"], default=1.0)
    cli.add_argument("--method")
    args = cli.parse_args()
    if args.action in ("calibrate", "freeze"):
        (lock_distill if args.action == "calibrate" else freeze)()
        return
    require(args.seed in PLAN["seeds"][args.target], "Unexpected seed")
    ctx = configure(args.target, args.seed)
    if args.action in ("validation", "test"):
        score(ctx, args.seed, args.action)
        return
    if args.action == "verify_resume":
        verify_resume(ctx, args.seed)
        return
    if args.action in ("distill", "interrupt", "resume_smoke") or (
            args.action == "smoke" and args.method == "distill"):
        mode = {"distill": "fit", "smoke": "sanity"}.get(args.action, args.action)
        fit_replay(ctx, args.seed, mode, args.weight,
                   name="clean" if mode == "sanity" else "resume")
        return
    smoke = args.action == "smoke"
    methods = PLAN["standard_fits"][args.target]
    if args.method:
        require(args.method in methods, "Unexpected standard method")
        methods = [args.method]
    start = time.monotonic()
    if any(method != "Replay" for method in methods):
        counts, data, cutoff, digest = load_data(ctx, ("train", "validation"), 2048 if smoke else None)
        for method in methods:
            if method != "Replay":
                fit_standard(ctx, method, args.seed, counts, data, cutoff, digest, smoke)
        del data
    if "Replay" in methods:
        fit_replay(ctx, args.seed, "sanity" if smoke else "fit")
    print(f"FIT {args.target} {args.seed} {time.monotonic() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
