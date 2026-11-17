#!/usr/bin/env python3
"""Fit and evaluate the preregistered IP&M adaptation variants."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from src.shared_backbone import DomainItemHead, SharedSASRecEncoder, SharedBackboneDeltaGateAdapter, freeze_module
from scripts.train_shared_backbone_seed42 import TimeConcatTargetModel
from scripts.train_b9_extra_points import GapOnlyDeltaGateAdapter, capacity_matched_gap_only_hidden_dim, deltagate_trainable_params

TARGET = "video_games"
VARIANTS = ("S0", "S4a", "S5", "S6", "S8", "history_only", "gap_only", "continued", "time_start", "time_start_s5", "meantime")
PARENTS = {"time_start_s5": "time_start", **{v:"S0" for v in VARIANTS[1:8]}}
STRICT = ("S4a", "S5", "S6")
SEEDS = (9308, 42096, 169953, 231339, 799822)
WIDTH = 64


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def code_fingerprint():
    root = HERE.parents[1]
    paths = [HERE/n for n in ("train.py","freeze.py","profile_cost.py","collect.py")]
    paths += [root/"scripts"/n for n in ("train_shared_backbone_seed42.py","train_b9_extra_points.py")]
    paths += list((root/"src/shared_backbone").glob("*.py"))
    paths += list((HERE/"vendor_meantime/meantime").rglob("*.py"))
    # BSD tar can create AppleDouble metadata files on Linux. Python never imports them.
    receipt = {str(p.relative_to(root)):sha(p) for p in sorted(paths) if not p.name.startswith("._")}
    return hashlib.sha256(json.dumps(receipt,sort_keys=True).encode()).hexdigest()


def atomic(path, writer):
    temporary = path.with_suffix(path.suffix + ".tmp")
    writer(temporary)
    os.replace(temporary, path)


def save_json(path, value):
    atomic(path, lambda tmp: tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n"))


def load_data(root, windows, limit=None):
    manifest = json.loads((root / "manifest.json").read_text())
    result = {}
    for domain, d in manifest["domains"].items():
        for window in windows:
            record = d["windows"][window]
            path = root / record["file"]
            if sha(path) != record["sha256"]:
                raise RuntimeError(f"Data checksum mismatch: {path}")
            with np.load(path, allow_pickle=False) as archive:
                arrays = {k:torch.from_numpy(archive[k][:limit]) for k in archive.files}
            if limit is None and len(arrays["target"]) != record["rows"]:
                raise RuntimeError(f"Data count mismatch: {path}")
            result.setdefault(window, {})[domain] = arrays
    return {k:d["item_count"] for k,d in manifest["domains"].items()}, result, manifest


class Service(nn.Module):
    def __init__(self, counts, time_aware):
        super().__init__()
        self.encoder = SharedSASRecEncoder(WIDTH, 2, 2, 50, 0.0)
        self.heads = nn.ModuleDict({d:DomainItemHead(n,WIDTH) for d,n in counts.items()})
        self.time = nn.ModuleDict({d:nn.Sequential(nn.Linear(1,WIDTH), nn.GELU(), nn.Linear(WIDTH,WIDTH))
                                   for d in counts} if time_aware else {})

    def hidden(self, domain, batch):
        seq = batch["seq"]
        hidden = self.encoder(self.heads[domain].embed_sequence(seq),seq.eq(0))
        if domain in self.time:
            hidden = hidden + self.time[domain](torch.log1p(batch["gap"]).unsqueeze(-1))
        return hidden

    def forward(self, domain, batch):
        return self.heads[domain].logits(self.hidden(domain,batch))


class Adaptation(nn.Module):
    def __init__(self, variant, base):
        super().__init__()
        self.variant = variant
        self.base = base
        freeze_module(base)
        if variant in ("S4a","S6","S8"):
            self.target = TimeConcatTargetModel(base.encoder,base.heads[TARGET],WIDTH)
        elif variant in ("S5","history_only","time_start_s5"):
            self.target = SharedBackboneDeltaGateAdapter(base.encoder,base.heads[TARGET],WIDTH)
        elif variant == "gap_only":
            n = capacity_matched_gap_only_hidden_dim(WIDTH,deltagate_trainable_params(WIDTH))
            self.target = GapOnlyDeltaGateAdapter(base.encoder,base.heads[TARGET],WIDTH,hidden_dim=n)
        if variant in ("S6","S8","continued"):
            base.heads[TARGET].requires_grad_(True)
        if variant in ("S8","continued"):
            base.encoder.requires_grad_(True)

    def forward(self, domain, batch):
        if domain != TARGET or self.variant == "continued":
            return self.base(domain,batch)
        if self.variant == "time_start_s5":
            hidden = self.base.hidden(domain,batch).detach()
            gap = torch.log1p(batch["gap"]).unsqueeze(-1)
            gate = self.target.gate_mlp(torch.cat((gap,hidden),dim=-1))
            return self.base.heads[domain].logits(gate * hidden + (1-gate)*self.target.global_prior)
        gap = torch.zeros_like(batch["gap"]) if self.variant == "history_only" else batch["gap"]
        result = self.target(batch["seq"],gap)
        return result[0] if isinstance(result,tuple) else result


class Meantime(nn.Module):
    def __init__(self,n,seed,cutoff):
        super().__init__()
        sys.path.insert(0,str(HERE/"vendor_meantime"))
        from meantime.models.transformer_models.meantime import MeantimeModel
        self.n = n
        self.cutoff = cutoff
        self.model = MeantimeModel(SimpleNamespace(
            num_items=n, num_days=cutoff+1, hidden_units=WIDTH, num_blocks=2,
            num_heads=4, dropout=.2, max_len=51, output_info=False,
            absolute_kernel_types="p-d", relative_kernel_types="s-l",
            headtype="dot", head_use_ln=True, model_init_seed=seed,
            model_init_range=.02, time_unit_divide=86400, freq=10000, residual_ln_type="pre",
        ))

    def inputs(self, batch, training):
        seq = batch["seq"]
        mask = torch.full_like(seq[:, :1],self.n+1)
        days = torch.cat((batch["seq_day"],batch["query_day"][:,None]),dim=1)
        tokens = torch.cat((seq,mask),dim=1)
        labels = None
        if training:
            observed = torch.cat((seq,batch["target"][:,None]+1),dim=1)
            random_values = torch.rand(observed.shape,device=seq.device)
            supervised = (observed > 0) & (random_values < .2)
            tokens = torch.where(supervised & (random_values < .16),mask,observed)
            random_items = torch.randint(1,self.n+1,observed.shape,device=seq.device)
            tokens = torch.where(supervised & (random_values >= .16) & (random_values < .18),random_items,tokens)
            labels = torch.where(supervised,observed,0)
        return {"tokens":tokens,"timestamps":days*86400,
                "days":days.clamp(max=self.cutoff),"labels":labels}

    def loss(self,batch):
        inputs = self.inputs(batch,True)
        hidden,_ = self.model.get_logits(inputs)
        selected = inputs["labels"] > 0
        # The boolean indexing keeps a batch dimension even for one masked token.
        labels = inputs["labels"][selected]
        if labels.numel() == 0:
            return hidden.sum()*0
        return F.cross_entropy(self.model.head(hidden[selected]),labels)

    def forward(self,domain,batch):
        hidden,_ = self.model.get_logits(self.inputs(batch,False))
        return self.model.head(hidden[:,-1])[:,1:]


def model_for(variant,counts,seed,cutoff,base_path=None):
    torch.manual_seed(seed)
    if variant == "meantime":
        return Meantime(counts[TARGET],seed,cutoff)
    base = Service(counts,variant in ("time_start","time_start_s5"))
    if variant in PARENTS:
        if base_path is None:
            raise ValueError("Parent checkpoint required")
        base.load_state_dict(torch.load(base_path,map_location="cpu",weights_only=True)["model"])
        torch.manual_seed(seed+981)
        return Adaptation(variant,base)
    return base


def slices(n,size):
    for i in range(0,n,size):
        yield slice(i,min(i+size,n))


def batch_from(rows,index,device,temporal=False):
    keys = ("seq","target","gap","seq_day","query_day") if temporal else ("seq","target","gap")
    return {k:rows[k][index].to(device,dtype=torch.float32 if k=="gap" else torch.long) for k in keys}


def amp(device):
    return torch.autocast("cuda",dtype=torch.bfloat16) if device.type=="cuda" else nullcontext()


def summarize(rank,rows):
    out = {}
    known = rows["target"].numpy() >= 0
    long = rows["gap"].numpy() > 365
    hit = (rank > 0) & (rank <= 10)
    gain = hit / np.log2(np.maximum(rank,1)+1)
    groups = {"all":np.ones(len(rank),dtype=bool),"all_known":known,
              "long_all":long,"long_known":long&known,"long_known_untied":long&known&rows["untied"].numpy()}
    for k,mask in groups.items():
        n = int(mask.sum())
        out[k] = {"n":n,"hit10":float(hit[mask].mean()) if n else None,
                  "ndcg10":float(gain[mask].mean()) if n else None}
    return out


@torch.inference_mode()
def stable_top10(logits):
    bits = logits.float().contiguous().view(torch.int32).to(torch.int64)
    ordered = torch.where(bits < 0, -2147483648 - bits, bits)
    n = logits.shape[1]
    ids = torch.arange(n,device=logits.device)
    key = ordered*(n+1)+(n-ids)
    return key.topk(min(10,n),dim=1).indices


@torch.inference_mode()
def evaluate(model,domain,rows,device,batch_size,reference=None,save_top=False):
    model.eval()
    if reference is not None:
        reference.eval()
    ranks = []
    tops = []
    mismatches = 0
    max_delta = 0.0
    nonfinite_rows = 0
    for index in slices(len(rows["target"]),batch_size):
        batch = batch_from(rows,index,device,isinstance(model,Meantime))
        with amp(device):
            logits = model(domain,batch)
            ref_logits = reference(domain,batch) if reference is not None else None
        target = batch["target"]
        ids = torch.arange(logits.shape[1],device=device)
        safe_target = target.clamp_min(0)
        score = logits.gather(1,safe_target[:,None])
        rank = 1 + ((logits > score) | ((logits == score) & (ids < target[:,None]))).sum(dim=1)
        rank = torch.where(target >= 0,rank,-1)
        ranks.append(rank.cpu().numpy().astype(np.int32))
        if save_top:
            tops.append(stable_top10(logits).cpu().numpy().astype(np.int32))
        nonfinite_rows += int((~torch.isfinite(logits).all(dim=1)).sum())
        if ref_logits is not None:
            delta = (logits.float()-ref_logits.float()).abs()
            mismatches += int(delta.ne(0).any(dim=1).sum())
            max_delta = max(max_delta,float(delta.max()))
    if nonfinite_rows:
        raise RuntimeError(f"Nonfinite evaluation logits in {domain}: {nonfinite_rows} rows")
    ranks = np.concatenate(ranks)
    return (summarize(ranks,rows),ranks,
            {"compared":reference is not None,"mismatch_rows":mismatches,"max_abs":max_delta},
            np.concatenate(tops) if save_top else None)


def trial_contract(variant,args,lr,parent_sha,data_sha):
    temporal = variant == "meantime"
    epochs = args.parent_epochs if variant in ("S0","time_start","meantime") else args.adapt_epochs
    size = min(256,args.batch_size) if temporal else args.batch_size
    return {"seed":args.seed,"lr":lr,"variant":variant,"epochs":epochs,"batch":size,
            "data_sha":data_sha,"parent_sha":parent_sha,"source_sha":code_fingerprint(),
            "torch_version":str(torch.__version__),
            "mode":args.mode,"limit":getattr(args,"limit",None),"device":args.device}


def fit_trial(variant, model, data, args, lr, trial, parent_sha, data_sha):
    start = time.monotonic()
    device = torch.device(args.device)
    model = model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    temporal = variant == "meantime"
    optimizer = torch.optim.Adam(params,lr=lr) if temporal else torch.optim.AdamW(params,lr=lr)
    train = data["train"]
    domains = sorted(train) if variant in ("S0","time_start") else [TARGET]
    contract = trial_contract(variant,args,lr,parent_sha,data_sha)
    epochs, size = contract["epochs"], contract["batch"]
    done = 0
    train_seconds = 0.0
    validation_seconds = 0.0
    best = None
    epoch_log = []
    trial.parent.mkdir(parents=True,exist_ok=True)
    checkpoint = trial.with_suffix(".resume.pt")
    if checkpoint.is_file():
        state = torch.load(checkpoint,map_location=device,weights_only=True)
        if state["contract"] != contract:
            raise RuntimeError("Resume contract drift")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["cpu_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"].cpu(),device=device)
        done = state["epoch"]
        train_seconds = state["train_seconds"]
        validation_seconds = state["validation_seconds"]
        best = state["best"]
        epoch_log = state["epoch_log"]
    last_save = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    baseline_memory = torch.cuda.memory_allocated() if device.type == "cuda" else None
    valid = data["validation"][TARGET]
    interval = epochs if args.mode == "cost" else max(1,epochs//5)
    for epoch in range(done,epochs):
        epoch_start = time.monotonic()
        model.train()
        if isinstance(model,Adaptation) and variant not in ("S8","continued"):
            model.base.eval()
        losses = torch.zeros((),device=device)
        steps = 0
        for domain_i,domain in enumerate(domains):
            rows = train[domain]
            order = np.random.default_rng(args.seed + epoch*8191 + domain_i).permutation(len(rows["target"]))
            for index in slices(len(order),size):
                batch = batch_from(rows,order[index],device,temporal)
                optimizer.zero_grad(set_to_none=True)
                with amp(device):
                    loss = model.loss(batch) if temporal else F.cross_entropy(model(domain,batch),batch["target"])
                loss.backward()
                if temporal:
                    nn.utils.clip_grad_norm_(params,5.0)
                optimizer.step()
                losses.add_(loss.detach())
                steps += 1
        loss_value = float(losses)/steps
        train_seconds += time.monotonic()-epoch_start
        if not math.isfinite(loss_value):
            raise RuntimeError(f"Nonfinite training loss: {variant} epoch {epoch+1}")
        print(json.dumps({"variant":variant,"seed":args.seed,"lr":lr,"epoch":epoch+1,
                          "loss":loss_value,"steps":steps,"elapsed_seconds":time.monotonic()-start}),flush=True)
        epoch_log.append({"epoch":epoch+1,"loss":loss_value})
        if (epoch+1)%interval == 0 or epoch+1 == epochs:
            t0 = time.monotonic()
            metrics,_,_,_ = evaluate(model,TARGET,valid,device,min(args.batch_size,256) if temporal else args.batch_size)
            validation_seconds += time.monotonic()-t0
            score = metrics["long_known"]["ndcg10"]
            if score is None and args.mode == "sanity":
                score = metrics["all_known"]["ndcg10"]
            if score is None:
                raise RuntimeError("Validation selection has no usable labels")
            if best is None or score > best["score"]:
                best = {"score":score,"epoch":epoch+1,"metrics":metrics,
                        "model":{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}}
        if epoch+1==epochs or time.monotonic()-last_save >= 300:
            snapshot = {"model":model.state_dict(),"optimizer":optimizer.state_dict(),
                        "contract":contract,"epoch":epoch+1,"cpu_rng":torch.get_rng_state(),
                        "cuda_rng":torch.cuda.get_rng_state(device) if device.type=="cuda" else None,
                        "train_seconds":train_seconds,"validation_seconds":validation_seconds,
                        "best":best,"epoch_log":epoch_log}
            atomic(checkpoint,lambda tmp:torch.save(snapshot,tmp))
            last_save = time.monotonic()
    model.load_state_dict(best["model"])
    return model,{"contract":contract,"train_seconds":train_seconds,"validation_seconds":validation_seconds,
                  "validation":best["metrics"],"selected_epoch":best["epoch"],"epochs":epoch_log,
                  "trainable_params":sum(p.numel() for p in params),
                  "baseline_allocated_bytes":baseline_memory,
                  "peak_allocated_bytes":torch.cuda.max_memory_allocated() if device.type=="cuda" else None}


def fit(args):
    if args.mode == "tune" and args.seed != 9308:
        raise ValueError("Validation tuning is bound to seed 9308")
    if args.mode == "fit" and args.seed not in SEEDS[1:]:
        raise ValueError("Use the locked recipe for the four confirmation seeds")
    counts,data,manifest = load_data(args.data,("train","validation"),args.limit)
    source = sha(args.data/"manifest.json")
    cutoff = manifest["cutoff_days_inclusive"][0]
    recipe = json.loads(args.recipe.read_text()) if args.recipe else None
    if args.mode == "fit" and (recipe is None or recipe["development_sha"] != source or recipe["source_sha"] != code_fingerprint()):
        raise ValueError("Missing validation recipe or provenance mismatch")
    choices = {}
    for variant in VARIANTS:
        parent = args.out/f"{PARENTS[variant]}.pt" if variant in PARENTS else None
        parent_sha = sha(parent) if parent else None
        best_path = args.out/f"{variant}.pt"
        receipt = args.out/f"{variant}.json"
        lrs = [3e-4,1e-3,3e-3] if args.mode=="tune" else [recipe["lrs"][variant] if recipe else 1e-3]
        if best_path.is_file() and receipt.is_file():
            choice = json.loads(receipt.read_text())
            if (choice["checkpoint_sha"] != sha(best_path) or choice["contract"]["lr"] not in lrs
                or choice["contract"] != trial_contract(variant,args,choice["contract"]["lr"],parent_sha,source)):
                raise RuntimeError(f"Completed variant drift: {variant}")
            choices[variant] = choice["contract"]["lr"]
            continue
        best = None
        best_score = -float("inf")
        for lr in lrs:
            model = model_for(variant,counts,args.seed,cutoff,parent)
            trial = args.out/"trials"/f"{variant}_{lr:.3e}".replace(".","p")
            model,result = fit_trial(variant,model,data,args,lr,trial,parent_sha,source)
            score = result["validation"]["long_known"]["ndcg10"]
            if score is None:
                if args.mode != "sanity":
                    raise RuntimeError("No validation long-gap labels")
                score = result["validation"]["all_known"]["ndcg10"]
            save_json(trial.with_suffix(".json"),result)
            if score > best_score:
                best_score,best = score,result
                atomic(best_path,lambda tmp:torch.save({"model":model.state_dict()},tmp))
            del model
        best["checkpoint_sha"] = sha(best_path)
        save_json(receipt,best)
        choices[variant] = best["contract"]["lr"]
    if args.mode == "tune":
        save_json(args.out/"recipe.json",{"seed":args.seed,"development_sha":source,
                  "source_sha":code_fingerprint(),"lrs":choices,"selection":"validation long_known NDCG@10",
                  "b0_sha256":manifest["b0_sha256"],"cutoff_days_inclusive":manifest["cutoff_days_inclusive"],
                  "catalog_sizes":counts})
    save_json(args.out/"FIT_COMPLETE.json",{"mode":args.mode,"seed":args.seed,"development_sha":source,"lrs":choices})


def test(args):
    if not args.recipe or not args.freeze:
        raise ValueError("Test needs locked validation recipe and freeze manifest")
    recipe = json.loads(args.recipe.read_text())
    freeze = json.loads(args.freeze.read_text())
    if freeze["recipe_sha"] != sha(args.recipe) or sorted(freeze["seeds"]) != sorted(SEEDS):
        raise ValueError("Test freeze receipt mismatch")
    if freeze["test_manifest_sha"] != sha(args.data/"manifest.json") or recipe["source_sha"] != code_fingerprint():
        raise ValueError("Test data or executable differs from freeze receipt")
    counts,data,manifest = load_data(args.data,("test",))
    if (manifest["b0_sha256"] != recipe["b0_sha256"]
        or manifest["cutoff_days_inclusive"] != recipe["cutoff_days_inclusive"]
        or counts != recipe["catalog_sizes"]):
        raise ValueError("Test and development split identity mismatch")
    device = torch.device(args.device)
    cutoff = manifest["cutoff_days_inclusive"][0]
    for variant in VARIANTS:
        path = args.out/f"{variant}.pt"
        receipt = json.loads((args.out/f"{variant}.json").read_text())
        if sha(path)!=receipt["checkpoint_sha"] or receipt["contract"]["data_sha"]!=recipe["development_sha"]:
            raise RuntimeError("Checkpoint provenance mismatch")
        if freeze["checkpoints"][str(args.seed)][variant] != sha(path):
            raise RuntimeError("Checkpoint is not bound to freeze receipt")
        model = model_for(variant,counts,args.seed,cutoff,args.out/f"{PARENTS[variant]}.pt" if variant in PARENTS else None)
        model.load_state_dict(torch.load(path,map_location="cpu",weights_only=True)["model"])
        model = model.to(device)
        reference = None
        if variant in STRICT:
            reference = model_for("S0",counts,args.seed,cutoff).to(device)
            reference.load_state_dict(torch.load(args.out/"S0.pt",map_location=device,weights_only=True)["model"])
        metrics, identities = {}, {}
        for domain,rows in data["test"].items():
            if variant=="meantime" and domain!=TARGET:
                continue
            metrics[domain],rank,identities[domain],top10 = evaluate(
                model,domain,rows,device,min(args.batch_size,256 if variant=="meantime" else 512),
                reference if domain!=TARGET else None,save_top=True)
            np.savez_compressed(args.out/f"test_{variant}_{domain}.npz",rank=rank,user=rows["user"].numpy(),
                                gap=rows["gap"].numpy(),known=rows["target"].numpy()>=0,
                                untied=rows["untied"].numpy(),top10=top10)
            if identities[domain]["mismatch_rows"]:
                raise RuntimeError(f"Strict retained output contract failed: {variant} {domain}")
        save_json(args.out/f"test_{variant}.json",{"metrics":metrics,"retained_identity":identities,
                  "checkpoint_sha":sha(path),"test_data_sha":sha(args.data/"manifest.json")})
        print("TEST",variant,flush=True)
    save_json(args.out/"TEST_COMPLETE.json",{"seed":args.seed,"freeze_sha":sha(args.freeze)})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("sanity","tune","fit","test"))
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--recipe",type=Path)
    p.add_argument("--freeze",type=Path)
    p.add_argument("--device",default="cuda")
    p.add_argument("--batch-size",type=int,default=2048)
    p.add_argument("--parent-epochs",type=int,default=20)
    p.add_argument("--adapt-epochs",type=int,default=10)
    p.add_argument("--limit",type=int)
    args = p.parse_args()
    if args.mode != "sanity" and (args.limit is not None or args.parent_epochs!=20 or args.adapt_epochs!=10):
        raise ValueError("Formal run shape differs from the plan")
    if args.mode != "sanity" and args.device!="cuda":
        raise ValueError("Formal run requires CUDA")
    torch.set_float32_matmul_precision("high")
    # The fused eval path can return NaNs for left padding plus the causal mask.
    torch.backends.mha.set_fastpath_enabled(False)
    args.out.mkdir(parents=True,exist_ok=True)
    (test if args.mode=="test" else fit)(args)


if __name__ == "__main__":
    main()
