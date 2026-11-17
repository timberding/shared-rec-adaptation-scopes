#!/usr/bin/env python3
"""Measure one controlled full adaptation replay from the frozen seed 42096 parent."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from train import fit_trial, load_data, model_for, save_json, sha, trial_contract, TARGET

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("variant", choices=("S4a","S5","S6","S8"))
p.add_argument("replicate", type=int, choices=(1,2,3))
p.add_argument("--data", type=Path, required=True)
p.add_argument("--runs", type=Path, required=True)
args = p.parse_args()
recipe_path = args.runs / "9308/recipe.json"
recipe = json.loads(recipe_path.read_text())
freeze = json.loads((args.runs / "fit_weights.json").read_text())
parent = args.runs / "42096/S0.pt"
if sha(parent) != freeze["checkpoints"]["42096"]["S0"] or sha(recipe_path)!=freeze["recipe_sha"]:
    raise RuntimeError("Cost-profile parent mismatch")
torch.backends.mha.set_fastpath_enabled(False)
torch.set_float32_matmul_precision("high")
counts, data, manifest = load_data(args.data, ("train", "validation"))
model = model_for(args.variant,counts,42096,manifest["cutoff_days_inclusive"][0],parent)
settings = SimpleNamespace(seed=42096,device="cuda",parent_epochs=20,adapt_epochs=10,batch_size=2048,mode="cost")
output = args.runs / "cost" / f"{args.variant}_rep{args.replicate}"
receipt = output.with_suffix(".json")
if receipt.exists():
    row = json.loads(receipt.read_text())
    expected = trial_contract(args.variant,settings,recipe["lrs"][args.variant],sha(parent),sha(args.data/"manifest.json"))
    if row["contract"] != expected or row["replicate"] != args.replicate:
        raise RuntimeError("Completed cost replay provenance mismatch")
    raise SystemExit(0)
model, result = fit_trial(args.variant,model,data,settings,recipe["lrs"][args.variant],output,sha(parent),sha(args.data/"manifest.json"))
payload = {name:tensor.detach().cpu() for name,tensor in model.named_parameters() if tensor.requires_grad}
serialized = output.with_suffix(".update.pt")
torch.save(payload,serialized)
result["serialized_update_bytes"] = serialized.stat().st_size
result["gpu_name"] = torch.cuda.get_device_name()
result["replicate"] = args.replicate
if args.variant == "S8":
    copy_state = {k:v for k,v in model.state_dict().items() if k.startswith(("base.encoder.","base.heads."+TARGET+".","target.time_projection."))}
    target_copy = output.with_suffix(".target_copy.pt")
    torch.save(copy_state,target_copy)
    result["target_copy_file_bytes"] = target_copy.stat().st_size
    result["target_copy_tensor_bytes"] = sum(v.numel()*v.element_size() for v in copy_state.values())
    target_copy.unlink()
save_json(receipt,result)
serialized.unlink()
output.with_suffix(".resume.pt").unlink()
