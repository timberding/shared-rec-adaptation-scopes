"""Fit only the six absent Base checkpoints using the original 20-epoch recipe."""
import argparse
from pathlib import Path
from types import SimpleNamespace

from paths import GROUPS, PLAN, PLAN_PATH, SEEDS, base_paths, parent_family, read, require, sha
from runtime import configure, inventory, rows_for, torch


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("family")
    cli.add_argument("seed", type=int, choices=SEEDS)
    args = cli.parse_args()
    family, seed = args.family, args.seed
    require(inventory()["parents"][family][str(seed)].get("new"), "Parent already exists in the archive")
    group = next(g for g in GROUPS if parent_family(g) == family)
    train, runtime = configure(group)
    train.TARGET = "shopping" if family == "yelp-sasrec" else "video_games"
    weights, receipt = base_paths(family, seed)
    weights.parent.mkdir(parents=True, exist_ok=True)
    settings = SimpleNamespace(seed=seed, device="cuda", batch_size=PLAN["batch_size"],
                               parent_epochs=PLAN["base_epochs"], adapt_epochs=10, limit=None,
                               mode=f"ipm100-parent-{sha(Path(__file__))}")
    lr = PLAN["learning_rates"]["S0"]
    counts, data, cutoff, digest = rows_for(group, "fit")
    contract = train.trial_contract("S0", settings, lr, None, digest)
    if receipt.is_file():
        past = read(receipt)
        require(past["contract"] == contract and past["checkpoint_sha"] == sha(weights),
                "Completed Base fit differs")
        return
    network = train.model_for("S0", counts, seed, cutoff)
    fitted, metrics = train.fit_trial("S0", network, data, settings, lr,
                                      weights.with_suffix(".trial"), None, digest)
    train.atomic(weights, lambda tmp: torch.save({"model": fitted.state_dict()}, tmp))
    train.save_json(receipt, {**metrics, "checkpoint_sha": sha(weights),
                             "plan_sha": sha(PLAN_PATH), "runtime": runtime})
    print(f"BASE_COMPLETE {family} {seed}", flush=True)


if __name__ == "__main__":
    main()
