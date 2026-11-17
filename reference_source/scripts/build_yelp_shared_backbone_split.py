#!/usr/bin/env python3
"""Build a deterministic Yelp shared-backbone split for TOIS external evidence.

The builder maps Yelp businesses into disjoint coarse service domains by
category priority, then writes the same processed pickle schema consumed by the
shared-backbone runner. It also includes ``val_samples`` for later
dataset-level recipe tuning; the existing runner ignores that field during
bounded smoke tests.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DOMAIN_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("yelp_restaurants", ("Restaurants",)),
    ("yelp_shopping", ("Shopping",)),
    ("yelp_beauty_spas", ("Beauty & Spas",)),
    ("yelp_food", ("Food",)),
    ("yelp_automotive", ("Automotive",)),
    ("yelp_home_services", ("Home Services",)),
    ("yelp_event_planning", ("Event Planning & Services",)),
    ("yelp_health_medical", ("Health & Medical",)),
    ("yelp_nightlife", ("Nightlife",)),
    ("yelp_active_life", ("Active Life",)),
]

DEFAULT_DOMAINS = ["yelp_shopping", "yelp_restaurants", "yelp_beauty_spas"]
DEFAULT_TARGET = "yelp_shopping"


def parse_day(value: str) -> int:
    return dt.date.fromisoformat(value[:10]).toordinal()


def day_to_iso(value: int | None) -> str | None:
    if value is None:
        return None
    return dt.date.fromordinal(value).isoformat()


def gap_bucket(days: int) -> str:
    if days > 365:
        return ">365d"
    if days > 180:
        return "180-365d"
    if days > 90:
        return "90-180d"
    if days > 30:
        return "30-90d"
    return "short"


def stable_bucket(value: str, modulus: int) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big") % modulus


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def category_set(value: str | None) -> set[str]:
    if not value:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def assign_domain(categories: set[str], selected_domains: set[str]) -> str | None:
    for domain, required_categories in DOMAIN_RULES:
        if domain not in selected_domains:
            continue
        if any(category in categories for category in required_categories):
            return domain
    return None


def load_business_domains(
    business_path: Path,
    selected_domains: set[str],
) -> tuple[dict[str, str], dict[str, Any]]:
    business_to_domain: dict[str, str] = {}
    domain_businesses: Counter[str] = Counter()
    category_collision_counts: Counter[str] = Counter()
    rows = 0
    assigned = 0

    for row in iter_jsonl(business_path):
        rows += 1
        categories = category_set(row.get("categories"))
        matched = [
            domain
            for domain, required in DOMAIN_RULES
            if domain in selected_domains and any(category in categories for category in required)
        ]
        if len(matched) > 1:
            category_collision_counts["+".join(matched)] += 1
        domain = assign_domain(categories, selected_domains)
        if domain is None:
            continue
        business_to_domain[str(row["business_id"])] = domain
        domain_businesses[domain] += 1
        assigned += 1

    return business_to_domain, {
        "business_rows": rows,
        "assigned_businesses": assigned,
        "domain_businesses": dict(sorted(domain_businesses.items())),
        "category_priority": [domain for domain, _ in DOMAIN_RULES if domain in selected_domains],
        "category_collisions_top20": dict(category_collision_counts.most_common(20)),
    }


def build_item_maps(events: list[tuple[int, str, str]]) -> tuple[dict[str, int], list[str]]:
    items = sorted({item for _, _, item in events})
    item_to_idx = {item: idx + 1 for idx, item in enumerate(items)}
    return item_to_idx, items


def make_sample(
    user: str,
    day: int,
    item: str,
    history_items: list[str],
    history_days: list[int],
    item_to_idx: dict[str, int],
) -> dict[str, Any] | None:
    if item not in item_to_idx:
        return None
    seq_idx = [item_to_idx[value] for value in history_items if value in item_to_idx]
    if not seq_idx:
        return None
    gap_days = int(day - history_days[-1])
    intervals = [
        float(history_days[idx] - history_days[idx - 1])
        for idx in range(1, len(history_days))
    ]
    return {
        "user_id": user,
        "seq_idx": seq_idx,
        "target_idx": item_to_idx[item],
        "gap_days": float(gap_days),
        "bucket": gap_bucket(gap_days),
        "intervals": intervals,
    }


def build_samples(
    events: list[tuple[int, str, str]],
    item_to_idx: dict[str, int],
    min_history: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    per_user: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for day, user, item in events:
        per_user[user].append((day, item))

    train_samples: list[dict[str, Any]] = []
    val_samples: list[dict[str, Any]] = []
    test_samples: list[dict[str, Any]] = []
    train_items: set[int] = set()
    bucket_counts_train: Counter[str] = Counter()
    bucket_counts_val: Counter[str] = Counter()
    bucket_counts_test: Counter[str] = Counter()
    sequence_lengths: list[int] = []
    skipped_missing_idx = 0
    pre_val_items: set[int] = set()
    pre_test_items: set[int] = set()

    for user, user_events in per_user.items():
        user_events.sort(key=lambda value: (value[0], value[1]))
        sequence_lengths.append(len(user_events))
        if len(user_events) < min_history + 1:
            continue

        eligible_last_idx = len(user_events) - 1
        eligible_val_idx = len(user_events) - 2 if len(user_events) >= min_history + 2 else None
        for idx, (_, item) in enumerate(user_events):
            item_idx = item_to_idx.get(item)
            if item_idx is None:
                continue
            if eligible_val_idx is not None and idx < eligible_val_idx:
                pre_val_items.add(item_idx)
            if idx < eligible_last_idx:
                pre_test_items.add(item_idx)

        for idx in range(min_history, len(user_events)):
            day, item = user_events[idx]
            history = user_events[:idx]
            sample = make_sample(
                user,
                day,
                item,
                [history_item for _, history_item in history],
                [history_day for history_day, _ in history],
                item_to_idx,
            )
            if sample is None:
                skipped_missing_idx += 1
                continue
            if idx == eligible_last_idx:
                test_samples.append(sample)
                bucket_counts_test[sample["bucket"]] += 1
            elif eligible_val_idx is not None and idx == eligible_val_idx:
                val_samples.append(sample)
                bucket_counts_val[sample["bucket"]] += 1
            else:
                train_samples.append(sample)
                train_items.add(int(sample["target_idx"]))
                bucket_counts_train[sample["bucket"]] += 1

    test_train_unseen = sum(1 for sample in test_samples if int(sample["target_idx"]) not in train_items)
    val_train_unseen = sum(1 for sample in val_samples if int(sample["target_idx"]) not in train_items)
    test_pretest_unseen = sum(1 for sample in test_samples if int(sample["target_idx"]) not in pre_test_items)
    val_preval_unseen = sum(1 for sample in val_samples if int(sample["target_idx"]) not in pre_val_items)
    summary = {
        "users": len(per_user),
        "events": len(events),
        "users_ge_min_history_plus_target": sum(1 for value in sequence_lengths if value >= min_history + 1),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "test_samples": len(test_samples),
        "train_bucket_counts": dict(sorted(bucket_counts_train.items())),
        "val_bucket_counts": dict(sorted(bucket_counts_val.items())),
        "test_bucket_counts": dict(sorted(bucket_counts_test.items())),
        "test_pretest_unseen": test_pretest_unseen,
        "test_pretest_unseen_share": (test_pretest_unseen / len(test_samples)) if test_samples else None,
        "val_preval_unseen": val_preval_unseen,
        "val_preval_unseen_share": (val_preval_unseen / len(val_samples)) if val_samples else None,
        "strict_test_train_target_unseen": test_train_unseen,
        "strict_test_train_target_unseen_share": (test_train_unseen / len(test_samples)) if test_samples else None,
        "strict_val_train_target_unseen": val_train_unseen,
        "strict_val_train_target_unseen_share": (val_train_unseen / len(val_samples)) if val_samples else None,
        "skipped_missing_idx": skipped_missing_idx,
        "sequence_length_max": max(sequence_lengths) if sequence_lengths else 0,
    }
    return train_samples, val_samples, test_samples, summary


def write_config(
    output: Path,
    processed_root: Path,
    domains: list[str],
    target_domain: str,
    seed: int,
) -> None:
    retained = [domain for domain in domains if domain != target_domain]
    config = {
        "name": f"yelp_shared_backbone_seed{seed}",
        "seed": seed,
        "status": "yelp_protocol_split_ready",
        "run_root": "runs/yelp_shared_backbone_seed42096",
        "report_root": "reports/yelp_shared_backbone_seed42096",
        "dataset": {
            "processed_root": str(processed_root),
            "target_domain": target_domain,
            "retained_domains": retained,
            "domains": {
                domain: {
                    "processed_file": f"{domain}_processed.pkl",
                    "role": "target" if domain == target_domain else "retained_service",
                    **({"primary_bucket": ">365d"} if domain == target_domain else {}),
                }
                for domain in domains
            },
        },
        "model_surface": {
            "shared_encoder": ["position_embedding", "transformer_or_gru", "layer_norm"],
            "domain_specific": ["item_embedding", "output_head"],
            "target_adapter": ["gate_mlp", "global_prior"],
            "surface_rule": "Yelp businesses are assigned to disjoint category-derived service domains by a fixed priority rule.",
        },
        "comparisons": [
            {"id": "m0_shared_sasrec_base", "trainable": ["shared_encoder", "domain_specific_heads"], "evaluation": ["target", "retained"]},
            {"id": "m1_timeconcat_e2e_target_finetune", "trainable": ["shared_encoder", "target_domain_head", "time_feature_surface"], "evaluation": ["target", "retained"]},
            {"id": "m1_frozen_target_timeconcat", "trainable": ["target_domain_head", "time_feature_surface"], "evaluation": ["target", "retained"]},
            {"id": "m2_frozen_deltagate_adapter", "trainable": ["gate_mlp", "global_prior"], "evaluation": ["target", "retained"]},
        ],
        "metrics": {
            "target_primary": [f"{target_domain}_>365d_hit10", f"{target_domain}_>365d_ndcg10"],
            "target_secondary": [f"{target_domain}_all_hit10", f"{target_domain}_all_ndcg10"],
            "retention": [f"{domain}_all_{metric}_delta_vs_m0" for domain in retained for metric in ("hit10", "ndcg10")],
            "cost": ["trainable_parameter_count"],
        },
        "stop_gates": [
            {"id": "g0_data_contract", "pass_condition": "processed files expose the shared-backbone schema"},
            {"id": "g1_tiny_overfit", "pass_condition": "bounded runner smoke passes"},
            {"id": "g2_m0_service_baseline", "pass_condition": "base metrics are finite"},
            {"id": "g3_timeconcat_tradeoff", "pass_condition": "full TimeConcat comparator is visible"},
            {"id": "g4_deltagate_no_encoder_drift", "pass_condition": "frozen DeltaGate contract passes"},
            {"id": "g5_claim_gate", "pass_condition": "Yelp evidence remains gated until dataset-level validation tuning is frozen"},
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yelp-root", type=Path, default=Path("data/raw/yelp"))
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/yelp_p_y1"))
    parser.add_argument("--domains", nargs="+", default=DEFAULT_DOMAINS)
    parser.add_argument("--target-domain", default=DEFAULT_TARGET)
    parser.add_argument("--min-history", type=int, default=2)
    parser.add_argument("--max-review-rows", type=int, default=0)
    parser.add_argument("--user-mod", type=int, default=4)
    parser.add_argument("--user-bucket", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42096)
    parser.add_argument("--config-out", type=Path, default=Path("configs/yelp_shared_backbone_protocol.json"))
    parser.add_argument("--summary-out", type=Path, default=Path("reports/yelp_shared_backbone_split_summary.json"))
    args = parser.parse_args()

    selected_domains = set(args.domains)
    if args.target_domain not in selected_domains:
        raise ValueError("--target-domain must be included in --domains")
    if args.user_mod and (args.user_bucket < 0 or args.user_bucket >= args.user_mod):
        raise ValueError("--user-bucket must be in [0, --user-mod)")

    business_to_domain, business_summary = load_business_domains(
        args.yelp_root / "yelp_academic_dataset_business.json",
        selected_domains,
    )
    domain_events: dict[str, list[tuple[int, str, str]]] = {domain: [] for domain in args.domains}
    domain_users: dict[str, set[str]] = {domain: set() for domain in args.domains}
    min_day: int | None = None
    max_day: int | None = None
    rows = 0
    selected_rows = 0
    user_bucket_skipped = 0
    unmapped_business_skipped = 0

    for row in iter_jsonl(args.yelp_root / "yelp_academic_dataset_review.json"):
        rows += 1
        if args.max_review_rows and rows > args.max_review_rows:
            break
        user = str(row["user_id"])
        if args.user_mod and stable_bucket(user, args.user_mod) != args.user_bucket:
            user_bucket_skipped += 1
            continue
        domain = business_to_domain.get(str(row["business_id"]))
        if domain is None:
            unmapped_business_skipped += 1
            continue
        day = parse_day(str(row["date"]))
        item = str(row["business_id"])
        domain_events[domain].append((day, user, item))
        domain_users[domain].add(user)
        min_day = day if min_day is None else min(min_day, day)
        max_day = day if max_day is None else max(max_day, day)
        selected_rows += 1

    if max_day is None:
        raise RuntimeError("no selected Yelp reviews found")

    args.output_root.mkdir(parents=True, exist_ok=True)
    domain_summaries: dict[str, Any] = {}
    for domain, events in domain_events.items():
        item_to_idx, idx_to_item = build_item_maps(events)
        train_samples, val_samples, test_samples, summary = build_samples(events, item_to_idx, args.min_history)
        payload = {
            "train_samples": train_samples,
            "val_samples": val_samples,
            "test_samples": test_samples,
            "item_to_idx": item_to_idx,
            "idx_to_item": idx_to_item,
            "item_vocab_size": len(idx_to_item),
        }
        with (args.output_root / f"{domain}_processed.pkl").open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        domain_summaries[domain] = {
            **summary,
            "item_vocab_size": len(idx_to_item),
            "processed_file": f"{domain}_processed.pkl",
        }

    retained_domains = [domain for domain in args.domains if domain != args.target_domain]
    user_overlaps = {
        f"{args.target_domain}__{domain}": len(domain_users[args.target_domain] & domain_users[domain])
        for domain in retained_domains
    }
    retained_overlap = len(set.intersection(*(domain_users[domain] for domain in args.domains))) if args.domains else 0

    write_config(args.config_out, args.output_root, args.domains, args.target_domain, args.seed)
    report = {
        "status": "pass",
        "protocol": "P-Y1 Yelp shopping target with restaurants and beauty/spas retained",
        "yelp_root": str(args.yelp_root),
        "output_root": str(args.output_root),
        "config_out": str(args.config_out),
        "rows_scanned": rows,
        "selected_rows": selected_rows,
        "user_mod": args.user_mod,
        "user_bucket": args.user_bucket,
        "user_bucket_skipped": user_bucket_skipped,
        "unmapped_business_skipped_after_bucket": unmapped_business_skipped,
        "min_day": min_day,
        "max_day": max_day,
        "min_date": day_to_iso(min_day),
        "max_date": day_to_iso(max_day),
        "min_history": args.min_history,
        "business_mapping": business_summary,
        "domains": domain_summaries,
        "target_retained_user_overlap": user_overlaps,
        "all_three_domain_user_overlap": retained_overlap,
        "claim_boundary": "Yelp is split-builder and validation-tuning candidate evidence only until a dataset-level recipe is frozen.",
    }
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "summary": str(args.summary_out), "config": str(args.config_out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
