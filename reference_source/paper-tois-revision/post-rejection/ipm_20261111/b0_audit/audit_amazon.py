#!/usr/bin/env python3
"""CPU-only metadata feasibility audit for the proposed IP&M split."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import groupby
import gzip
import hashlib
import json
import math
from pathlib import Path
import pickle
import time

DOMAINS = ("Video_Games", "Movies_and_TV", "CDs_and_Vinyl")
DAY = 86400
MIN_TRAIN_HISTORY = 1
MIN_EVAL_HISTORY = 4
MAX_HISTORY = 50


def day_date(day):
    return datetime.fromtimestamp(day * DAY, timezone.utc).date().isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def load_projection(path):
    with path.open("rb") as stream:
        return pickle.load(stream)


def project(raw_path, out):
    users = defaultdict(list)
    items = {}
    days = Counter()
    not_midnight = 0
    review_date_mismatches = 0
    sampled_review_dates = 0
    with gzip.open(raw_path, "rb") as stream:
        for index, line in enumerate(stream):
            row = json.loads(line)
            ts = row["unixReviewTime"]
            day = ts // DAY
            not_midnight += (ts % DAY != 0)
            if index % 1000 == 0:
                sampled_review_dates += 1
                date = datetime.strptime(row["reviewTime"], "%m %d, %Y").date().isoformat()
                review_date_mismatches += (date != day_date(day))
            user = row["reviewerID"]
            item = row["asin"]
            item_index = items.setdefault(item, len(items))
            users[user].append((day, item_index))
            days[day] += 1
    if not_midnight:
        raise ValueError(f"{raw_path} contains {not_midnight} non-midnight timestamps")
    with out.open("wb") as stream:
        pickle.dump((dict(users), items), stream, protocol=pickle.HIGHEST_PROTOCOL)
    return days, {
        "raw_path": str(raw_path), "sha256": sha256(raw_path),
        "bytes": raw_path.stat().st_size, "rows": sum(days.values()),
        "users": len(users), "items": len(items),
        "first_utc_date": day_date(min(days)), "last_utc_date": day_date(max(days)),
        "unixReviewTime_all_utc_midnight": not not_midnight,
        "reviewTime_date_mismatches_sampled": review_date_mismatches,
        "reviewTime_dates_checked": sampled_review_dates,
    }


def quantile_day(days, q):
    rank = math.ceil(sum(days.values()) * q)
    cumulative = 0
    for day in sorted(days):
        cumulative += days[day]
        if cumulative >= rank:
            return day
    raise ValueError("No quantile day")


def bucket(gap):
    for threshold, label in ((30, "1-30"), (90, "31-90"), (180, "91-180"),
                             (365, "181-365"), (730, "366-730")):
        if gap <= threshold:
            return label
    return "731+"


def split_counts(users, train_last, val_last):
    stats = {name: Counter() for name in ("train", "validation", "test")}
    gaps = {name: Counter() for name in stats}
    known_gaps = {name: Counter() for name in stats}
    windows = {name: [] for name in stats}
    vocab = {item for events in users.values() for day, item in events if day <= train_last}
    user_windows = {name: set() for name in stats}
    duplicates = 0
    first_day_ties = {name: Counter() for name in stats}
    for user, raw_events in users.items():
        events = sorted(dict.fromkeys(raw_events), key=lambda x: x[0])
        duplicates += len(raw_events) - len(events)
        history = []
        observed = set()
        first_window_seen = set()
        for day, group in groupby(events, key=lambda x: x[0]):
            targets = [item for _, item in group]
            window = "train" if day <= train_last else "validation" if day <= val_last else "test"
            stat = stats[window]
            stat["unique_interactions"] += len(targets)
            user_windows[window].add(user)
            windows[window].append(day)
            if window == "train" or window not in first_window_seen:
                first_window_seen.add(window)
                chosen = targets if window == "train" else targets[:1]
                if window != "train":
                    first_day_ties[window]["users_multiple_targets_on_first_day"] += len(targets) > 1
                min_history = MIN_TRAIN_HISTORY if window == "train" else MIN_EVAL_HISTORY
                known_history = sum(item in vocab for _, item in history[-MAX_HISTORY:])
                if len(history) < min_history:
                    stat["excluded_no_earlier_history"] += len(chosen)
                elif known_history < min_history:
                    stat["excluded_no_encodable_history"] += len(chosen)
                else:
                    gap = day - history[-1][0]
                    assert gap > 0
                    for target in chosen:
                        stat["eligible"] += 1
                        stat["long_gap_gt365"] += gap > 365
                        stat["repeat_of_earlier_item"] += target in observed
                        stat["history_ge4_raw"] += len(history) >= 4
                        stat["long_gap_history_ge4_raw"] += gap > 365 and len(history) >= 4
                        stat["eligible_multiple_targets_same_day"] += len(targets) > 1
                        gaps[window][bucket(gap)] += 1
                        if target in vocab:
                            stat["known_catalog_target"] += 1
                            stat["long_gap_known_catalog"] += gap > 365
                            known_gaps[window][bucket(gap)] += 1
                            stat["long_gap_known_untied"] += gap > 365 and len(targets) == 1
                        else:
                            stat["oov_target"] += 1
                            stat["long_gap_oov_target"] += gap > 365
            history.extend((day, item) for item in targets)
            observed.update(targets)
    return {
        "train_catalog_items": len(vocab), "duplicate_user_item_day_rows": duplicates,
        "windows": {
            name: {**stat, **first_day_ties[name], "gap_buckets": dict(gaps[name]),
                   "known_target_gap_buckets": dict(known_gaps[name]),
                   "distinct_active_users": len(user_windows[name]),
                   "first_utc_date": day_date(min(windows[name])),
                   "last_utc_date": day_date(max(windows[name]))}
            for name, stat in stats.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("scratch_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    combined = Counter()
    sources = {}
    for domain in DOMAINS:
        counts, source = project(args.raw_dir / f"{domain}_5.json.gz",
                                 args.scratch_dir / f"{domain}.pkl")
        combined.update(counts)
        sources[domain] = source
        print(domain, source["rows"], source["first_utc_date"], source["last_utc_date"], flush=True)
    train_last = quantile_day(combined, .70)
    val_last = quantile_day(combined, .85)
    domains = {}
    users_by_domain = {}
    for domain in DOMAINS:
        users, items = load_projection(args.scratch_dir / f"{domain}.pkl")
        domains[domain] = split_counts(users, train_last, val_last)
        users_by_domain[domain] = set(users)
        del users, items
        print(domain, {k: (v["eligible"], v.get("long_gap_gt365", 0))
                       for k, v in domains[domain]["windows"].items()}, flush=True)
    vg = domains["Video_Games"]["windows"]
    write_json(args.out_dir / "SPLIT_MANIFEST.json", {
        "status": "metadata_feasible" if vg["test"]["long_gap_known_catalog"] >= 2000
                  and vg["validation"]["long_gap_known_catalog"] >= 500 else "insufficient_counts",
        "raw_files": sources, "domains": domains,
        "cutoff_basis": "nearest-rank day quantiles over all raw rows from the three domains, before review deduplication",
        "train_last_inclusive_utc": day_date(train_last),
        "validation_first_inclusive_utc": day_date(train_last + 1),
        "validation_last_inclusive_utc": day_date(val_last),
        "test_first_inclusive_utc": day_date(val_last + 1),
        "train_exclusive_unix_seconds": (train_last + 1) * DAY,
        "test_start_unix_seconds": (val_last + 1) * DAY,
        "fraction_raw_through_train": sum(n for d, n in combined.items() if d <= train_last) / sum(combined.values()),
        "fraction_raw_through_validation": sum(n for d, n in combined.items() if d <= val_last) / sum(combined.values()),
        "duplicate_policy": "first raw occurrence for each user/item/UTC day",
        "within_day_policy": "history strictly earlier UTC days; validation/test first window day by date, first raw row on that day; no later-target substitution for an ineligible first event",
        "known_history_policy": "use the last 50 earlier raw deduplicated events, retain train-catalog events in order; require 1 for training and 4 for validation/test; gap measured from last raw earlier day",
        "candidate_policy": "domain train-known catalog; no history-item masking; flag OOV targets",
        "all_three_raw_user_overlap": len(set.intersection(*users_by_domain.values())),
        "cross_domain_raw_user_overlap": {
            f"{a}:{b}": len(users_by_domain[a] & users_by_domain[b])
            for i, a in enumerate(DOMAINS) for b in DOMAINS[i + 1:]
        },
        "runtime_seconds": round(time.monotonic() - started, 2),
        "scope": "Metadata only. No model scores, training, or materialized experimental splits.",
    })


if __name__ == "__main__":
    main()
