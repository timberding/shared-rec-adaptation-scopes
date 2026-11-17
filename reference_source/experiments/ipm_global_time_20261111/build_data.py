#!/usr/bin/env python3
"""Materialize the B0 split as compact train/validation and separate test bundles."""
import argparse
from collections import Counter
from itertools import groupby
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

B0 = Path(__file__).resolve().parents[2] / "paper-tois-revision/post-rejection/ipm_20261111/b0_audit"
sys.path.insert(0, str(B0))
from audit_amazon import DAY, DOMAINS, MAX_HISTORY, project, load_projection, sha256, write_json

NAMES = dict(zip(DOMAINS, ("video_games", "movies_tv", "cds_vinyl")))
WINDOWS = ("train", "validation", "test")


def materialize(users, items, expected, cutoffs, out, name):
    train_last, val_last = cutoffs
    used = {item for events in users.values() for day, item in events if day <= train_last}
    # Map train-known items by ASIN order. Zero is padding and -1 is unknown target.
    known = {item: i + 1 for i, item in enumerate(sorted(used, key={v:k for k,v in items.items()}.__getitem__))}
    assert len(known) == expected["train_catalog_items"]
    arrays = {}
    for window in WINDOWS:
        n = expected["windows"][window]["eligible"]
        arrays[window] = {
            "seq": np.zeros((n, MAX_HISTORY), dtype=np.int32),
            "seq_day": np.zeros((n, MAX_HISTORY), dtype=np.int32) if name == "video_games" else None,
            "target": np.empty(n, dtype=np.int32),
            "query_day": np.empty(n, dtype=np.int32),
            "gap": np.empty(n, dtype=np.int32),
            "user": np.empty(n, dtype=np.int32),
            "untied": np.empty(n, dtype=bool),
        }
    cursor = Counter()
    for user_num, events in enumerate(users.values()):
        history = []
        visited = set()
        for day, group in groupby(sorted(dict.fromkeys(events), key=lambda row: row[0]), key=lambda row: row[0]):
            targets = [item for _, item in group]
            window = "train" if day <= train_last else "validation" if day <= val_last else "test"
            if window == "train" or window not in visited:
                visited.add(window)
                prefix = [(ts,known[item]) for ts,item in history[-MAX_HISTORY:] if item in known]
                if len(prefix) >= (1 if window == "train" else 4):
                    a = arrays[window]
                    gap = day - history[-1][0]
                    assert gap > 0 and prefix[-1][0] < day
                    for item in targets if window == "train" else targets[:1]:
                        pos = cursor[window]
                        a["seq"][pos, -len(prefix):] = [idx for _,idx in prefix]
                        if a["seq_day"] is not None:
                            a["seq_day"][pos, -len(prefix):] = [ts for ts,_ in prefix]
                        a["target"][pos] = known[item] - 1 if item in known else -1
                        a["query_day"][pos] = day
                        a["gap"][pos] = gap
                        a["user"][pos] = user_num
                        a["untied"][pos] = len(targets) == 1
                        cursor[window] += 1
            history.extend((day,item) for item in targets)
    entries = {}
    for window, a in arrays.items():
        n = cursor[window]
        want = expected["windows"][window]
        assert n == want["eligible"]
        assert ((a["target"] >= 0) & (a["gap"] > 365)).sum() == want["long_gap_known_catalog"]
        assert ((a["target"] >= 0) & (a["gap"] > 365) & a["untied"]).sum() == want["long_gap_known_untied"]
        assert (a["target"] < 0).sum() == want.get("oov_target", 0)
        if window == "train":
            assert a["query_day"].max() <= train_last and np.all(a["target"] >= 0)
        elif window == "validation":
            assert a["query_day"].min() > train_last and a["query_day"].max() <= val_last
        else:
            assert a["query_day"].min() > val_last
        if window != "train":
            assert len(set(a["user"].tolist())) == n
        if a["seq_day"] is not None:
            assert np.all(a["seq_day"] < a["query_day"][:,None])
        assert a["seq"].max() <= len(known)
        stage = "test_sealed" if window == "test" else "development"
        path = out / stage / f"{name}_{window}.npz"
        np.savez_compressed(path, **{k:v for k,v in a.items() if v is not None})
        entries[window] = {"file":path.name,"rows":n,"sha256":sha256(path),"bytes":path.stat().st_size}
    return {"item_count":len(known),"windows":entries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    reference = B0 / "SPLIT_MANIFEST.json"
    split = json.loads(reference.read_text())
    cutoffs = (split["train_exclusive_unix_seconds"] // DAY - 1, split["test_start_unix_seconds"] // DAY - 1)
    args.out.mkdir(parents=True, exist_ok=False)
    for stage in ("development", "test_sealed"):
        (args.out / stage).mkdir()
    manifest = {"b0_sha256":sha256(reference),"cutoff_days_inclusive":cutoffs,"domains":{}}
    with tempfile.TemporaryDirectory(prefix="ipm-development-") as temp:
        projection = Path(temp) / "projection.pkl"
        for domain in DOMAINS:
            source = split["raw_files"][domain]
            _, receipt = project(Path(source["raw_path"]),projection)
            if receipt["sha256"] != source["sha256"]:
                raise RuntimeError(f"Raw input identity mismatch in {domain}")
            users, items = load_projection(projection)
            name = NAMES[domain]
            manifest["domains"][name] = materialize(users,items,split["domains"][domain],cutoffs,args.out,name)
            del users, items
            print(name, manifest["domains"][name], flush=True)
    write_json(args.out / "development/manifest.json", {
        **manifest, "domains":{d:{"item_count":v["item_count"],"windows":{w:r for w,r in v["windows"].items() if w!="test"}}
                              for d,v in manifest["domains"].items()}})
    write_json(args.out / "test_sealed/manifest.json", {
        **manifest, "domains":{d:{"item_count":v["item_count"],"windows":{"test":v["windows"]["test"]}}
                              for d,v in manifest["domains"].items()}})


if __name__ == "__main__":
    main()
