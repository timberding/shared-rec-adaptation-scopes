"""Count Yelp P-Y1 domains under the IP&M first-return protocol, without fitting."""
from datetime import date
from hashlib import sha256
import json
from pathlib import Path
import runpy
import sys

import numpy as np

project = Path(sys.argv[1])
raw = Path(sys.argv[2])
out = Path(sys.argv[3])
expected = json.loads(Path(sys.argv[4]).read_text())
builder = runpy.run_path(str(project / "scripts/build_yelp_shared_backbone_split.py"))
domains = tuple(builder["DEFAULT_DOMAINS"])
business = raw / "yelp_academic_dataset_business.json"
reviews = raw / "yelp_academic_dataset_review.json"
mapping, audit = builder["load_business_domains"](business, set(domains))
catalog = {item: i for i, item in enumerate(mapping)}
item_domain = np.array([domains.index(mapping[item]) for item in catalog], dtype=np.uint8)
start_val = date(2020, 1, 1).toordinal()
start_test = date(2021, 1, 1).toordinal()
end = date(2022, 1, 1).toordinal()
source = sha256()
capacity = 7_000_000
users = np.empty(capacity, dtype=np.int32)
items = np.empty_like(users)
dates = np.empty_like(users)
user_ids = {}
n = 0
scan_count = 0
with reviews.open("rb") as stream:
    for line in stream:
        source.update(line)
        scan_count += 1
        row = json.loads(line)
        item = catalog.get(row["business_id"])
        if item is None:
            continue
        when = date.fromisoformat(row["date"][:10]).toordinal()
        if when >= end:
            continue
        user = row["user_id"]
        users[n] = user_ids.setdefault(user, len(user_ids))
        items[n] = item
        dates[n] = when
        n += 1
users, items, dates = (x[:n] for x in (users, items, dates))
del user_ids
service = item_domain[items]
known_item = np.zeros(len(catalog), dtype=bool)
known_item[items[dates < start_val]] = True

# Eliminate repeated same-day observations of the same item. Retain raw order
# for distinct ties before reconstructing each user/domain chronology.
order = np.lexsort((items, dates, users, service))
keys = (items[order], dates[order], users[order], service[order])
distinct = np.ones(n, dtype=bool)
distinct[1:] = np.logical_or.reduce([x[1:] != x[:-1] for x in keys])
raw_order = np.sort(order[distinct])
order = raw_order[np.lexsort((dates[raw_order], users[raw_order], service[raw_order]))]
users, items, dates, service = (x[order] for x in (users, items, dates, service))
first = np.r_[True, (users[1:] != users[:-1]) | (service[1:] != service[:-1])]
day_first = np.r_[True, first[1:] | (dates[1:] != dates[:-1])]
positions = np.arange(len(items))
starts = np.maximum.accumulate(np.where(first, positions, 0))
seen_before = np.r_[0, np.cumsum(known_item[items], dtype=np.int32)]
gap = np.r_[0, dates[1:] - dates[:-1]]
next_same_date = np.r_[~day_first[1:], False]
out_domains = {}
eligible_users = {}
for code, domain in enumerate(domains):
    domain_rows = service == code
    by_window = {}
    for name, start, stop in [("validation", start_val, start_test), ("test", start_test, end)]:
        in_window = (dates >= start) & (dates < stop)
        i = positions[domain_rows & in_window & (first | np.r_[True, ~in_window[:-1]])]
        assert np.all(day_first[i])
        i = i[(seen_before[i] - seen_before[np.maximum(starts[i], i - 50)]) >= 4]
        i = i[~first[i]]
        assert np.all(gap[i] > 0) and len(np.unique(users[i])) == len(i)
        l = i[gap[i] > 365]
        covered = l[known_item[items[l]]]
        by_window[name] = {
            "eligible":len(i), "long_gap_eligible":len(l), "long_gap_known":len(covered),
            "long_gap_known_untied":int((~next_same_date[covered]).sum()),
        }
        eligible_users[domain, name] = users[i]
    # All distinct training targets with a strictly earlier day in the domain.
    first_day = dates[starts]
    by_window["train"] = {
        "catalog_items":int((known_item & (item_domain == code)).sum()),
        "supervised_prefixes":int((domain_rows & (dates < start_val) & (dates > first_day)).sum()),
    }
    out_domains[domain] = by_window

def overlap(name):
    return {f"{a}__{b}":len(np.intersect1d(eligible_users[a,name], eligible_users[b,name], assume_unique=True))
            for j, a in enumerate(domains) for b in domains[j+1:]}

assert source.hexdigest() == expected["review_sha256"] and scan_count == expected["scanned_review_rows"]
assert out_domains == expected["domains"]
assert all(overlap(k) == expected["eligible_user_overlap"][k] for k in ("validation", "test"))
out.mkdir(exist_ok=False)
dev = out / "development"
sealed = out / "test_sealed"
for folder in (dev, sealed):
    folder.mkdir()
meta = {"raw_review_sha256": source.hexdigest(), "cutoff_days_inclusive":[start_val-1,start_test-1],
        "domains":{}}
event_start = np.maximum.accumulate(np.where(day_first, positions, 0))
for code, domain in enumerate(domains):
    name = domain.removeprefix("yelp_")
    take = service == code
    ordered_catalog = sorted(
        (idx for idx, item in enumerate(catalog) if known_item[idx] and item_domain[idx] == code),
        key=list(catalog).__getitem__,
    )
    count = len(ordered_catalog)
    assert count < 65536 and count == out_domains[domain]["train"]["catalog_items"]
    idx = np.zeros(len(catalog), dtype=np.uint16)
    idx[ordered_catalog] = np.arange(1, count + 1, dtype=np.uint16)
    encoded = idx[items]
    meta["domains"][name] = {"item_count": count, "windows":{}}
    for window, start, stop, folder in [
        ("train", 0, start_val, dev), ("validation", start_val, start_test, dev),
        ("test", start_test, end, sealed),
    ]:
        within = (dates >= start) & (dates < stop)
        if window == "train":
            chosen = positions[take & within & (event_start > starts)]
            assert len(chosen) == out_domains[domain]["train"]["supervised_prefixes"]
        else:
            mask = take & within & (first | np.r_[True, ~within[:-1]])
            chosen = positions[mask]
            n_known = seen_before[chosen] - seen_before[np.maximum(starts[chosen], chosen-50)]
            chosen = chosen[n_known >= 4]
            assert len(chosen) == out_domains[domain][window]["eligible"]
        seq = np.empty((len(chosen),50),dtype=np.uint16)
        for offset in range(0, len(chosen), 16384):
            batch = chosen[offset:offset+16384]
            history = event_start[batch,None] - np.arange(50,0,-1)
            present = history >= starts[batch,None]
            values = np.where(present, encoded[np.maximum(0,history)], 0)
            if window != "train":
                permutation = np.argsort(values != 0, axis=1, kind="stable")
                values = np.take_along_axis(values, permutation, axis=1)
            seq[offset:offset+len(batch)] = values
        target = idx[items[chosen]].astype(np.int32)-1
        gaps = dates[chosen]-dates[event_start[chosen]-1]
        assert np.all(gaps>0) and np.all(seq[:,-1]>0)
        data = {"seq":seq,"target":target,"gap":gaps}
        if window != "train":
            assert np.all((seq!=0).sum(axis=1)>=4)
            data.update(user=users[chosen],untied=~next_same_date[chosen],query_day=dates[chosen])
            checks = out_domains[domain][window]
            known_long=(target>=0)&(gaps>365)
            assert len(set(data["user"]))==len(chosen)
            assert int(known_long.sum())==checks["long_gap_known"]
            assert int((known_long&data["untied"]).sum())==checks["long_gap_known_untied"]
        else:
            assert np.all(target>=0)
        path=folder / f"{name}_{window}.npz"
        np.savez_compressed(path, **data)
        with path.open("rb") as stream:
            fingerprint=__import__("hashlib").file_digest(stream,"sha256").hexdigest()
        meta["domains"][name]["windows"][window]={
            "file":path.name,"rows":len(chosen),"sha256":fingerprint,"bytes":path.stat().st_size,
        }
        print(name,window,len(chosen),path.stat().st_size,flush=True)
for folder, use in ((dev, ("train","validation")), (sealed, ("test",))):
    summary={**meta, "domains":{key:{"item_count":entry["item_count"],
             "windows":{k:entry["windows"][k] for k in use}} for key,entry in meta["domains"].items()}}
    (folder/"manifest.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
