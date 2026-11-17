#!/usr/bin/env python3
"""Summarize the five-seed test and pair the observed user outcomes."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np

SEEDS = (9308,42096,169953,231339,799822)
METHODS = ("S0","S4a","S5","S6","S8","history_only","gap_only","continued","time_start","time_start_s5","meantime")
PAIRS = (("S5","S4a"),("S5","S6"),("S5","S8"),("S5","history_only"),("S5","gap_only"),
         ("S8","continued"),("time_start_s5","time_start"),("S5","meantime"))


def interval(values):
    center = statistics.mean(values)
    se = statistics.stdev(values)/len(values)**.5
    return {"mean":center,"t95":[center-2.776445105*se,center+2.776445105*se],"seed_values":list(values)}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def outcomes(rank):
    hit = (rank>0)&(rank<=10)
    return {"hit10":hit.astype(float),"ndcg10":hit/np.log2(np.maximum(rank,1)+1)}


def bootstrap(v, draws=10000):
    rng = np.random.default_rng(20261111)
    means = []
    for start in range(0,draws,128):
        n = min(128,draws-start)
        si = rng.integers(v.shape[0],size=(n,v.shape[0],1))
        ui = rng.integers(v.shape[1],size=(n,1,v.shape[1]))
        means.extend(v[si,ui].mean(axis=(1,2)))
    return np.quantile(means,[.025,.975]).tolist()


def subgroup_report(runs,predictions):
    data = runs.parent/"data"
    m = json.loads((data/"development/manifest.json").read_text())
    with np.load(data/"development/video_games_train.npz",allow_pickle=False) as a:
        popularity = np.bincount(a["target"],minlength=m["domains"]["video_games"]["item_count"])
    with np.load(data/"development/video_games_validation.npz",allow_pickle=False) as a:
        validation = {k:a[k] for k in a.files}
    with np.load(data/"test_sealed/video_games_test.npz",allow_pickle=False) as a:
        test = {k:a[k] for k in a.files}
    def features(a):
        mask = a["seq"] > 0
        length = mask.sum(1)
        prior = a["query_day"] - a["gap"]
        activity = ((a["seq_day"] >= prior[:,None]-180)&mask).sum(1)
        frequencies = np.r_[0,popularity][a["seq"]]
        pop = (np.log1p(frequencies)*mask).sum(1)/length
        return np.column_stack((np.log1p(length),np.log1p(activity),pop))
    xv,xt = features(validation),features(test)
    center,scale = xv.mean(0),np.maximum(xv.std(0),1e-9)
    xv,xt = (xv-center)/scale,(xt-center)/scale
    xv,xt = np.c_[np.ones(len(xv)),xv],np.c_[np.ones(len(xt)),xt]
    yv,yt = validation["gap"]>365,test["gap"]>365
    def prob(x,coef):
        return 1/(1+np.exp(-np.clip(x@coef,-30,30)))
    coef = np.zeros(xv.shape[1])
    penalty = np.diag([1e-8,1,1,1])
    for _ in range(30):
        p = prob(xv,coef)
        step = np.linalg.solve(xv.T@((p*(1-p))[:,None]*xv)+penalty,xv.T@(yv-p)-penalty@coef)
        coef += step
        if np.linalg.norm(step) < 1e-8:
            break
    pv,pt = prob(xv,coef),prob(xt,coef)
    lo,hi = max(pv[yv].min(),pv[~yv].min()),min(pv[yv].max(),pv[~yv].max())
    support = (pt>=lo)&(pt<=hi)
    edges = np.unique(np.quantile(pv[(pv>=lo)&(pv<=hi)],np.linspace(0,1,6)))
    common = []
    paired = np.asarray([outcomes(a["rank"])["ndcg10"]-outcomes(b["rank"])["ndcg10"]
                         for a,b in zip(predictions["S5"],predictions["history_only"])])
    delta = paired.mean(axis=0)
    assert np.array_equal(predictions["S5"][0]["user"],test["user"])
    for i in range(len(edges)-1):
        mask = support&(pt>=edges[i])&((pt<edges[i+1]) if i<len(edges)-2 else (pt<=edges[i+1]))
        a,b = mask&yt,mask&~yt
        common.append({"n_long":int(a.sum()),"n_short":int(b.sum()),
                       "pre_gap_feature_mean_diff_validation_sd":(xt[a,1:].mean(0)-xt[b,1:].mean(0)).tolist() if a.any() and b.any() else None,
                       "S5_minus_history_only_ndcg_long":float(delta[a].mean()) if a.any() else None,
                       "S5_minus_history_only_ndcg_short":float(delta[b].mean()) if b.any() else None})
    gaps = []
    for lower,upper in ((0,30),(30,90),(90,180),(180,365),(365,730),(730,np.inf)):
        mask = (test["gap"]>lower)&(test["gap"]<=upper)&(test["target"]>=0)
        gaps.append({"lower":lower,"upper":None if upper==np.inf else upper,"n":int(mask.sum()),
                     "S5_minus_history_only_ndcg":interval(paired[:,mask].mean(axis=1).tolist()) if mask.any() else None})
    return {"gap_strata_known":gaps,"propensity_validation_fit":{"features":["encodable_length_capped_50","prior_180d_activity_capped_50","mean_log_train_item_frequency"],"support": [float(lo),float(hi)],"test_excluded":int((~support).sum()),"strata":common,"scope":"Observational comparison. OOV targets count as failures in propensity strata."}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs",type=Path)
    p.add_argument("--out",type=Path,required=True)
    args = p.parse_args()
    freeze_file = args.runs/"freeze.json"
    freeze = json.loads(freeze_file.read_text())
    freeze_hash = digest(freeze_file)
    for seed in SEEDS:
        done = json.loads((args.runs/str(seed)/"TEST_COMPLETE.json").read_text())
        assert done["seed"]==seed and done["freeze_sha"]==freeze_hash
    for name,h in freeze["cost_sha256"].items():
        assert digest(args.runs/"cost"/name)==h
    assert len(freeze["cost_sha256"])==12
    scores, predictions, retained, checks = {}, {}, {}, {}
    for method in METHODS:
        rows = []
        predictions[method] = []
        retained[method] = []
        checks[method] = []
        for seed in SEEDS:
            dir = args.runs/str(seed)
            r = json.loads((dir/f"test_{method}.json").read_text())
            assert r["checkpoint_sha"]==freeze["checkpoints"][str(seed)][method]
            assert r["test_data_sha"]==freeze["test_manifest_sha"]
            rows.append(r["metrics"]["video_games"])
            retained[method].append({d:m["all"] for d,m in r["metrics"].items() if d!="video_games"})
            checks[method].append(r["retained_identity"])
            with np.load(dir/f"test_{method}_video_games.npz",allow_pickle=False) as a:
                predictions[method].append({k:a[k] for k in a.files})
        for a in predictions[method]:
            for key in ("user","gap","known","untied"):
                assert np.array_equal(a[key],predictions["S0"][0][key])
        scores[method] = {scope:{metric:interval([r[scope][metric] for r in rows])
                         for metric in ("hit10","ndcg10")} for scope in rows[0]}
    test_pair = {}
    for a,b in PAIRS:
        matrix = {"hit10":[],"ndcg10":[]}
        for aa,bb in zip(predictions[a],predictions[b]):
            for key in ("user","gap","known","untied"):
                assert np.array_equal(aa[key],bb[key])
            mask = (aa["gap"] > 365)&aa["known"]
            ma,mb = outcomes(aa["rank"][mask]),outcomes(bb["rank"][mask])
            for metric in matrix:
                matrix[metric].append(100*(ma[metric]-mb[metric]))
        test_pair[f"{a}-{b}"] = {
            m:{**interval(np.asarray(v).mean(axis=1).tolist()),"paired_user_seed_bootstrap95":bootstrap(np.asarray(v))}
            for m,v in matrix.items()}
        for m, result in test_pair[f"{a}-{b}"].items():
            half = 5.261057576*statistics.stdev(result["seed_values"])/len(SEEDS)**.5
            result["bonferroni95_over_8_contrasts_per_metric"] = [result["mean"]-half,result["mean"]+half]
    signed = {a:{domain:{metric:interval([100*(u[domain][metric]-v[domain][metric])
                  for u,v in zip(retained[a],retained["S0"])]) for metric in ("hit10","ndcg10")}
                  for domain in retained[a][0]} for a in METHODS if a!="meantime"}
    movement = {a:[100*max(abs(u[d]["hit10"]-v[d]["hit10"]) for d in u)
                   for u,v in zip(retained[a],retained["S0"])] for a in METHODS if a!="meantime"}
    cost = [json.loads((args.runs/"cost"/name).read_text()) for name in sorted(freeze["cost_sha256"])]
    decisions = {}
    for name,row in freeze["validation_decisions"].items():
        surface = row["selected"]
        method = "S8" if surface=="S8_copy" else surface
        decisions[name] = {**row,"test_target":scores[method],
                           "deployment_update_bytes":freeze["validated_storage_bytes"][surface],
                           "retained_service":"S0" if surface in ("S8_copy","meantime") else method}
    args.out.write_text(json.dumps({"status":"five_seed_test_complete",
                    "target_metrics":scores,"long_known_paired_pp":test_pair,
                    "retained_signed_pp":signed,"retained_full_scores_identity":checks,
                    "retained_max_hit10_movement_pp":movement,"cost_profiles":cost,
                    "validation_locked_decisions":decisions,
                    "cohort_diagnostics":subgroup_report(args.runs,predictions)},indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    main()
