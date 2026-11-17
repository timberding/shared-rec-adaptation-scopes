"""Recompute validation choices and seed-paired contrasts from included records."""
import ast
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev
from scope_policy import policy

ROOT = Path(__file__).resolve().parents[1]
read = lambda name: json.loads((ROOT / name).read_text())
records = read('records/paired_records.json')
expected = read('records/expected_results.json')
SEEDS = list(map(str, records['seeds']))
checks = 0


def close(a, b):
    global checks
    if isinstance(a, dict):
        assert set(a) == set(b), (set(a), set(b))
        for k in a: close(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b): close(x, y)
    elif isinstance(a, (int, float)):
        assert math.isfinite(a) and math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12), (a, b)
    else:
        assert a == b, (a, b)
    checks += 1


def paired(values):
    assert len(values) == 5 and all(math.isfinite(v) for v in values)
    m = mean(values)
    half = 2.7764451051977987 * stdev(values) / math.sqrt(5)
    return {'mean': m, 'ci95': [m-half, m+half], 'by_seed': dict(zip(SEEDS, values))}


close(paired([0]*5)['ci95'], [0, 0])
close(paired([-2,-1,0,1,2])['mean'], 0)
output = {}
for group, row in records['groups'].items():
    target = row['recipe']['target']
    output[group] = {}
    for horizon in map(str, records['horizons']):
        chosen = policy([row['seeds'][s]['validation'][horizon] for s in SEEDS])
        close(chosen, row['expected_decisions'][horizon])
        truth = expected['groups'][group]['horizons'][horizon]
        def score(s, method):
            return 100*row['seeds'][s]['test'][horizon][method]['domains'][target]['metrics']['long_known']['ndcg10']
        def contrast(a, b): return paired([score(s,a)-score(s,b) for s in SEEDS])
        scopes = {}
        for method, ref in truth['scopes'].items():
            effect = contrast(method, 'S0')
            close(effect, ref['ndcg_minus_base_pp'])
            close(mean(score(s,method) for s in SEEDS), ref['test_ndcg_pct'])
            scopes[method] = effect
        compact = contrast(chosen['capped'], chosen['exact'])
        close(compact, truth['capped_minus_exact_pp'])
        for pair, ref in truth['replay_and_distillation'].items():
            a,b = pair.split('_minus_')
            close(contrast(a,b), ref['ndcg_pp'])
            for domain, metrics in ref['retained'].items():
                for metric, val in metrics.items():
                    scale=100 if metric in ('delta_hit','deep','top1_changed') else 1
                    differences=[scale*(row['seeds'][s]['test'][horizon][a]['retained'][domain][metric]-row['seeds'][s]['test'][horizon][b]['retained'][domain][metric]) for s in SEEDS]
                    close(paired(differences), val)
        output[group][horizon]={'choices':{k:chosen[k] for k in ('unrestricted','joint','exact','capped')},'compact_minus_exact_pp':compact,'scope_minus_base_pp':scopes,'replay_and_distillation':truth['replay_and_distillation']}
    c=output[group]['50'];e=c['compact_minus_exact_pp']
    print(f"{group}: exact={c['choices']['exact']}, 1MiB={c['choices']['capped']}; compact-exact={e['mean']:+.6f} pp [{e['ci95'][0]:+.6f}, {e['ci95'][1]:+.6f}]")
# The secondary L2 calibration is separate from the original seven-scope policy.
l2=read('records/l2_analysis.json')
for phase in ('validation','test'):
    rows=[read(f'records/precision/l2/1e-2/runs/sasrec-video_games/{s}/S7/{phase}.json') for s in SEEDS]
    def score_l2(r): return r['selected']['val']['long_known']['ndcg10'] if phase=='validation' else r['domains']['video_games']['scores']['long_known']['ndcg10']
    for method, ref in l2['phases'][phase]['target_contrasts'].items():
        def before(s):
            r=records['groups']['sasrec-video_games']['seeds'][s][phase]['50'][method]
            return r['selection_ndcg'] if phase=='validation' else r['domains']['video_games']['metrics']['long_known']['ndcg10']
        values=[100*(score_l2(r)-before(s)) for s,r in zip(SEEDS,rows)]
        got=paired(values)
        close(got['mean'],ref['mean']);close(got['ci95'],[ref['lo'],ref['hi']])
for entry in read('source_manifest.json'):
    p=ROOT/entry['path']
    assert hashlib.sha256(p.read_bytes()).hexdigest()==entry['package_sha256'], entry['path']
    if p.suffix=='.py': ast.parse(p.read_text(), filename=entry['path'])
print(f'PASS: {checks} numeric/structural comparisons; 7 settings, 5 paired seeds, 3 horizons. Training and request-level rescoring not run.')
# Writing is opt-in to keep an extracted reviewer package immutable by default.
if '--write' in __import__('sys').argv:
    Path('recomputed_results.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
