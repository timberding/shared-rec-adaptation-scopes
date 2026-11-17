"""Paired precision effects and fixed-output KL coverage, without refitting."""
import ast
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
OLD = HERE.parents[1] / 'paper-tois-revision/qa/20261116_ipm_topk_theory/novelty/probe_fixed_outputs.py'
tree = ast.parse(OLD.read_text())
function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'radius_bounds')
exec(compile(ast.Module(body=[function], type_ignores=[]), str(OLD), 'exec'))
SEEDS = (9308, 42096, 169953, 231339, 799822)
METHODS = ('S0', 'S8', 'Replay', 'Distill_1', 'Distill_4')
DOMAINS = ('cds_vinyl', 'movies_tv')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def interval(values):
    x = np.asarray(values, dtype=float)
    assert len(x) == 5
    mean = x.mean(); radius = 2.7764451051977987*x.std(ddof=1)/np.sqrt(len(x))
    return dict(mean=float(mean), lo=float(mean-radius), hi=float(mean+radius))


def main():
    receipts, arrays, inputs = {}, {}, {}
    for seed in SEEDS:
        for phase in ('validation','test'):
            for precision in ('bf16','fp32'):
                p = HERE/'results'/str(seed)/phase/(precision+'.json')
                r = json.loads(p.read_text()); a = p.with_suffix('.npz')
                assert sha(a) == r['arrays_sha'] and r['binding']['seed'] == seed
                assert r['binding']['source'] == sha(HERE/'precision.py')
                receipts[seed,phase,precision] = r; inputs[str(p.relative_to(HERE))] = sha(p)
                arrays[seed,phase,precision] = np.load(a, allow_pickle=False)
                if precision == 'bf16':
                    assert all(v['original_rank_mismatch'] == v['original_top10_mismatch'] == 0
                               for d in r['metrics'].values() for v in d.values())
    out = dict(inputs=inputs, radius_source_sha=sha(OLD), estimates={}, contrasts={}, coverage={})
    for phase in ('validation','test'):
        for precision in ('bf16','fp32'):
            for domain in DOMAINS:
                group = f'{phase}/{precision}/{domain}'
                vals = [receipts[s,phase,precision]['metrics'][domain] for s in SEEDS]
                out['estimates'][group] = {m: {k: interval([r[m][k] for r in vals]) for k in
                    ('hit','delta_hit','ndcg','deep','replaced','rbo','top1','boundary_tie_rate',
                     'deep_both_miss_all','deep_given_both_miss')} for m in METHODS}
                out['contrasts'][group] = {f'{a}-{b}': {k: interval([r[a][k]-r[b][k] for r in vals])
                    for k in ('hit','deep','rbo','top1')} for a,b in
                    (('Replay','S8'),('Distill_1','Replay'),('Distill_4','Replay'))}
                if phase != 'validation': continue
                cover = {m: {str(d): dict(n=0, passed=0, observed=0, pinsker=0, tied=0) for d in (1,5,10)}
                         for m in METHODS[1:]}
                for seed in SEEDS:
                    z = arrays[seed,phase,precision]; top = z[f'{domain}/S0/top20']
                    strict = ~z[f'{domain}/S0/ties']
                    for depth in (1,5,10):
                        low, high, pin = radius_bounds(top,10,depth)
                        assert np.all(low <= high+1e-15)
                        for m in METHODS[1:]:
                            kl, replaced = z[f'{domain}/{m}/kl'], z[f'{domain}/{m}/replaced']
                            passed = strict & (kl+1e-10 < low-1e-10)
                            assert np.all(replaced[passed] < depth)
                            row=cover[m][str(depth)]; row['n']+=len(kl); row['passed']+=int(passed.sum())
                            row['observed']+=int((replaced<depth).sum());row['tied']+=int((~strict).sum())
                            row['pinsker']+=int((strict & (kl+1e-10<pin-1e-10)).sum())
                out['coverage'][group]=cover
    out['total_scoring_seconds']=sum(r['seconds'] for r in receipts.values())
    out['max_peak_bytes']=max(r['peak_allocated'] for r in receipts.values())
    (HERE/'PRECISION_ANALYSIS.json').write_text(json.dumps(out,indent=2)+'\n')
    for domain in DOMAINS:
        for precision in ('bf16','fp32'):
            print(domain,precision,'test', {m:round(out['estimates'][f'test/{precision}/{domain}'][m]['deep']['mean']*100,4) for m in METHODS[1:]})
            print('KL',out['coverage'][f'validation/{precision}/{domain}']['Distill_1'])


if __name__=='__main__':
    main()
