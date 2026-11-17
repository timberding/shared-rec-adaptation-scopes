"""Evaluate the locked lambda without replacing the original seven-scope study."""
import json
from pathlib import Path
from analyze import interval, sha, SEEDS, HERE

read = lambda p: json.loads(p.read_text())
lock = read(HERE/'L2_SELECTION.json')
assert lock['plan_sha'] == sha(HERE/'PLAN.json') and lock['selector_sha'] == sha(HERE/'select_l2.py')
assert lock['selected_lambda'] == '1e-2'
inputs = {}
for remote, digest in lock['inputs'].items():
    if '/deltagate_ipm_precision_closure_20261116/' in remote:
        path = HERE/remote.split('/deltagate_ipm_precision_closure_20261116/')[1]
    elif remote.endswith('/validation/COMPLETE.json'):
        path = HERE/'original/9308/validation.json'
    else:
        path = HERE/'original/9308/FIT.json'
    assert sha(path) == digest

out = {'selected_lambda':lock['selected_lambda'],'calibration':lock['validation'],
       'calibration_eligible':lock['eligible'],'phases':{},'inputs':inputs}
fits=[]
for phase in ('validation','test'):
    new, old = [], []
    for seed in SEEDS:
        folder=HERE/f'l2/1e-2/runs/sasrec-video_games/{seed}/S7'
        path=folder/(phase+'.json');r=read(path);f=read(folder/'FIT.json')
        assert r['fit_sha']==sha(folder/'FIT.json') and r['plan_sha']==sha(HERE/'PLAN.json')
        assert f['binding']['plan']==sha(HERE/'PLAN.json') and f['epochs']==50
        assert r['data_sha']==read(HERE/f'original/{seed}/{phase}.json')['data_sha']
        inputs[str(path.relative_to(HERE))]=sha(path)
        new.append(r);old.append(read(HERE/f'original/{seed}/{phase}.json')['horizons']['50'])
        if phase=='test':fits.append(f)
    def target(r):
        return r['selected']['val']['long_known']['ndcg10'] if phase=='validation' else r['domains']['video_games']['scores']['long_known']['ndcg10']
    def original(r,m):
        return r[m]['selection_ndcg'] if phase=='validation' else r[m]['domains']['video_games']['metrics']['long_known']['ndcg10']
    record={'target_ndcg':interval([target(r) for r in new]),
            'target_contrasts':{m:interval([(target(r)-original(o,m))*100 for r,o in zip(new,old)])
                                for m in ('S0','S4a','S6','S7','S8','Replay')}, 'retained':{}}
    for d in ('cds_vinyl','movies_tv'):
        record['retained'][d]={k:interval([r['domains'][d]['retained'][k] for r in new])
                              for k in ('delta_hit','deep','mean_replaced','rbo')}
        record['retained'][d]['deep_minus_original_l2_pp']=interval([
            (r['domains'][d]['retained']['deep']-o['S7']['retained'][d]['deep'])*100 for r,o in zip(new,old)])
    record['joint_eligible']=all(v['delta_hit']['mean']>=-.001 and v['deep']['mean']<=.25 for v in record['retained'].values())
    out['phases'][phase]=record
out['selected_epochs']=[f['selected']['50']['epoch'] for f in fits]
all_fits=[read(p) for p in (HERE/'l2').glob('*/runs/*/*/S7/FIT.json')]
out['new_fits']=len(all_fits)
out['train_seconds']=sum(f['train_seconds'] for f in all_fits)
out['validation_seconds']=sum(f['validation_seconds'] for f in all_fits)
(HERE/'L2_ANALYSIS.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ('inputs','calibration')},indent=2))
