"""Read-only audit of copied Linux G2 clean calibration; never starts training."""
from pathlib import Path
import copy
import csv
import json
import math
import statistics
import sys
import types
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments import rtc_g2_clean_calibration as stage
from experiments import rtc_reference_memory_validation as validation
from experiments import rtc_r3_lower_tail_stage as lower
from analysis.rtc_reference_history_review.verify import read,sha,compare
from analysis.rtc_r1c_review.verify import direct_reference

BATCH=ROOT/'logs/rtc_g2_clean_calibration'
HERE=Path(__file__).resolve().parent


def audit():
    lock=read(BATCH/'g2c_lock.json')
    assert lock['protocol']==read(stage.PROTOCOL) and lock['accepted_parent']==read(stage.RECEIPT)
    assert len(lock['cells'])==lock['training_units']==4
    assert {(c['batch'],c['attack'],c['seed'],c['defense']) for c in lock['cells']}=={
        (n,a,s,d) for n,a,s,ds in stage.batches() for d in ds}
    for name,h in lock['artifacts'].items():assert sha((BATCH/name).read_bytes())==h,name
    with zipfile.ZipFile(BATCH/'g2c_sources.zip') as z:
        assert len(z.namelist())==len(set(z.namelist())) and set(z.namelist())==set(lock['sources'])
        sources={n:z.read(n) for n in z.namelist()}
    for name,h in lock['sources'].items():assert sha(sources[name])==h,name
    relevant=['experiments/rtc_g2_clean_calibration.py','experiments/rtc_reference_memory_validation.py',
        'experiments/rtc_r3_lower_tail_stage.py','experiments/rtc_r2_raw_norm.py',
        'experiments/rtc_r0b_observation.py','defenses/rtc/lower_tail.py','defenses/rtc/filtered_lower_tail.py',
        'defenses/rtc/raw_norm.py','defenses/rtc/spectral_direction.py','defenses/rtc/corroboration.py',
        'analysis/rtc_r3_diagnostics/replay.py']
    for name in relevant:
        assert (ROOT/name).read_bytes().replace(b'\r\n',b'\n')==sources[name].replace(b'\r\n',b'\n'),name
    bridge=(ROOT/'experiments/rtc_i12_bridge.py').read_text().replace("'--batch-size', '48'","'--batch-size', '96'")
    bridge=bridge.replace("'--ray-client-num-gpus', '.25'","'--ray-client-num-gpus', '.125'")
    assert bridge==sources['experiments/rtc_i12_bridge.py'].decode().replace('\r\n','\n')
    configs={}
    for c in lock['cells']:
        cfg=read(BATCH/c['batch']/'resolved'/(c['run_id']+'.json'))
        assert cfg['client']==stage.CLIENT_CONTRACT
        assert cfg['ray']['client_num_gpus']==.125
        configs[c['seed'],c['defense']]=cfg
    for seed in (106,107):
        cfg=copy.deepcopy(configs[seed,stage.OBSERVER]);p=cfg['security']['defense']['custom_params']
        assert p.pop('lower_tail_mode')=='observe'
        assert p.pop('lower_tail_calibration')=='config/rtc_r3_lower_tail_calibration.json'
        assert cfg==configs[seed,stage.BASE]
    mappings=set()
    def imported_read(path):
        p=Path(path);s=p.as_posix();marker='/logs/rtc_g2_clean_calibration/'
        if p.is_file() and p.resolve().is_relative_to(BATCH):return read(p)
        if marker in s:
            rel=s.split(marker,1)[1];target=(BATCH/rel).resolve()
            assert target.is_relative_to(BATCH) and rel in lock['artifacts']
            assert sha(target.read_bytes())==lock['artifacts'][rel]
            mappings.add((s,str(target)));return read(target)
        if '/config/' in s:return json.loads(sources['config/'+s.split('/config/',1)[1]])
        return read(p)
    scope=dict(validation.verify_cell.__globals__,read_json=imported_read,
        MODES={stage.BASE:('cap','cap')},RTC=(stage.BASE,))
    verify=types.FunctionType(validation.verify_cell.__code__,scope)
    oldcal=json.loads(sources['config/rtc_r3_lower_tail_calibration.json'])
    runs=[];max_error=0.;records=0
    for cell in lock['cells']:
        run=verify(BATCH,dict(cell,defense=stage.BASE),lock);run['cell']=cell
        if cell['defense']==stage.OBSERVER:lower.verify_lower(run,oldcal)
        lookup={(c['round'],c['cid']):c for c in run['clients']}
        for r in run['rounds'][1:]:
            g=np.asarray(json.loads(r['fit_rtc_r1_gram_json']));eig,vec=np.linalg.eigh(g)
            assert eig.min()>=-1e-8*max(1.,np.diag(g).max())
            x=vec*np.sqrt(np.maximum(eig,0))[None,:]
            for i,cid in enumerate(json.loads(r['fit_rtc_r1_client_ids_json'])):
                ref=direct_reference(x,i);row=lookup[r['round'],cid]
                assert (ref is not None)==stage.prior.truth(row['rtc_r1_valid'])
                if ref:
                    error=abs(ref['cosine']-float(row['rtc_r1_cosine']));assert error<1e-8
                    max_error=max(max_error,error)
                records+=1
        runs.append(run)
    result=stage.decide(runs,lock['protocol'])
    result['source_lock_sha256']=sha((BATCH/'g2c_lock.json').read_bytes())
    server=read(BATCH/'analysis/decision.json');compare(server,result,'decision')
    # Independent standard-library ratios and streaks; no production scorer.
    independent={}
    for run in runs:
        if run['cell']['defense']!=stage.OBSERVER:continue
        ratios=[];state={};flags=union=0
        for rnd in range(1,61):
            cc=[c for c in run['clients'] if int(c['round'])==rnd]
            assert not any(stage.prior.truth(c['rtc_r2_flagged']) for c in cc)
            for row in cc:
                peers=[float(c['rtc_r3l_residual_norm']) for c in cc if c['principal_id']!=row['principal_id']]
                assert len(peers)==9
                ratio=float(row['rtc_r3l_residual_norm'])/statistics.median(peers)
                assert math.isfinite(ratio) and ratio>0;ratios.append(ratio)
                pid=row['principal_id'];state[pid]=min(2,state.get(pid,0)+1) if ratio<result['threshold'] else 0
                flag=state[pid]>=2;flags+=flag
                union+=flag or float(row['rtc_r1_q'])<1 or float(row['rtc_r2_q'])<1
        independent[run['cell']['seed']]=dict(count=len(ratios),minimum=min(ratios),flags=flags,union=union)
    # Production computes the one-client principal mean as (mass*norm)/mass;
    # allow only roundoff relative to this direct norm calculation.
    assert abs(independent[106]['minimum']-result['threshold'])<=1e-12
    assert independent[107]['flags']==result['holdout_flags'] and independent[107]['union']==result['holdout_union_flags']
    result.update(source_archive_sha256=sha((BATCH/'g2c_sources.zip').read_bytes()),
        sources_verified=len(sources),artifacts_verified=len(lock['artifacts']),verified_runs=len(runs),
        runner_gates=sum(len(list(csv.DictReader((BATCH/n/'quality_gates.csv').open(encoding='utf-8-sig')))) for n,_,_,_ in stage.batches()),
        independent_reference_records=records,max_cosine_error=max_error,independent_calibration=independent,
        server_decision_matches=True,training_environment=lock['environment'],path_mappings=sorted(mappings),
        evidence={p.relative_to(BATCH).as_posix():sha(p.read_bytes()) for p in BATCH.rglob('*') if p.is_file() and p.suffix in ('.json','.csv')})
    return result


if __name__=='__main__':
    if not __debug__:raise RuntimeError('Assertions must remain enabled')
    result=audit();HERE.mkdir(exist_ok=True)
    (HERE/'review.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('quality_accepted','calibration_accepted','candidate_accepted',
        'threshold','holdout_flags','verified_runs','runner_gates','sources_verified','artifacts_verified','max_cosine_error')}))
