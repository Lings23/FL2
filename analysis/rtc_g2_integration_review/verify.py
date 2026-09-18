"""Read-only audit of copied Linux G2 integration; never starts training."""
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
from experiments import rtc_g2_integration as stage
from experiments import rtc_reference_memory_validation as validation
from experiments.rtc_g2_validation import verify_lower
from analysis.rtc_reference_history_review.verify import read,sha,compare
from analysis.rtc_r1c_review.verify import direct_reference

BATCH=ROOT/'logs/rtc_g2_integration'
HERE=Path(__file__).resolve().parent


def audit():
    lock=read(BATCH/'g2_lock.json')
    assert lock['protocol']==read(stage.PROTOCOL) and lock['accepted_parent']==read(stage.RECEIPT)
    assert len(lock['cells'])==lock['training_units']==32
    assert {(c['batch'],c['attack'],c['seed'],c['defense']) for c in lock['cells']}=={
        (n,a,s,d) for n,a,s,ds in stage.batches() for d in ds}
    for name,h in lock['artifacts'].items():assert sha((BATCH/name).read_bytes())==h,name
    with zipfile.ZipFile(BATCH/'g2_sources.zip') as z:
        assert len(z.namelist())==len(set(z.namelist())) and set(z.namelist())==set(lock['sources'])
        sources={n:z.read(n) for n in z.namelist()}
    for name,h in lock['sources'].items():assert sha(sources[name])==h,name
    relevant=['experiments/rtc_g2_integration.py','experiments/rtc_reference_memory_validation.py',
        'experiments/rtc_g2_validation.py','experiments/rtc_r2_raw_norm.py',
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
    for _,attack,seed,_ in stage.batches():
        # Config lookup includes condition; pairwise equality is checked below.
        cells=[c for c in lock['cells'] if c['seed']==seed and c['attack']==attack]
        cfgs={c['defense']:read(BATCH/c['batch']/'resolved'/(c['run_id']+'.json')) for c in cells}
        candidate=copy.deepcopy(cfgs[stage.OBSERVER])
        candidate['security']['defense']['custom_params']['lower_tail_mode']='observe'
        assert candidate==cfgs[stage.BASE]
        candidate=copy.deepcopy(cfgs[stage.OBSERVER])
        candidate['security']['defense']['custom_params']['lower_tail_reference_policy']='all'
        assert candidate==cfgs[stage.PARENT]
    mappings=set()
    def imported_read(path):
        p=Path(path);s=p.as_posix();marker='/logs/rtc_g2_integration/'
        if p.is_file() and p.resolve().is_relative_to(BATCH):return read(p)
        if marker in s:
            rel=s.split(marker,1)[1];target=(BATCH/rel).resolve()
            assert target.is_relative_to(BATCH) and rel in lock['artifacts']
            assert sha(target.read_bytes())==lock['artifacts'][rel]
            mappings.add((s,str(target)));return read(target)
        if '/config/' in s:return json.loads(sources['config/'+s.split('/config/',1)[1]])
        return read(p)
    runs=[];max_error=0.;records=0
    newcal=json.loads(sources['config/rtc_g2_lower_tail_calibration.json'])
    for cell in lock['cells']:
        rtc=cell['defense'] in (stage.BASE,stage.PARENT,stage.OBSERVER)
        alias='rtc_i12_eligibility_confirmed' if rtc else cell['defense']
        scope=dict(validation.verify_cell.__globals__,read_json=imported_read,
            MODES={alias:stage.MODES[cell['defense']]},RTC=('rtc_i12_eligibility_confirmed',))
        verify=types.FunctionType(validation.verify_cell.__code__,scope)
        run=verify(BATCH,dict(cell,defense=alias),lock);run['cell']=cell
        if rtc:
            verify_lower(run,newcal,'observe' if cell['defense']==stage.BASE else 'cap',
                'all' if cell['defense']==stage.PARENT else 'raw_eligible')
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
    result['source_lock_sha256']=sha((BATCH/'g2_lock.json').read_bytes())
    server=read(BATCH/'analysis/decision.json')
    for k,v in result.items():compare(server[k],v,k)
    for name,h in server['evidence'].items():assert sha((BATCH/name).read_bytes())==h,name
    result.update(source_archive_sha256=sha((BATCH/'g2_sources.zip').read_bytes()),
        sources_verified=len(sources),artifacts_verified=len(lock['artifacts']),verified_runs=len(runs),
        runner_gates=sum(len(list(csv.DictReader((BATCH/n/'quality_gates.csv').open(encoding='utf-8-sig')))) for n,_,_,_ in stage.batches()),
        independent_reference_records=records,max_cosine_error=max_error,
        server_decision_matches=True,training_environment=lock['environment'],path_mappings=sorted(mappings),
        evidence={p.relative_to(BATCH).as_posix():sha(p.read_bytes()) for p in BATCH.rglob('*') if p.is_file() and p.suffix in ('.json','.csv')})
    return result


if __name__=='__main__':
    if not __debug__:raise RuntimeError('Assertions must remain enabled')
    result=audit();HERE.mkdir(exist_ok=True)
    (HERE/'review.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('quality_accepted','candidate_accepted','verified_runs','runner_gates','sources_verified','artifacts_verified','max_cosine_error')}))
