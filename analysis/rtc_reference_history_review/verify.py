"""Offline audit of the copied, frozen reference-history observation batch; never trains."""
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics
import sys
import types
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments import rtc_reference_history_observation as stage
from experiments import rtc_i12_validation as validation
from analysis.rtc_r1c_review.verify import direct_reference

BATCH = ROOT / 'logs/rtc_reference_history_observation'
HERE = Path(__file__).resolve().parent


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(b):
    return hashlib.sha256(b).hexdigest()


def compare(a,b,path):
    if isinstance(a,dict):
        assert isinstance(b,dict) and set(a)==set(b),path
        for k in a:compare(a[k],b[k],path+'/'+str(k))
    elif isinstance(a,list):
        assert isinstance(b,list) and len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+'/'+str(i))
    elif type(a) is float and type(b) is float:
        assert math.isfinite(a) and math.isfinite(b) and abs(a-b)<=1e-12*max(1,abs(a),abs(b)),(path,a,b)
    else:assert a==b,(path,a,b)


def audit():
    lock = read(BATCH/'r1h_lock.json')
    assert lock['protocol'] == read(stage.PROTOCOL)
    assert lock['accepted_parent'] == read(stage.RECEIPT)
    assert len(lock['cells']) == lock['training_units'] == 5
    assert {(c['batch'], c['attack'], c['seed'], c['defense']) for c in lock['cells']} == {
        (n,a,s,d) for n,a,s,ds in stage.batches() for d in ds}
    for name, expected in lock['artifacts'].items():
        assert sha((BATCH/name).read_bytes()) == expected, name
    with zipfile.ZipFile(BATCH/'r1h_sources.zip') as archive:
        assert len(archive.namelist()) == len(set(archive.namelist()))
        assert set(archive.namelist()) == set(lock['sources'])
        sources = {n: archive.read(n) for n in archive.namelist()}
    for name, expected in lock['sources'].items():
        assert sha(sources[name]) == expected, name
    # Verify the local executable audit dependencies against original server bytes.
    relevant = ['defenses/rtc/spectral_direction.py', 'defenses/rtc/corroboration.py',
        'defenses/rtc/raw_norm.py',
        'experiments/rtc_reference_history_observation.py', 'experiments/rtc_i12_validation.py',
        'experiments/rtc_i12_bridge.py', 'experiments/rtc_r2_raw_norm.py',
        'experiments/rtc_r0b_observation.py', 'analysis/rtc_r3_diagnostics/replay.py',
        'config/rtc_reference_history_observation.json']
    for name in relevant:
        assert (ROOT/name).read_bytes().replace(b'\r\n',b'\n') == sources[name].replace(b'\r\n',b'\n'), name
    mappings=set()
    def imported_read(path):
        p=Path(path);s=p.as_posix()
        if p.is_file() and p.resolve().is_relative_to(BATCH):
            return read(p)
        marker='/logs/rtc_reference_history_observation/'
        if marker in s:
            rel=s.split(marker,1)[1];target=(BATCH/rel).resolve()
            assert target.is_relative_to(BATCH) and rel in lock['artifacts']
            assert sha(target.read_bytes()) == lock['artifacts'][rel]
            mappings.add((s,str(target)))
            return read(target)
        if '/config/' in s:
            return json.loads(sources['config/'+s.split('/config/',1)[1]])
        return read(p)
    scope=dict(validation.verify_cell.__globals__, read_json=imported_read,
        MODES=stage.MODES, RTC=(stage.PARENT,stage.OBSERVER))
    verify=types.FunctionType(validation.verify_cell.__code__,scope)
    runs=[verify(BATCH,c,lock) for c in lock['cells']]
    result=stage.decide(runs)
    records=0;max_error=0.
    for run in runs:
        lookup={(c['round'],c['cid']):c for c in run['clients']}
        for r in run['rounds'][1:]:
            g=np.asarray(json.loads(r['fit_rtc_r1_gram_json']));eig,vec=np.linalg.eigh(g)
            assert eig.min()>=-1e-8*max(1.,np.diag(g).max())
            x=vec*np.sqrt(np.maximum(eig,0))[None,:];ids=json.loads(r['fit_rtc_r1_client_ids_json'])
            for i,cid in enumerate(ids):
                c=lookup[r['round'],cid];ref=direct_reference(x,i)
                assert (ref is not None)==stage.prior.truth(c['rtc_r1_valid'])
                if ref:
                    error=abs(ref['cosine']-float(c['rtc_r1_cosine']));assert error<1e-8
                    max_error=max(max_error,error)
                records+=1
    server=read(BATCH/'analysis/observation.json')
    for k in ('observation_quality_accepted','candidate_accepted','observations','summaries'):
        compare(server[k],result[k],k)
    distributions=[];flagged=[]
    for observation in result['observations']:
        for bad in (False,True):
            rows=[r for r in observation['records'] if r['attack_active']==bad]
            valid=[r for r in rows if r['valid']]
            flags=[r for r in rows if r['r1_flagged']]
            values=[r['reference_previous_cosine'] for r in valid]
            distributions.append(dict(batch=observation['batch'],attack_active=bad,records=len(rows),valid=len(valid),
                minimum=min(values) if values else None,median=statistics.median(values) if values else None,
                maximum=max(values) if values else None,flagged=len(flags),
                flagged_reference_positive=sum(r['valid'] and r['reference_previous_cosine']>0 for r in flags),
                flagged_reference_nonpositive=sum(r['valid'] and r['reference_previous_cosine']<=0 for r in flags)))
            flagged.extend(dict(batch=observation['batch'],**r) for r in flags)
    result.update(source_lock_sha256=sha((BATCH/'r1h_lock.json').read_bytes()),
        source_archive_sha256=sha((BATCH/'r1h_sources.zip').read_bytes()),
        sources_verified=len(sources),artifacts_verified=len(lock['artifacts']),verified_runs=len(runs),
        runner_gates=sum(len(list(csv.DictReader((BATCH/n/'quality_gates.csv').open(encoding='utf-8-sig')))) for n,_,_,_ in stage.batches()),
        independent_reference_records=records,max_cosine_error=max_error,server_decision_matches=True,server_comparison_float_tolerance=1e-12,
        distributions=distributions,flagged_history_rows=flagged,path_mappings=sorted(mappings),training_environment=lock['environment'],
        evidence={p.relative_to(BATCH).as_posix():sha(p.read_bytes()) for p in BATCH.rglob('*')
            if p.is_file() and p.suffix in ('.json','.csv')})
    return result


if __name__=='__main__':
    if not __debug__:raise RuntimeError('Assertions must remain enabled')
    result=audit();HERE.mkdir(exist_ok=True)
    (HERE/'review.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('observation_quality_accepted','candidate_accepted','sources_verified',
        'artifacts_verified','verified_runs','runner_gates','independent_reference_records','max_cosine_error','distributions')},indent=2))
