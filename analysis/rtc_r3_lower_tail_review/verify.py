"""Offline audit of the copied, frozen R3 lower-tail batch; never trains."""
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
from experiments import rtc_r3_lower_tail_stage as stage
from experiments import rtc_i12_validation as validation
from analysis.rtc_r1c_review.verify import direct_reference

BATCH = ROOT / 'logs/rtc_r3_lower_tail'
HERE = Path(__file__).resolve().parent


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(b):
    return hashlib.sha256(b).hexdigest()


def audit():
    lock = read(BATCH/'r3l_lock.json')
    assert lock['protocol'] == read(stage.PROTOCOL)
    assert lock['accepted_parent'] == read(stage.RECEIPT)
    assert len(lock['cells']) == lock['training_units'] == 24
    assert {(c['batch'], c['attack'], c['seed'], c['defense']) for c in lock['cells']} == {
        (n,a,s,d) for n,a,s,ds in stage.batches() for d in ds}
    for name, expected in lock['artifacts'].items():
        assert sha((BATCH/name).read_bytes()) == expected, name
    with zipfile.ZipFile(BATCH/'r3l_sources.zip') as archive:
        assert len(archive.namelist()) == len(set(archive.namelist()))
        assert set(archive.namelist()) == set(lock['sources'])
        sources = {n: archive.read(n) for n in archive.namelist()}
    for name, expected in lock['sources'].items():
        assert sha(sources[name]) == expected, name
    # Verify the local executable audit dependencies against original server bytes.
    relevant = ['defenses/rtc/spectral_direction.py', 'defenses/rtc/corroboration.py',
        'defenses/rtc/raw_norm.py', 'defenses/rtc/lower_tail.py',
        'experiments/rtc_r3_lower_tail_stage.py', 'experiments/rtc_i12_validation.py',
        'experiments/rtc_i12_bridge.py', 'experiments/rtc_r2_raw_norm.py',
        'experiments/rtc_r0b_observation.py', 'analysis/rtc_r3_diagnostics/replay.py',
        'config/rtc_r3_lower_tail_calibration.json', 'config/rtc_r3_lower_tail_protocol.json']
    for name in relevant:
        assert (ROOT/name).read_bytes().replace(b'\r\n',b'\n') == sources[name].replace(b'\r\n',b'\n'), name
    mappings=set()
    def imported_read(path):
        p=Path(path);s=p.as_posix()
        if p.is_file() and p.resolve().is_relative_to(BATCH):
            return read(p)
        marker='/logs/rtc_r3_lower_tail/'
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
    calibration=json.loads(sources['config/rtc_r3_lower_tail_calibration.json'])
    assert stage.lower_tail.calibration_hash(calibration)==lock['protocol']['lower_calibration_hash']
    for run in runs:
        if run['cell']['defense'] in (stage.PARENT,stage.OBSERVER):
            stage.verify_lower(run,calibration)
    result=stage.decide(runs,lock['protocol'])
    diagnostics=[];independent=[];max_cosine_error=0.;records=0
    for run,summary in zip(runs,result['summaries']):
        cell=run['cell'];start=1 if cell['attack']=='none' else 11
        rr=[r for r in run['rounds'] if int(r['round'])>=start]
        cc=[c for c in run['clients'] if int(c['round'])>=start]
        good=[c for c in cc if not stage.prior.truth(c['attack_active'])]
        bad=[c for c in cc if stage.prior.truth(c['attack_active'])]
        rtc=cell['defense'] in (stage.PARENT,stage.OBSERVER)
        union=lambda c:rtc and (float(c['rtc_r1_q'])<1 or float(c['rtc_r2_q'])<1 or float(c['rtc_r3l_q'])<1)
        values=dict(active_accuracy=statistics.mean(float(r['server_accuracy']) for r in rr),
            final_accuracy=float(run['rounds'][-1]['server_accuracy']),
            malicious_weight_per_round=sum(float(c['aggregation_weight']) for c in bad)/len(rr) if bad else None,
            benign_union_flag_rate=sum(union(c) for c in good)/len(good))
        for k,v in values.items():
            assert (v is None and summary[k] is None) or (v is not None and abs(v-summary[k])<1e-12),(cell,k)
        independent.append(dict(cell,**values,benign_records=len(good),malicious_records=len(bad),
            benign_union_flags=sum(union(c) for c in good)))
        lookup={(c['round'],c['cid']):c for c in run['clients']}
        for r in run['rounds'][1:]:
            gram=np.asarray(json.loads(r['fit_rtc_r1_gram_json']))
            eig,vec=np.linalg.eigh(gram)
            assert eig.min()>=-1e-8*max(1.,np.diag(gram).max())
            x=vec*np.sqrt(np.maximum(eig,0))[None,:]
            ids=json.loads(r['fit_rtc_r1_client_ids_json'])
            cohort=[lookup[r['round'],cid] for cid in ids]
            for i,c in enumerate(cohort):
                ref=direct_reference(x,i)
                assert (ref is not None)==stage.prior.truth(c['rtc_r1_valid'])
                if ref:
                    error=abs(ref['cosine']-float(c['rtc_r1_cosine']))
                    max_cosine_error=max(max_cosine_error,error);assert error<1e-8
                    votes=sum(float(x[i]@x[j]/np.linalg.norm(x[i])/np.linalg.norm(x[j]))
                        <lock['calibration']['corroboration']['pairwise_threshold'] for j in ref['indices'])
                    assert votes==int(c['rtc_r1_corroboration_votes'])
                    assert math.ceil(2*len(ref['indices'])/3)==int(c['rtc_r1_corroboration_required'])
                records+=1
                if rtc and not stage.prior.truth(c['attack_active']) and union(c):
                    diagnostics.append(dict(cell,**c,
                        active_attackers=sum(stage.prior.truth(z['attack_active']) for z in cohort),
                        reference_attackers=sum(stage.prior.truth(lookup[r['round'],cid]['attack_active'])
                            for cid in json.loads(c['rtc_r1_reference_ids_json'])),
                        cohort_raw_norms=[float(z['rtc_r2_raw_norm']) for z in cohort],
                        cohort_lower_norms=[float(z['rtc_r3l_residual_norm']) for z in cohort]))
    server=read(BATCH/'analysis/decision.json')
    for key in ('candidate_accepted','quality_accepted','checks','summaries','effects'):
        assert server[key]==result[key],key
    result.update(source_lock_sha256=sha((BATCH/'r3l_lock.json').read_bytes()),
        source_archive_sha256=sha((BATCH/'r3l_sources.zip').read_bytes()),
        sources_verified=len(sources),artifacts_verified=len(lock['artifacts']),verified_runs=len(runs),
        runner_gates=sum(len(list(csv.DictReader((BATCH/n/'quality_gates.csv').open(encoding='utf-8-sig')))) for n,_,_,_ in stage.batches()),
        independent_reference_records=records,max_cosine_error=max_cosine_error,
        independently_recomputed_metrics=independent,server_decision_identical=True,
        failed_checks=[k for k,v in result['checks'].items() if not v],
        benign_flag_diagnostics=diagnostics,path_mappings=sorted(mappings),training_environment=lock['environment'],
        evidence={p.relative_to(BATCH).as_posix():sha(p.read_bytes()) for p in BATCH.rglob('*')
            if p.is_file() and p.suffix in ('.json','.csv')})
    return result


if __name__=='__main__':
    if not __debug__:raise RuntimeError('Assertions must remain enabled')
    result=audit();HERE.mkdir(exist_ok=True)
    (HERE/'review.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('quality_accepted','candidate_accepted','sources_verified',
        'artifacts_verified','verified_runs','runner_gates','independent_reference_records','max_cosine_error','failed_checks')},indent=2))
