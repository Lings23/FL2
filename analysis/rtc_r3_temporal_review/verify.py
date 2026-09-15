"""Offline import audit of frozen R3 temporal logs; no training or lock mutation."""
from pathlib import Path
import csv
import hashlib
import json
import math
import sys
import types
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments import rtc_i12_validation as validation
from experiments import rtc_r3_temporal_observation as stage

BATCH = ROOT / 'logs/rtc_r3_temporal_observation'
HERE = Path(__file__).resolve().parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def metadata_matches(actual, expected):
    # CSV transports numeric Flower metrics as floats, including bools.
    if isinstance(expected, (int, float, bool)):
        try:
            value = float(actual)
            return math.isfinite(value) and value == float(expected)
        except (ValueError, TypeError):
            return False
    return actual == expected


def main():
    lock = read(BATCH / 'r3t_lock.json')
    assert lock['protocol'] == read(stage.PROTOCOL)
    assert lock['accepted_parent'] == read(stage.RECEIPT)
    assert len(lock['cells']) == lock['training_units'] == 10
    assert {(c['batch'], c['attack'], c['seed'], c['defense']) for c in lock['cells']} == {
        (n,a,s,d) for n,a,s,ds in stage.batches() for d in ds}
    for name, expected in lock['artifacts'].items():
        assert sha((BATCH / name).read_bytes()) == expected, name
    with zipfile.ZipFile(BATCH / 'r3t_sources.zip') as archive:
        assert set(archive.namelist()) == set(lock['sources'])
        sources = {n: archive.read(n) for n in archive.namelist()}
    for name, expected in lock['sources'].items():
        assert sha(sources[name]) == expected, name
    for name in ('experiments/rtc_r3_temporal_observation.py', 'experiments/rtc_i12_validation.py',
                 'experiments/rtc_i12_bridge.py', 'experiments/rtc_r2_raw_norm.py',
                 'defenses/rtc/spectral_direction.py', 'defenses/rtc/corroboration.py',
                 'defenses/rtc/raw_norm.py', 'defenses/rtc/temporal_observe.py',
                 'analysis/rtc_r3_diagnostics/replay.py'):
        assert (ROOT/name).read_bytes().replace(b'\r\n',b'\n') == sources[name].replace(b'\r\n',b'\n'), name

    mappings = set()
    def imported_read(path):
        p=Path(path); s=p.as_posix()
        if p.is_file() and p.resolve().is_relative_to(BATCH):
            return read(p)
        if '/logs/rtc_r3_temporal_observation/' in s:
            rel=s.split('/logs/rtc_r3_temporal_observation/',1)[1]
            target=(BATCH/rel).resolve()
            assert target.is_relative_to(BATCH) and rel in lock['artifacts']
            assert sha(target.read_bytes()) == lock['artifacts'][rel]
            mappings.add((s,str(target)))
            return read(target)
        if '/config/' in s:
            rel='config/'+s.split('/config/',1)[1]
            return json.loads(sources[rel])
        return read(p)
    scope=dict(validation.verify_cell.__globals__, read_json=imported_read,
               MODES=stage.MODES, RTC=(stage.PARENT,stage.OBSERVER))
    verify=types.FunctionType(validation.verify_cell.__code__,scope)
    runs=[verify(BATCH,c,lock) for c in lock['cells']]
    score_rows=[]; checks={}; summaries=[]; runner_gates=0
    for name,attack,seed,defenses in stage.batches():
        index={r['cell']['defense']:r for r in runs if r['cell']['batch']==name}
        a,b=index[stage.PARENT],index[stage.OBSERVER]
        for run in index.values():
            stage.prior.paired(a,run)
        with (BATCH/name/'quality_gates.csv').open(encoding='utf-8-sig') as f:
            runner_gates += len(list(csv.DictReader(f)))
        for x,y in zip(a['rounds'][1:],b['rounds'][1:]):
            for key in ('server_accuracy','server_loss','fit_aggregate_update_sketch_json','fit_defense_random_seed'):
                assert x[key]==y[key],(name,x['round'],key)
            for key,value in stage.metadata().items():
                assert metadata_matches(y['fit_'+key],value),(key,y['fit_'+key],value)
        original={(r['round'],r['cid']):r for r in a['clients']}
        for row in b['clients']:
            for key in ('aggregation_weight','rtc_r1_q','rtc_r2_q','rtc_v3_cumulative_q_full'):
                assert row[key]==original[row['round'],row['cid']][key]
        causal=stage.coherence(b['clients'])
        lookup={(int(r['round']),r['cid']):r for r in b['clients']}
        for row in causal:
            source=lookup[row['round'],row['cid']]
            # Independent pairwise-cosine identity: ||mean unit vectors||^2.
            history=sorted([r for r in b['clients'] if r['principal_id']==row['principal_id']
                            and int(r['round'])<=row['round']],key=lambda r:int(r['round']))[-5:]
            vectors=[np.asarray(json.loads(r['rtc_r3t_sketch_json']),dtype=float) for r in history]
            norms=[float(np.linalg.norm(v)) for v in vectors]
            valid=len(history)==5 and all(n>1e-12 for n in norms)
            assert row['valid']==valid
            if valid:
                independent=math.sqrt(max(0.,sum(float(x@y)/(nx*ny)
                    for x,nx in zip(vectors,norms) for y,ny in zip(vectors,norms))/25))
                assert abs(independent-row['coherence'])<1e-12
            peers=[float(r['rtc_r3t_residual_norm']) for r in b['clients']
                   if int(r['round'])==row['round'] and r['principal_id']!=row['principal_id']]
            assert len(peers)==9
            median=float(np.median(peers))
            ratio=float(source['rtc_r3t_residual_norm'])/median if median>1e-12 else None
            score_rows.append(dict(batch=name,attack=attack,seed=seed,**row,
                trainable_residual_norm=float(source['rtc_r3t_residual_norm']),
                sketch_norm=float(np.linalg.norm(json.loads(source['rtc_r3t_sketch_json']))),
                loo_residual_ratio=ratio, aggregation_weight=float(source['aggregation_weight']),
                existing_cumulative_q=float(source['rtc_v3_cumulative_q_full'])))
        checks[name+'/inert_trajectory_and_weights']=True
        checks[name+'/typed_metadata_and_independent_causal_scores']=True
    for r in runs:
        summaries.append(stage.prior.summarize(r))
    result=dict(stage='R3-temporal-observe',observation_quality_accepted=True,candidate_accepted=False,
        final_goal_achieved=False,source_lock_sha256=sha((BATCH/'r3t_lock.json').read_bytes()),
        source_archive_sha256=sha((BATCH/'r3t_sources.zip').read_bytes()),sources_verified=len(sources),
        artifacts_verified=len(lock['artifacts']),verified_runs=len(runs),runner_gates=runner_gates,
        checks=checks,summaries=summaries,path_mappings=sorted(mappings),training_environment=lock['environment'],
        analyzer_transport_fix='Numeric/bool Flower CSV values compared by finite numeric value, not textual spelling; thresholds unchanged',
        conclusions=['Observation is inert and valid; not a new defense acceptance',
            'Original high-coherence hypothesis is not supported by the development LIE runs',
            'Low trainable residual relative to peers is exploratory and requires separate clean-only calibration and preregistration'],
        evidence={p.relative_to(BATCH).as_posix():sha(p.read_bytes()) for p in BATCH.rglob('*')
                  if p.is_file() and p.suffix in ('.json','.csv')})
    HERE.mkdir(exist_ok=True)
    (HERE/'review.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    with (HERE/'causal_scores.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(score_rows[0]));writer.writeheader();writer.writerows(score_rows)
    print(json.dumps({k:result[k] for k in ('observation_quality_accepted','candidate_accepted','verified_runs','runner_gates','sources_verified','artifacts_verified')},indent=2))


if __name__=='__main__':
    if not __debug__: raise RuntimeError('Assertions must remain enabled')
    main()
