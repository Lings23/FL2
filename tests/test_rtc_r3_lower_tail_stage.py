"""Synthetic analysis/entry-point checks; no real training."""
import copy
import json
from pathlib import Path
import pytest
from experiments import rtc_r3_lower_tail_stage as stage


def test_fixed_matrix_and_incomplete_run_rejection(tmp_path,monkeypatch):
    assert sum(len(ds) for _,_,_,ds in stage.batches())==24
    assert {s for _,_,s,_ in stage.batches()}=={201,202}
    for name,attack,seed,ds in stage.batches():
        command=stage.command(tmp_path,name,attack,seed,ds)
        assert command[-1]=='--dry-run' and '--smoke' not in command
        assert stage.command(tmp_path,name,attack,seed,ds,True)==command[:-1]
    monkeypatch.setattr(stage,'verify_lock',lambda root:{'cells':[dict(batch='b',run_id='r')]})
    monkeypatch.setattr(stage.prior,'ensure_no_training_process',lambda:None)
    monkeypatch.setattr(stage.subprocess,'run',lambda *a,**k:pytest.fail('Must not start training'))
    path=tmp_path/'b/status/r.json';path.parent.mkdir(parents=True)
    path.write_text(json.dumps(dict(state='failed',exit_code=1,last_round=3)))
    with pytest.raises(ValueError,match='Incomplete/failed'):stage.execute(tmp_path)


def test_quality_gates_reject_lie_failure_and_old_attack_regression(monkeypatch):
    runs=[]
    for name,attack,seed,ds in stage.batches():
        for defense in ds:
            candidate=defense==stage.OBSERVER
            acc=.82 if candidate and attack=='lie' else .8
            row=dict(attack=attack,seed=seed,defense=defense,active_accuracy=acc,final_accuracy=acc,
                     benign_union_flag_rate=0.,malicious_weight_per_round=.1 if candidate else .2)
            runs.append(dict(cell=dict(batch=name,attack=attack,seed=seed,defense=defense),computed=row,
                             rounds=[{'round':0}]))
    monkeypatch.setattr(stage,'metrics',lambda r:r['computed'])
    monkeypatch.setattr(stage.prior,'paired',lambda a,b:True)
    protocol=stage.prior.read_json(stage.PROTOCOL)
    assert stage.decide(runs,protocol)['candidate_accepted']
    bad=copy.deepcopy(runs)
    next(r for r in bad if r['cell']['attack']=='lie' and r['cell']['defense']==stage.OBSERVER)['computed']['active_accuracy']=.805
    assert not stage.decide(bad,protocol)['candidate_accepted']
    bad=copy.deepcopy(runs)
    next(r for r in bad if r['cell']['attack']=='sign_flip' and r['cell']['defense']==stage.OBSERVER)['computed']['final_accuracy']=.79
    assert not stage.decide(bad,protocol)['candidate_accepted']
    with pytest.raises(AssertionError):stage.decide(runs[:-1],protocol)


def lower_run():
    calibration=stage.lower_tail.load_calibration(stage.LOWER_CAL)
    rr=[dict(round='0')];cc=[]
    for rd in (1,2):
        rr.append(dict(round=str(rd),fit_rtc_r3l_version=stage.lower_tail.VERSION,fit_rtc_r3l_mode='cap',
                       fit_rtc_r3l_calibration_hash=stage.lower_tail.calibration_hash(calibration)))
        for i in range(10):
            small=i==9;flag=small and rd==2;norm=.1 if small else 3.
            cc.append(dict(round=str(rd),cid=str(i),principal_id=str(i),rtc_r3l_residual_norm=norm,
                rtc_r3l_principal_residual_norm=norm,rtc_r3l_reference_count=9,rtc_r3l_reference_median=3.,
                rtc_r3l_valid=True,rtc_r3l_low=small,rtc_r3l_ratio=norm/3.,rtc_r3l_streak=rd if small else 0,
                rtc_r3l_flagged=flag,rtc_r3l_q=0. if flag else 1.,rtc_r3l_existing_q=1.,rtc_r3l_nominal_mass=.1,
                rtc_r3l_applied=flag,rtc_r1_existing_q=1.,rtc_r1_q=1.,rtc_r2_q=1.,rtc_r1_nominal_mass=.1,
                aggregation_weight=0. if flag else .1))
    return dict(cell={'defense':stage.OBSERVER},rounds=rr,clients=cc),calibration


def test_independent_lower_replay_rejects_corrupt_state_and_cap():
    run,cal=lower_run();stage.verify_lower(run,cal)
    for key,value in [('rtc_r3l_streak',1),('rtc_r3l_q',1.),('aggregation_weight',.1),('rtc_r3l_ratio',2.)]:
        bad=copy.deepcopy(run);bad['clients'][-1][key]=value
        with pytest.raises(AssertionError):stage.verify_lower(bad,cal)


def test_float_encoded_observation_metadata_is_valid_but_drift_is_not():
    from analysis.rtc_r3_temporal_review.verify import metadata_matches
    assert metadata_matches('512.0',512) and metadata_matches('1.0',True)
    assert not metadata_matches('511.9',512) and not metadata_matches('nan',512)
    assert not metadata_matches('2.0',True)
