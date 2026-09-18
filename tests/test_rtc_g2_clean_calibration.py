"""Synthetic inputs only. No clients, datasets, or training subprocesses."""
import copy
import json
import pytest
from defenses.rtc.filtered_lower_tail import FilteredLowerTailEvidence
from defenses.rtc.lower_tail import VERSION
from experiments import rtc_g2_clean_calibration as stage


def scorer():
    return FilteredLowerTailEvidence(dict(version=VERSION,ratio_threshold=.8,
        min_peers=9,required_visits=2,cap_multiplier=0,provenance={}))


def test_rejected_principal_removal_abstains_and_resets_without_committing():
    s=scorer();ids=list(map(str,range(10)))
    rows,state=s.prepare([.5]+[1.]*9,ids,[.1]*10,'cap',[False]*10)
    assert not rows[0]['rtc_r3l_flagged'] and not s.state_dict()
    s.commit(state)
    rows,state=s.prepare([.5]+[1.]*9,ids,[.1]*10,'cap',[False]*9+[True])
    assert all(not r['rtc_r3l_valid'] and r['rtc_r3l_q']==1 for r in rows)
    assert state['0']==0 and s.state_dict()['0']==1
    s.commit(state)
    rows,state=s.prepare([.5]+[1.]*9,ids,[.1]*10,'cap',[False]*10)
    assert not rows[0]['rtc_r3l_flagged']
    s.commit(state)
    rows,_=s.prepare([.5]+[1.]*9,ids,[.1]*10,'cap',[False]*10)
    assert rows[0]['rtc_r3l_q']==0


def test_duplicate_principal_rejection_cannot_reenter_reference():
    ids=list(map(str,range(10)))+['9'];s=scorer()
    rows,_=s.prepare([1.]*11,ids,[.1]*11,'observe',[False]*10+[True])
    assert not rows[9]['rtc_g2_reference_eligible'] and not rows[10]['rtc_g2_reference_eligible']
    assert rows[0]['rtc_r3l_reference_count']==8
    with pytest.raises(ValueError):s.prepare([1.]*10,ids[:10],[.1]*9+[-.1],'cap',[False]*9+[True])
    with pytest.raises(ValueError):s.prepare([1.]*10,ids[:10],[.1]*10,'cap',[0]*10)


def synthetic_runs():
    runs=[]
    for _,attack,seed,ds in stage.batches():
        for defense in ds:
            rounds=[dict(round=str(r),server_accuracy='.8',server_loss='1',fit_aggregate_update_sketch_json='[]',fit_defense_random_seed='1') for r in range(61)]
            clients=[dict(round=str(r),cid=str(i),principal_id=str(i),rtc_r3l_residual_norm=.9 if i==0 else 1.,
                rtc_r1_nominal_mass=.1,rtc_r2_flagged=False,rtc_r1_q=1.,rtc_r2_q=1.,aggregation_weight=.1)
                for r in range(1,61) for i in range(10)]
            runs.append(dict(cell=dict(seed=seed,attack=attack,defense=defense),rounds=rounds,clients=clients))
    return runs


def test_threshold_only_uses_calibration_and_holdout_failure_is_not_retuned(monkeypatch):
    monkeypatch.setattr(stage.prior,'paired',lambda *a:None)
    runs=synthetic_runs();protocol={'pending':'attack integration'}
    good=stage.decide(runs,protocol)
    assert good['threshold']==.9 and good['calibration_accepted'] and not good['candidate_accepted']
    for run in runs:
        if run['cell']['seed']==107 and run['cell']['defense']==stage.OBSERVER:
            for row in run['clients']:
                if row['cid']=='0':row['rtc_r3l_residual_norm']=.5
    bad=stage.decide(runs,protocol)
    assert bad['threshold']==.9 and not bad['calibration_accepted']
    assert bad['holdout_flags']==59
    runs[0]['rounds'][1]['server_accuracy']='.7'
    with pytest.raises(AssertionError):stage.decide(runs,protocol)


def test_defaults_resources_and_failed_run_do_not_start_training(tmp_path,monkeypatch):
    from experiments.run import parse_args
    assert sum(len(ds) for _,_,_,ds in stage.batches())==4
    for name,attack,seed,ds in stage.batches():
        cmd=stage.command(tmp_path,name,attack,seed,ds)
        args=parse_args(cmd[3:])
        assert args.dry_run and args.batch_size==96 and args.ray_client_num_gpus==.125
        assert attack=='none' and cmd[-1]=='--dry-run'
    monkeypatch.setattr(stage,'verify_lock',lambda root:{'cells':[dict(batch='b',run_id='r')]})
    monkeypatch.setattr(stage.prior,'ensure_no_training_process',lambda:None)
    monkeypatch.setattr(stage.subprocess,'run',lambda *a,**kw:pytest.fail('Training forbidden'))
    p=tmp_path/'b/status/r.json';p.parent.mkdir(parents=True)
    p.write_text(json.dumps(dict(state='failed',exit_code=1,last_round=2)))
    with pytest.raises(ValueError):stage.execute(tmp_path)


def test_missing_outputs_fail_before_analysis(tmp_path,monkeypatch):
    monkeypatch.setattr(stage,'verify_lock',lambda root:{'cells':[dict(batch='b',run_id='r',defense=stage.BASE)]})
    with pytest.raises(FileNotFoundError):stage.analyze(tmp_path)
