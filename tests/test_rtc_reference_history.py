"""Synthetic state/aggregation and manual handoff checks; no training."""
import copy
import json
import numpy as np
import pytest
from defenses.rtc.reference_history import ReferenceHistory,fingerprint
from defenses.rtc.spectral_direction import measure,load_calibration
from experiments import rtc_reference_history_observation as stage


def fixture_measure():
    rng=np.random.default_rng(617)
    clipped=[[np.array([1.,.05])+rng.normal(0,.002,2)] for _ in range(10)]
    measured=measure(clipped,[0],load_calibration('config/rtc_r1c_pairwise_calibration.json'),'cap')
    return clipped,measured


def test_exact_full_space_cosines_transactional_history_and_zero():
    clipped,measured=fixture_measure();h=ReferenceHistory()
    original=copy.deepcopy(clipped);q=[r['q'] for r in measured['rows']]
    rows,summary,next_state=h.prepare(clipped,[np.array([1.,0.])],measured)
    assert h.previous is None and all(not r['rtc_r1h_valid'] for r in rows)
    assert summary['rtc_r1h_previous_norm'] is None
    h.commit(next_state);next_state[0][:]=99
    rows,summary,next_state=h.prepare(clipped,[np.array([-1.,0.])],measured)
    assert summary['rtc_r1h_previous_sha256']==fingerprint([np.array([1.,0.])])
    for row,score in zip(rows,measured['rows']):
        ref=sum(w*clipped[i][0]/np.linalg.norm(clipped[i][0]) for i,w in zip(score['reference_indices'],score['reference_weights']))
        assert row['rtc_r1h_reference_cosine']==pytest.approx(ref[0]/np.linalg.norm(ref),abs=1e-12)
    h.commit(next_state)
    rows,_,_=h.prepare(clipped,[np.zeros(2)],measured)
    assert all(r['rtc_r1h_reference_cosine']<0 for r in rows)
    h.commit([np.zeros(2)])
    assert all(not r['rtc_r1h_valid'] for r in h.prepare(clipped,[np.zeros(2)],measured)[0])
    assert all(np.array_equal(a[0],b[0]) for a,b in zip(original,clipped))
    assert q==[r['q'] for r in measured['rows']]
    with pytest.raises(ValueError,match='layout'):h.prepare(clipped,[np.zeros(3)],measured)


def test_independent_history_analyzer_rejects_wrong_chain_and_projection():
    clipped,m=fixture_measure();h=ReferenceHistory();cc=[];rr=[dict(round='0')]
    ids=list(map(str,range(10)))
    for rd in (1,2):
        actual=[np.array([1.,0.])];rows,summary,nxt=h.prepare(clipped,actual,m);h.commit(nxt)
        r=dict(round=str(rd),fit_rtc_r1_client_ids_json=json.dumps(ids),fit_rtc_r1_gram_json=json.dumps(m['gram'].tolist()),fit_rtc_r1_actual_aggregate_norm='1.0')
        r.update({'fit_'+k:('' if v is None else str(v)) for k,v in summary.items()});rr.append(r)
        for cid,row,score in zip(ids,rows,m['rows']):
            c=dict(round=str(rd),cid=cid,attack_active=False,rtc_r1_valid=score['valid'],rtc_r1_flagged=False,
                rtc_r1_reference_ids_json=json.dumps([ids[i] for i in score['reference_indices']]),
                rtc_r1_reference_weights_json=json.dumps(score['reference_weights']))
            c.update({k:('' if v is None else str(v)) for k,v in row.items()});cc.append(c)
    run=dict(rounds=rr,clients=cc)
    assert len(stage.verify_history(run))==20
    for key,value in [('rtc_r1h_reference_cosine','-.5'),('rtc_r1h_previous_dot','1000')]:
        bad=copy.deepcopy(run);bad['clients'][-1][key]=value
        with pytest.raises(AssertionError):stage.verify_history(bad)
    bad=copy.deepcopy(run);bad['rounds'][-1]['fit_rtc_r1h_previous_sha256']='f'*64
    with pytest.raises(AssertionError):stage.verify_history(bad)


def test_stage_defaults_to_five_dry_runs_and_refuses_failed_run(tmp_path,monkeypatch):
    assert sum(len(ds) for _,_,_,ds in stage.batches())==5
    for name,attack,seed,ds in stage.batches():
        command=stage.command(tmp_path,name,attack,seed,ds)
        assert seed==201 and command[-1]=='--dry-run' and '--smoke' not in command
        assert stage.command(tmp_path,name,attack,seed,ds,True)==command[:-1]
    monkeypatch.setattr(stage,'verify_lock',lambda root:{'cells':[dict(batch='b',run_id='r')]})
    monkeypatch.setattr(stage.prior,'ensure_no_training_process',lambda:None)
    monkeypatch.setattr(stage.subprocess,'run',lambda *a,**kw:pytest.fail('Must not start training'))
    p=tmp_path/'b/status/r.json';p.parent.mkdir(parents=True)
    p.write_text(json.dumps(dict(state='failed',exit_code=1,last_round=2)))
    with pytest.raises(ValueError,match='Incomplete'):stage.execute(tmp_path)
    with pytest.raises(AssertionError):stage.decide([])


def test_phase6_observer_preserves_actual_model_caps_weights_and_cumulative():
    from tests.test_rtc_v3_semantic_temporal_exposure import _manifest,_params,ROLES,NAMES
    from config.config_loader import DefenseConfig
    from defenses.rtc.v3 import RTCv3Defense
    from defenses.rtc.calibration import content_hash
    ids=list(map(str,range(10)));manifest=_manifest(frozen_cones=True)
    manifest['cumulative']={'enabled':True,'resolutions':{'full':dict(scale_center=1.,scale_lower=1.,scale_upper=1.,kappa=.5,threshold=.5,eta=2.,q_min=.5)}}
    manifest['content_hash']=content_hash(manifest)
    params=dict(calibration_manifest=manifest,implementation_phase=6,parameter_roles=ROLES,
        principal_map={i:i for i in ids},principal_first_sampling_verified=True,semantic_intervention_risk_floor=.5,
        cumulative_q_cap_power=1,anchor_recycle_fraction=.51,anchor_recycle_weighting='accepted',
        spectral_direction_mode='cap',spectral_direction_calibration='config/rtc_r1c_pairwise_calibration.json',
        raw_norm_mode='cap',raw_norm_calibration='config/rtc_r2_raw_norm_calibration.json')
    a=RTCv3Defense(DefenseConfig(enabled=True,type='rtc_v3_candidate',custom_params=params),num_clients=10)
    b=RTCv3Defense(DefenseConfig(enabled=True,type='rtc_v3_candidate',custom_params={**params,'reference_history_observe_only':True}),num_clients=10)
    base=_params();last_hash=''
    for rd in range(1,4):
        results=[]
        for d in (a,b):
            d.set_context(rd,ids,base,principal_ids=ids,parameter_roles=ROLES,parameter_names=NAMES,trainable_parameter_indices=[0,1,2])
            updates=[([base[0]+np.ones(4,np.float32)*(i-4)*.01,base[1].copy(),base[2].copy()],10) for i in range(10)]
            results.append(d.aggregate(updates))
        assert all(np.array_equal(x,y) for x,y in zip(*results))
        assert a.last_client_aggregation_weights==b.last_client_aggregation_weights
        assert a._cumulative.state_dict()==b._cumulative.state_dict()
        assert a._last_raw_norm_rows==b._last_raw_norm_rows
        assert b.last_round_metrics['rtc_r1h_previous_sha256']==last_hash
        last_hash=b.last_round_metrics['rtc_r1h_current_sha256']
        base=results[0]
