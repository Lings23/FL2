"""Synthetic caps, budget, audit and manual-execution tests. No training."""
import copy
import json
import numpy as np
import pytest
from defenses.rtc.filtered_lower_tail import FilteredLowerTailEvidence
from defenses.rtc.lower_tail import load_calibration,calibration_hash,VERSION,LowerTailEvidence
from experiments import rtc_g2_integration as stage
from experiments.rtc_g2_validation import integer,verify_lower
from tests.test_rtc_i12_bridge import combined_artifacts
from tests.test_rtc_r2_stage import write_csv


def test_integer_csv_accepts_decimal_not_fraction_or_nonfinite():
    assert integer('2.0')==2 and integer('3')==3
    for v in ('2.5','nan','inf',''):
        with pytest.raises((AssertionError,ArithmeticError)):integer(v)


@pytest.mark.parametrize('policy',['all','raw_eligible'])
@pytest.mark.parametrize('mode',['observe','cap'])
def test_full_lower_replay_rejects_state_reference_and_weight_forgery(policy,mode):
    cal=load_calibration(stage.LOWER_CAL)
    scorer=(FilteredLowerTailEvidence if policy=='raw_eligible' else LowerTailEvidence)(cal)
    ids=list(map(str,range(10)));cc=[];rr=[{'round':'0'}]
    for rnd in range(1,61):
        raw=[False]*9+[rnd==3]
        extra={'raw_rejected':raw} if policy=='raw_eligible' else {}
        scores,state=scorer.prepare([1.]*7+[.5]*3,ids,[.1]*10,mode,**extra)
        scorer.commit(state)
        rr.append(dict(round=str(rnd),fit_rtc_r3l_version=VERSION,fit_rtc_r3l_mode=mode,
            fit_rtc_r3l_reference_policy=policy,fit_rtc_r3l_calibration_hash=calibration_hash(cal),fit_rtc_r3l_seconds='0.1'))
        for i,(pid,row) in enumerate(zip(ids,scores)):
            existing=0. if raw[i] else 1.
            row.update(round=str(rnd),cid=pid,principal_id=pid,rtc_r1_nominal_mass=.1,
                rtc_r1_existing_q=1.,rtc_r1_q=1.,rtc_r2_q=existing,rtc_r2_flagged=raw[i],
                rtc_r3l_existing_q=existing,rtc_r3l_nominal_mass=.1,
                rtc_r3l_applied=row['rtc_r3l_q']<existing,aggregation_weight=.1*min(existing,row['rtc_r3l_q']))
            # Match CSV serialization, including optional numeric metadata.
            cc.append({k:'' if v is None else str(v) for k,v in row.items()})
    run=dict(rounds=rr,clients=cc)
    verify_lower(run,cal,mode,policy)
    for key,value in [('rtc_r3l_reference_count','8.5'),('rtc_r3l_streak','9'),
                      ('rtc_r3l_reference_median','99'),('rtc_r3l_q','.3'),('aggregation_weight','.2')]:
        bad=copy.deepcopy(run);bad['clients'][17][key]=value
        with pytest.raises(AssertionError):verify_lower(bad,cal,mode,policy)


def test_actual_aggregate_filters_raw_rejects_and_observer_keeps_m3_trajectory():
    from tests.test_rtc_v3_semantic_temporal_exposure import _manifest,_params,ROLES,NAMES
    from config.config_loader import DefenseConfig
    from defenses.rtc.v3 import RTCv3Defense
    from defenses.rtc.calibration import content_hash
    ids=list(map(str,range(10)));manifest=_manifest(frozen_cones=True)
    manifest['cumulative']={'enabled':True,'resolutions':{'full':dict(scale_center=1.,scale_lower=1.,scale_upper=1.,kappa=.5,threshold=.5,eta=2.,q_min=.5)}}
    manifest['content_hash']=content_hash(manifest)
    p=dict(calibration_manifest=manifest,implementation_phase=6,parameter_roles=ROLES,
        principal_map={i:i for i in ids},principal_first_sampling_verified=True,semantic_intervention_risk_floor=.5,
        cumulative_q_cap_power=1,anchor_recycle_fraction=.51,anchor_recycle_weighting='accepted',
        spectral_direction_mode='cap',spectral_direction_calibration='config/rtc_r1c_pairwise_calibration.json',
        raw_norm_mode='cap',raw_norm_calibration='config/rtc_r2_raw_norm_calibration.json',
        reference_eligibility_mode='cap',reference_eligibility_memory='confirmed')
    make=lambda cfg:RTCv3Defense(DefenseConfig(enabled=True,type='rtc_v3_candidate',custom_params=cfg),num_clients=10)
    parents=[make(p),make(dict(p,lower_tail_mode='observe',lower_tail_reference_policy='raw_eligible',lower_tail_calibration=str(stage.LOWER_CAL)))]
    cap=make(dict(p,lower_tail_mode='cap',lower_tail_reference_policy='raw_eligible',lower_tail_calibration=str(stage.LOWER_CAL)))
    base=_params();parent_base=_params()
    for rnd in range(1,4):
        steps=[-.03,-.02,-.01,.01,.02,.03,.04,0.,0.,0.]
        if rnd==3:
            steps[0]=100.
            steps[7:]=[.005,.006,.007]  # R2 requires nine positive-norm peers.
        outputs=[]
        for d,b in [(cap,base),*( (d,parent_base) for d in parents)]:
            d.set_context(rnd,ids,b,principal_ids=ids,parameter_roles=ROLES,parameter_names=NAMES,trainable_parameter_indices=[0,1,2])
            updates=[([b[0]+np.ones(4,np.float32)*v,b[1].copy(),b[2].copy()],10) for v in steps]
            outputs.append(d.aggregate(updates))
            assert d.last_round_metrics['rtc_v3_max_constraint_violation']<=1e-8
        assert all(np.array_equal(a,b) for a,b in zip(outputs[1],outputs[2]))
        assert parents[0].last_client_aggregation_weights==parents[1].last_client_aggregation_weights
        assert parents[0]._reference_eligibility.state_dict()==parents[1]._reference_eligibility.state_dict()
        if rnd==2:
            assert all(cap._last_lower_tail_rows[i]['rtc_r3l_q']==0 for i in (7,8,9))
            assert all(cap.last_client_aggregation_weights[str(i)]<=1e-8 for i in (7,8,9))
        if rnd==3:
            assert any(c['rtc_r2_flagged'] for c in cap._last_raw_norm_rows)
            assert all(not c['rtc_r3l_valid'] and c['rtc_r3l_streak']==0 for c in cap._last_lower_tail_rows)
        base,parent_base=outputs[0],outputs[1]
    for changes in [dict(lower_tail_reference_policy='bad'),dict(lower_tail_reference_policy='raw_eligible'),
                    dict(lower_tail_reference_policy='raw_eligible',lower_tail_mode='cap',raw_norm_mode='observe')]:
        with pytest.raises(ValueError):make(dict(p,**changes))


def test_each_seed_safety_and_g1_retention_gate_is_required(monkeypatch):
    runs=[]
    for name,attack,seed,ds in stage.batches():
        for d in ds:
            m=dict(attack=attack,seed=seed,defense=d,active_accuracy=.83 if d==stage.BASE else .85,
                final_accuracy=.88,benign_union_flag_rate=0.,benign_union_flags=0,malicious_weight_per_round=.2 if d==stage.BASE else .1)
            runs.append(dict(cell=dict(batch=name,attack=attack,seed=seed,defense=d),computed=m,rounds=[{}],clients=[]))
    monkeypatch.setattr(stage,'metrics',lambda r:r['computed'])
    monkeypatch.setattr(stage.prior,'paired',lambda *a:None)
    protocol=stage.prior.read_json(stage.PROTOCOL)
    assert stage.decide(runs,protocol)['candidate_accepted']
    for key,value in [('active_accuracy',.835),('final_accuracy',.87),('benign_union_flag_rate',.02),
                      ('benign_union_flags',1),('malicious_weight_per_round',.18)]:
        bad=copy.deepcopy(runs)
        row=next(r for r in bad if r['cell']['attack']=='lie' and r['cell']['seed']==206 and r['cell']['defense']==stage.OBSERVER)
        row['computed'][key]=value
        assert not stage.decide(bad,protocol)['candidate_accepted']
    with pytest.raises(AssertionError):stage.decide(runs[:-1],protocol)


def test_manual_guard_matrix_and_resource_contract(tmp_path,monkeypatch):
    from experiments.run import parse_args
    assert sum(len(ds) for _,_,_,ds in stage.batches())==32
    for n,a,s,ds in stage.batches():
        cmd=stage.command(tmp_path,n,a,s,ds);args=parse_args(cmd[3:])
        assert args.dry_run and args.batch_size==96 and args.ray_client_num_gpus==.125
    monkeypatch.setattr(stage,'verify_lock',lambda root:{'cells':[dict(batch='b',run_id='r')]})
    monkeypatch.setattr(stage.prior,'ensure_no_training_process',lambda:None)
    monkeypatch.setattr(stage.subprocess,'run',lambda *a,**kw:pytest.fail('Training forbidden'))
    p=tmp_path/'b/status/r.json';p.parent.mkdir(parents=True)
    p.write_text(json.dumps(dict(state='failed',exit_code=1,last_round=2)))
    with pytest.raises(ValueError):stage.execute(tmp_path)


def test_combined_memory_and_filtered_observer_full_artifact_validation(combined_artifacts):
    from defenses.rtc.reference_eligibility import ReferenceEligibility,VERSION as MEMORY_VERSION
    from defenses.rtc.spectral_direction import measure
    root,cell,lock=combined_artifacts;cell['defense']=stage.BASE
    batch=root/cell['batch'];rid=cell['run_id'];v=stage.validator
    rr=v.read_csv(batch/'rounds'/(rid+'.csv'));cc=v.read_csv(batch/'raw'/(rid+'_clients.csv'))
    memory=ReferenceEligibility();cal=load_calibration(stage.LOWER_CAL);lower=FilteredLowerTailEvidence(cal)
    for row in cc:row['principal_id']=row['cid']
    for r in rr[1:]:
        active=int(r['round'])>=11
        x=np.column_stack([np.array([1]*7+([-1,1,-1] if active else [1]*3)),np.eye(10)*.2])
        measured=measure([[a] for a in x],[0],lock['calibration'],'cap')
        clients=[c for c in cc if c['round']==r['round']];ids=[c['cid'] for c in clients]
        raw=[dict(rtc_r2_flagged=v.truth(c['rtc_r2_flagged'])) for c in clients]
        extra,original=memory.prepare(measured,ids,'cap')
        memory.commit(memory.transition_confirmed(ids,original,raw,extra))
        low,state=lower.prepare([1.]*10,ids,[float(c['rtc_r1_nominal_mass']) for c in clients],'observe',
            [c['rtc_r2_flagged'] for c in raw]);lower.commit(state)
        r.update(fit_rtc_r1e_version=MEMORY_VERSION,fit_rtc_r1e_mode='cap',fit_rtc_r1e_memory='confirmed',
            fit_rtc_r1e_quorum_numerator='2.0',fit_rtc_r1e_quorum_denominator='3.0',
            fit_rtc_r3l_version=VERSION,fit_rtc_r3l_mode='observe',fit_rtc_r3l_reference_policy='raw_eligible',
            fit_rtc_r3l_calibration_hash=calibration_hash(cal),fit_rtc_r3l_seconds=.1)
        for c,e,l in zip(clients,extra,low):
            c.update(e);c.update(l)
            existing=min(float(c['rtc_r1_existing_q']),float(c['rtc_r1_q']),float(c['rtc_r2_q']))
            c.update(rtc_r3l_existing_q=existing,rtc_r3l_nominal_mass=c['rtc_r1_nominal_mass'],rtc_r3l_applied=False)
    write_csv(batch/'rounds'/(rid+'.csv'),rr);write_csv(batch/'raw'/(rid+'_clients.csv'),cc)
    assert len(stage.verify_run(root,cell,lock)['clients'])==600
