"""Synthetic evidence/aggregation only; no dataset or client training."""
import copy
import numpy as np
import pytest
from defenses.rtc.lower_tail import LowerTailEvidence,load_calibration

CAL='config/rtc_r3_lower_tail_calibration.json'


def test_two_visits_reset_and_prepare_is_transactional():
    d=LowerTailEvidence(load_calibration(CAL));ids=list(map(str,range(10)));mass=[.1]*10
    a,next_state=d.prepare([3.]*7+[.1]*3,ids,mass,'cap')
    assert d.state_dict()=={} and all(r['rtc_r3l_q']==1 for r in a)
    d.commit(next_state)
    b,next_state=d.prepare([3.]*7+[.1]*3,ids,mass,'cap')
    assert [r['rtc_r3l_flagged'] for r in b]==[False]*7+[True]*3
    assert [r['rtc_r3l_q'] for r in b]==[1.]*7+[0.]*3
    d.commit(next_state)
    normal,state=d.prepare([3.]*10,ids,mass,'cap')
    assert all(r['rtc_r3l_streak']==0 and r['rtc_r3l_q']==1 for r in normal)
    d.commit(state)
    assert not any(r['rtc_r3l_flagged'] for r in d.prepare([3.]*7+[.1]*3,ids,mass,'cap')[0])


def test_zero_median_abstains_and_aliases_do_not_multiply_evidence():
    d=LowerTailEvidence(load_calibration(CAL));ids=list(map(str,range(10)))
    zero,state=d.prepare([0.]*10,ids,[.1]*10,'cap')
    assert not any(r['rtc_r3l_valid'] or r['rtc_r3l_flagged'] for r in zero)
    base,_=d.prepare([3.]*9+[.1],ids,[.1]*10,'observe')
    alias,_=d.prepare([3.]*9+[.1,.1],ids+['9'],[.1]*9+[.05,.05],'observe')
    for a,b in zip(base,alias[:10]):
        assert a['rtc_r3l_ratio']==b['rtc_r3l_ratio'] and a['rtc_r3l_reference_count']==b['rtc_r3l_reference_count']
    assert alias[-1]['rtc_r3l_ratio']==base[-1]['rtc_r3l_ratio']
    with pytest.raises(ValueError):d.prepare([np.nan]*10,ids,[.1]*10,'cap')
    with pytest.raises(ValueError):d.prepare([3.]*10,ids,[-.1]*10,'cap')


def test_observe_records_same_evidence_without_cap():
    cal=load_calibration(CAL);ids=list(map(str,range(10)));d=LowerTailEvidence(cal)
    _,state=d.prepare([3.]*9+[.1],ids,[.1]*10,'observe');d.commit(state)
    observed,_=d.prepare([3.]*9+[.1],ids,[.1]*10,'observe')
    capped,_=d.prepare([3.]*9+[.1],ids,[.1]*10,'cap')
    assert observed[-1]['rtc_r3l_flagged'] and observed[-1]['rtc_r3l_q']==1
    assert capped[-1]['rtc_r3l_flagged'] and capped[-1]['rtc_r3l_q']==0


def test_phase6_lower_cap_after_two_rounds_preserves_budget():
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
        raw_norm_mode='cap',raw_norm_calibration='config/rtc_r2_raw_norm_calibration.json',
        lower_tail_mode='cap',lower_tail_calibration=CAL)
    d=RTCv3Defense(DefenseConfig(enabled=True,type='rtc_v3_candidate',custom_params=params),num_clients=10)
    off=copy.deepcopy(params);off.pop('lower_tail_mode');off.pop('lower_tail_calibration')
    baseline=RTCv3Defense(DefenseConfig(enabled=True,type='rtc_v3_candidate',custom_params=off),num_clients=10)
    observer=RTCv3Defense(DefenseConfig(enabled=True,type='rtc_v3_candidate',custom_params={**params,'lower_tail_mode':'observe'}),num_clients=10)
    base=_params()
    parent_base=_params()
    for r in range(1,4):
        d.set_context(r,ids,base,principal_ids=ids,parameter_roles=ROLES,parameter_names=NAMES,trainable_parameter_indices=[0,1,2])
        step=[-.03,-.02,-.01,.01,.02,.03,.04,0.,0.,0.]
        updates=[([base[0]+np.ones(4,np.float32)*v,base[1].copy(),base[2].copy()],10) for v in step]
        base=d.aggregate(updates)
        assert d.last_round_metrics['rtc_v3_max_constraint_violation']<=1e-8
        if r>=2:
            assert all(d._last_lower_tail_rows[i]['rtc_r3l_flagged'] for i in (7,8,9))
            assert all(d.last_client_aggregation_weights[str(i)]<=1e-8 for i in (7,8,9))
        parents=[]
        for defense in (baseline,observer):
            defense.set_context(r,ids,parent_base,principal_ids=ids,parameter_roles=ROLES,parameter_names=NAMES,trainable_parameter_indices=[0,1,2])
            parent_updates=[([parent_base[0]+np.ones(4,np.float32)*v,parent_base[1].copy(),parent_base[2].copy()],10) for v in step]
            parents.append(defense.aggregate(parent_updates))
        assert all(np.array_equal(a,b) for a,b in zip(*parents))
        assert baseline.last_client_aggregation_weights==observer.last_client_aggregation_weights
        assert baseline._cumulative.state_dict()==observer._cumulative.state_dict()
        parent_base=parents[0]
