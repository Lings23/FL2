"""Synthetic H1 evidence and full aggregation checks; no client training."""
import copy
import numpy as np
import pytest
from defenses.rtc import reference_guard
from defenses.rtc.spectral_direction import measure,load_calibration


def measured(signs):
    x=[[np.array([float(s),.01*float(s)])] for s in signs]
    return x,measure(x,[0],load_calibration('config/rtc_r1c_pairwise_calibration.json'),'cap')


def test_positive_reference_preserves_cap_negative_reference_vetoes():
    x,m=measured([1]*7+[-1]*3)
    original=copy.deepcopy(m)
    records=reference_guard.apply(m,x,[np.array([1.,0.])],'cap')
    assert [r['q'] for r in m['rows']]==[r['q'] for r in original['rows']]
    assert all(m['rows'][i]['flagged'] for i in (7,8,9))
    x,m=measured([1]*4+[-1]*6)
    assert all(m['rows'][i]['flagged'] for i in range(4))
    records=reference_guard.apply(m,x,[np.array([1.,0.])],'cap')
    assert all(records[i]['rtc_r1g_vetoed'] and m['rows'][i]['q']==1 for i in range(4))
    assert all(np.array_equal(a[0],b[0]) for a,b in zip(x,measured([1]*4+[-1]*6)[0]))


def test_observe_missing_zero_and_strict_zero_boundary():
    x,m=measured([1]*7+[-1]*3);old=copy.deepcopy(m)
    reference_guard.apply(m,x,None,'observe')
    assert [r['q'] for r in m['rows']]==[r['q'] for r in old['rows']]
    for previous in (None,[np.zeros(2)],[np.array([.01,-1.])]):
        m=copy.deepcopy(old);rows=reference_guard.apply(m,x,previous,'cap')
        assert all(r['q']==1 for r in m['rows'])
        assert all(not r['rtc_r1g_passes'] for r in rows)
    with pytest.raises(ValueError):reference_guard.apply(m,x,[np.array([np.nan,0])],'cap')
    with pytest.raises(ValueError):reference_guard.apply(m,x,[np.zeros(3)],'cap')
    with pytest.raises(ValueError):reference_guard.apply(m,x,None,'invalid')


def test_phase6_guard_observe_is_inert(monkeypatch):
    from tests.test_rtc_reference_history import test_phase6_observer_preserves_actual_model_caps_weights_and_cumulative as check
    from defenses.rtc import v3
    original=v3.RTCv3Defense
    def build(cfg,*args,**kwargs):
        cfg=copy.deepcopy(cfg)
        if cfg.custom_params.pop('reference_history_observe_only',False):
            cfg.custom_params['reference_guard_mode']='observe'
        return original(cfg,*args,**kwargs)
    monkeypatch.setattr(v3,'RTCv3Defense',build)
    check()


def test_phase6_cap_veto_preserves_budget_and_logs_original_flags():
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
        raw_norm_mode='cap',raw_norm_calibration='config/rtc_r2_raw_norm_calibration.json',reference_guard_mode='cap')
    d=RTCv3Defense(DefenseConfig(enabled=True,type='rtc_v3_candidate',custom_params=params),num_clients=10)
    base=_params()
    for rd,signs in [(1,[1]*10),(2,[1]*4+[-1]*6)]:
        d.set_context(rd,ids,base,principal_ids=ids,parameter_roles=ROLES,parameter_names=NAMES,trainable_parameter_indices=[0,1,2])
        updates=[([base[0]+np.ones(4,np.float32)*s*.02,base[1].copy(),base[2].copy()],10) for s in signs]
        base=d.aggregate(updates)
        assert d.last_round_metrics['rtc_v3_max_constraint_violation']<=1e-8
    rows=d._last_spectral_rows
    assert all(rows[i]['rtc_r1g_original_flagged'] and rows[i]['rtc_r1g_vetoed'] and not rows[i]['rtc_r1_flagged'] for i in range(4))
    assert all(rows[i]['rtc_r1_q']==1 for i in range(4))
    assert all(abs(rows[i]['rtc_r1g_reference_cosine']-rows[i]['rtc_r1h_reference_cosine'])<1e-12 for i in range(4))
