"""Synthetic arrays only; never imports a client trainer or loads a dataset."""
import copy
import json
import numpy as np
import pytest
from defenses.rtc.temporal_observe import observe, DIMENSION
from experiments import rtc_r3_temporal_observation as stage


def test_trainable_scope_scale_invariance_and_no_input_or_rng_mutation():
    x = [[np.arange(16, dtype=float), np.array([99999.])]]
    before = copy.deepcopy(x)
    state = np.random.get_state()
    a = observe(x, [0])[0]
    b = observe([[x[0][0] * 2, np.array([-1e8])]], [0])[0]
    assert a['rtc_r3t_trainable_dimension'] == 16
    assert a['rtc_r3t_residual_norm'] == np.linalg.norm(x[0][0])
    assert np.allclose(json.loads(b['rtc_r3t_sketch_json']), json.loads(a['rtc_r3t_sketch_json']))
    assert b['rtc_r3t_residual_norm'] == 2*a['rtc_r3t_residual_norm']
    assert all(np.array_equal(p, q) for p, q in zip(x[0], before[0]))
    after = np.random.get_state()
    assert state[0] == after[0] and np.array_equal(state[1], after[1]) and state[2:] == after[2:]
    with pytest.raises(ValueError):
        observe([[np.array([np.nan])]], [0])
    with pytest.raises(ValueError):
        observe(x, [])


def test_causal_coherence_distinguishes_persistence_and_handles_zero_visits():
    def records(signs):
        return [dict(round=str(i+1), cid='0', principal_id='0', attack_active='False',
            rtc_r3t_residual_norm=abs(s), rtc_r3t_trainable_dimension=1,
            rtc_r3t_sketch_json=json.dumps([s]+[0.]*(DIMENSION-1))) for i,s in enumerate(signs)]
    steady = stage.coherence(records([1.]*7))
    alternating = stage.coherence(records([1.,-1.,1.,-1.,1.]))
    assert not any(r['valid'] for r in steady[:4])
    assert steady[4]['coherence'] == 1. and alternating[-1]['coherence'] == pytest.approx(.2)
    assert not stage.coherence(records([1.,1.,0.,1.,1.]))[-1]['valid']
    assert stage.coherence(records([1.]*5)) == steady[:5]


def test_phase6_observation_does_not_change_models_weights_or_cumulative_state():
    from tests.test_rtc_v3_semantic_temporal_exposure import _manifest, _params, ROLES, NAMES
    from config.config_loader import DefenseConfig
    from defenses.rtc.v3 import RTCv3Defense
    from defenses.rtc.calibration import content_hash
    ids = list(map(str, range(10)))
    manifest = _manifest(frozen_cones=True)
    manifest['cumulative'] = {'enabled': True, 'resolutions': {'full': dict(
        scale_center=1., scale_lower=1., scale_upper=1., kappa=.5, threshold=.5, eta=2., q_min=.5)}}
    manifest['content_hash'] = content_hash(manifest)
    custom = dict(calibration_manifest=manifest, implementation_phase=6, parameter_roles=ROLES,
        principal_map={i:i for i in ids}, principal_first_sampling_verified=True,
        semantic_intervention_risk_floor=.5, cumulative_q_cap_power=1,
        anchor_recycle_fraction=.51, anchor_recycle_weighting='accepted',
        spectral_direction_mode='cap', spectral_direction_calibration=str(stage.prior.CAL),
        raw_norm_mode='cap', raw_norm_calibration=str(stage.prior.RAW_CAL))
    defenses = [RTCv3Defense(DefenseConfig(enabled=True, type='rtc_v3_candidate',
        custom_params={**custom, 'temporal_residual_observe_only': flag}), num_clients=10) for flag in (False,True)]
    base = _params()
    for r in range(1,5):
        outputs=[]
        for d in defenses:
            d.set_context(r,ids,base,principal_ids=ids,parameter_roles=ROLES,parameter_names=NAMES,
                          trainable_parameter_indices=[0,1,2])
            step=[.1]*7+[-.1,3.,-3.]
            updates=[([base[0]+np.ones(4,np.float32)*s,base[1].copy(),base[2].copy()],10) for s in step]
            outputs.append(d.aggregate(updates))
        assert all(np.array_equal(a,b) for a,b in zip(*outputs))
        assert defenses[0].last_client_aggregation_weights == defenses[1].last_client_aggregation_weights
        assert defenses[0]._cumulative.state_dict() == defenses[1]._cumulative.state_dict()
        assert not defenses[0]._last_temporal_rows and len(defenses[1]._last_temporal_rows)==10
        base=outputs[0]


def test_manual_matrix_and_incomplete_results_fail_closed(tmp_path, monkeypatch):
    assert sum(len(ds) for _,_,_,ds in stage.batches()) == 10
    for name,attack,seed,ds in stage.batches():
        argv=stage.command(tmp_path,name,attack,seed,ds)
        assert argv[-1]=='--dry-run' and '--smoke' not in argv
        assert stage.command(tmp_path,name,attack,seed,ds,True)==argv[:-1]
    monkeypatch.setattr(stage,'verify_lock',lambda root:{'cells':[dict(batch='b',run_id='r')]})
    monkeypatch.setattr(stage.prior,'ensure_no_training_process',lambda:None)
    monkeypatch.setattr(stage.subprocess,'run',lambda *a,**k:pytest.fail('Training must not start'))
    p=tmp_path/'b/status/r.json';p.parent.mkdir(parents=True)
    p.write_text(json.dumps(dict(state='failed',exit_code=1,last_round=7)))
    with pytest.raises(ValueError,match='Incomplete/failed'):
        stage.execute(tmp_path)
