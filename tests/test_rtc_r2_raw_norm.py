"""Synthetic model arrays only. No client training or dataset loading."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from defenses.rtc import raw_norm

ROOT = Path(__file__).resolve().parents[1]
CAL = ROOT/'config/rtc_r2_raw_norm_calibration.json'


@pytest.mark.parametrize('field,value', [('f', True), ('ratio_threshold', 2.9), ('ratio_threshold', float('inf')),
    ('ratio_threshold', True), ('cap_multiplier', .1), ('version', 'other')])
def test_invalid_calibration(tmp_path, field, value):
    c = json.loads(CAL.read_text()); c[field] = value
    p = tmp_path/'invalid.json'; p.write_text(json.dumps(c))
    with pytest.raises(ValueError): raw_norm.load_calibration(p)


def test_loo_threshold_abstention_and_trainable_scope():
    cal = raw_norm.load_calibration(CAL)
    rows = raw_norm.score([1.]*7+[30.]*3, cal, 'cap')
    assert [r['rtc_r2_q'] for r in rows] == [1.]*7+[0.]*3
    assert raw_norm.score([1.]*9+[3.], cal, 'cap')[-1]['rtc_r2_q'] == 1.
    assert raw_norm.score([1.]*9+[3.000001], cal, 'cap')[-1]['rtc_r2_q'] == 0.
    assert all(r['rtc_r2_q'] == 1 for r in raw_norm.score([1.]*7+[30.]*3, cal, 'observe'))
    assert all(not r['rtc_r2_valid'] for r in raw_norm.score([0.]*10, cal, 'cap'))
    assert all(not r['rtc_r2_valid'] for r in raw_norm.score([1.]*8+[30.], cal, 'cap'))
    assert raw_norm.trainable_norms([[np.array([3.,4.]), np.array([1e20])]], [0]) == [5.]
    for values in ([float('nan')]*10, [-1.]*10):
        with pytest.raises(ValueError): raw_norm.score(values, cal, 'cap')


def result(mode, defense_type='rtc_v3_candidate', labels=False):
    from config.config_loader import StrategyConfig, DefenseConfig
    from defenses.rtc.calibration import build_manifest
    from strategies.fed_strategy import FedSecStrategy
    from flwr.common import FitRes, Status, Code, ndarrays_to_parameters, parameters_to_ndarrays
    base = [np.zeros(12, np.float32), np.zeros(1, np.float32)]
    custom = {'spectral_direction_mode':'observe', 'spectral_direction_calibration':str(ROOT/'config/rtc_r1c_pairwise_calibration.json')}
    if defense_type == 'rtc_v3_candidate':
        custom.update(calibration_manifest=build_manifest(params=base,clip_lower=10,clip_upper=10),
                      implementation_phase=1,anchor_recycle_fraction=.51,anchor_recycle_weighting='accepted')
    if mode != 'off':
        custom.update(raw_norm_mode=mode,raw_norm_calibration=str(CAL))
    cfg = DefenseConfig(enabled=True,type=defense_type,custom_params=custom,krum_num_malicious=3,krum_num_to_select=5)
    s = FedSecStrategy(StrategyConfig(),cfg,base,num_clients=10,clients_per_round=10,
                       scalable_parameter_indices={0},parameter_names={'0':'body.weight','1':'bn.running_mean'})
    x = np.eye(10,12)*.2+np.ones((10,12))
    x[7:] *= 30
    fits = [(SimpleNamespace(cid=str(i)),FitRes(Status(Code.OK,''),ndarrays_to_parameters([
        x[i].astype(np.float32),np.array([1000*(-1)**i],np.float32)]),10,
        {'client_id':i,'is_malicious':labels,'attack_active':labels})) for i in range(10)]
    output,metrics = s.aggregate_fit(1,fits,[])
    return parameters_to_ndarrays(output),s,metrics


def test_raw_cap_preclip_solver_and_labels():
    off, obs, cap = [result(mode) for mode in ('off','observe','cap')]
    assert all(np.array_equal(a,b) for a,b in zip(off[0],obs[0]))
    assert repr(off[1].defense.state_dict()) == repr(obs[1].defense.state_dict())
    rows = cap[1].last_client_records
    assert [r['rtc_r2_applied'] for r in rows] == [False]*7+[True]*3
    assert all(r['aggregation_weight'] <= 1e-10 for r in rows[7:])
    assert all(r['rtc_r1_q'] == 1 and not r['rtc_r1_applied'] for r in rows)
    assert all(r['rtc_r2_raw_norm'] > 100 for r in rows[7:])
    assert all(r['rtc_r1_trainable_clipped_norm'] < 10 for r in rows)
    assert cap[2]['rtc_v3_max_constraint_violation'] <= 1e-8
    assert cap[2]['rtc_r1_reconstruction_error'] < 1e-5
    relabeled = result('cap',labels=True)
    assert all(np.array_equal(a,b) for a,b in zip(cap[0],relabeled[0]))
    d = cap[1].defense
    with pytest.raises(ValueError,match='trainable metadata'):
        d.set_context(2,list(map(str,range(10))),cap[0])
    d.set_context(2,list(map(str,range(10))),cap[0],trainable_parameter_indices=[0],validated_masses=[0.]*10)
    d.aggregate([(cap[0],0) for _ in range(10)])
    assert d._last_raw_norm_rows == []


@pytest.mark.parametrize('defense', ['krum','rfa'])
def test_challenger_observation_is_inert(defense):
    off,obs = [result(mode,defense) for mode in ('off','observe')]
    assert all(np.array_equal(a,b) for a,b in zip(off[0],obs[0]))
    assert len(obs[1].last_client_records) == 10
    assert all(r['rtc_r2_q'] == 1 for r in obs[1].last_client_records)
    assert obs[2]['rtc_r1_reconstruction_error']/max(obs[2]['rtc_r1_actual_aggregate_norm'],1e-8)<1e-4


def test_phase6_multiple_rounds_raw_only():
    from tests.test_rtc_v3_semantic_temporal_exposure import _manifest, _params, ROLES, NAMES
    from config.config_loader import DefenseConfig
    from defenses.rtc.v3 import RTCv3Defense
    from defenses.rtc.calibration import content_hash
    ids = [str(i) for i in range(10)]
    outcomes = []
    for mode in ('off', 'observe', 'cap'):
        manifest = _manifest(frozen_cones=True)
        manifest['cumulative'] = {'enabled': True, 'resolutions': {'full': {
            'scale_center': 1., 'scale_lower': 1., 'scale_upper': 1., 'kappa': .5,
            'threshold': .5, 'eta': 2., 'q_min': .5}}}
        manifest['content_hash'] = content_hash(manifest)
        custom = dict(calibration_manifest=manifest, implementation_phase=6,
            parameter_roles=ROLES, principal_map={i: i for i in ids}, principal_first_sampling_verified=True,
            semantic_intervention_risk_floor=.5, cumulative_q_cap_power=1,
            anchor_recycle_fraction=.51, anchor_recycle_weighting='accepted')
        if mode != 'off':
            custom.update(raw_norm_mode=mode, raw_norm_calibration=str(CAL))
        d = RTCv3Defense(DefenseConfig(enabled=True, type='rtc_v3_candidate', custom_params=custom), num_clients=10)
        base = _params()
        for r in range(1, 5):
            d.set_context(r, ids, base, principal_ids=ids, parameter_roles=ROLES, parameter_names=NAMES,
                          trainable_parameter_indices=[0, 1, 2])
            # Small body-only opposed updates leave classifier semantics neutral.
            updates = [([base[0] + np.ones(4, np.float32) * (.1 if i < 7 else 3.),
                         base[1].copy(), base[2].copy()], 10) for i in range(10)]
            base = d.aggregate(updates)
            assert d.last_round_metrics['rtc_v3_max_constraint_violation'] <= 1e-8
            if mode == 'cap':
                assert all(d.last_client_aggregation_weights[i] <= 1e-10 for i in ids[7:])
        outcomes.append((base, d.state_dict()))
    assert all(np.array_equal(a, b) for a, b in zip(outcomes[0][0], outcomes[1][0]))
    assert repr(outcomes[0][1]) == repr(outcomes[1][1])
    assert not np.array_equal(outcomes[1][0][0], outcomes[2][0][0])
