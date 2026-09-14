"""Synthetic arrays/artifacts only: no data loading or client training."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

from experiments import rtc_i12_bridge as stage
from experiments import rtc_i12_validation as validation
from tests.test_rtc_r2_stage import write_json, write_csv


def test_commands_and_modes():
    assert sum(len(ds) for _, _, _, ds in stage.batches()) == 40
    for name, attack, seed, ds in stage.batches():
        argv = stage.command(stage.OUTPUT, name, attack, seed, ds)
        assert '--dry-run' in argv and '--execute' not in argv and '--smoke' not in argv
        assert argv[argv.index('--seeds') + 1] == str(seed)
        assert stage.command(stage.OUTPUT, name, attack, seed, ds, execute=True) == argv[:-1]
    assert stage.MODES[stage.COMBINED] == ('cap', 'cap')
    assert stage.MODES[stage.DIRECTION] == ('cap', 'observe')


def aggregate(mode, labels=False):
    from config.config_loader import DefenseConfig, StrategyConfig
    from defenses.rtc.calibration import build_manifest
    from strategies.fed_strategy import FedSecStrategy
    from flwr.common import FitRes, Status, Code, ndarrays_to_parameters, parameters_to_ndarrays
    base = [np.zeros(12, np.float32), np.zeros(1, np.float32)]
    dm, nm = stage.MODES[mode]
    custom = dict(calibration_manifest=build_manifest(params=base, clip_lower=10, clip_upper=10),
                  implementation_phase=1, anchor_recycle_fraction=.51, anchor_recycle_weighting='accepted',
                  spectral_direction_mode=dm, spectral_direction_calibration=str(stage.CAL),
                  raw_norm_mode=nm, raw_norm_calibration=str(stage.RAW_CAL))
    s = FedSecStrategy(StrategyConfig(), DefenseConfig(enabled=True, type='rtc_v3_candidate', custom_params=custom),
                       base, num_clients=10, clients_per_round=10, scalable_parameter_indices={0},
                       parameter_names={'0': 'body.weight', '1': 'bn.running_mean'})
    x = np.eye(10, 12) * .2 + np.ones((10, 12))
    x[7] *= -1; x[8] *= 30; x[9] *= -30
    fits = [(SimpleNamespace(cid=str(i)), FitRes(Status(Code.OK, ''), ndarrays_to_parameters([
        x[i].astype(np.float32), np.array([1000 * (-1)**i], np.float32)]), 10,
        {'client_id': i, 'is_malicious': labels, 'attack_active': labels})) for i in range(10)]
    result, metrics = s.aggregate_fit(1, fits, [])
    return parameters_to_ndarrays(result), s.last_client_records, metrics


def test_joint_cap_applies_union_and_is_label_independent():
    results = {mode: aggregate(mode) for mode in stage.RTC}
    for mode, expected_zero in [(stage.DIRECTION, (7, 9)), (stage.NORM, (8, 9)), (stage.COMBINED, (7, 8, 9))]:
        _, clients, metrics = results[mode]
        for i in expected_zero:
            assert clients[i]['aggregation_weight'] <= 1e-8
        assert all(clients[i]['aggregation_weight'] > .09 for i in range(7))
        assert metrics['rtc_v3_max_constraint_violation'] <= 1e-8
        assert metrics['rtc_r1_reconstruction_error'] / max(metrics['rtc_r1_actual_aggregate_norm'], 1e-8) < 1e-4
    labeled, _, _ = aggregate(stage.COMBINED, True)
    assert all(np.array_equal(a, b) for a, b in zip(labeled, results[stage.COMBINED][0]))


def test_phase6_joint_caps_across_rounds():
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
        principal_map={i: i for i in ids}, principal_first_sampling_verified=True,
        semantic_intervention_risk_floor=.5, cumulative_q_cap_power=1,
        anchor_recycle_fraction=.51, anchor_recycle_weighting='accepted',
        spectral_direction_mode='cap', spectral_direction_calibration=str(stage.CAL),
        raw_norm_mode='cap', raw_norm_calibration=str(stage.RAW_CAL))
    d = RTCv3Defense(DefenseConfig(enabled=True, type='rtc_v3_candidate', custom_params=custom), num_clients=10)
    base = _params()
    for r in range(1, 5):
        d.set_context(r, ids, base, principal_ids=ids, parameter_roles=ROLES, parameter_names=NAMES,
                      trainable_parameter_indices=[0, 1, 2])
        step = [.1]*7 + [-.1, 3., -3.]
        updates = [([base[0]+np.ones(4, np.float32)*step[i], base[1].copy(), base[2].copy()], 10) for i in range(10)]
        base = d.aggregate(updates)
        assert d.last_round_metrics['rtc_v3_max_constraint_violation'] <= 1e-8
        assert all(d.last_client_aggregation_weights[i] <= 1e-8 for i in ids[7:])
        assert all(d.last_client_aggregation_weights[i] > .09 for i in ids[:7])


@pytest.fixture
def combined_artifacts(tmp_path):
    from defenses.rtc.spectral_direction import measure, diagnostics, calibration_hash, load_calibration
    from defenses.rtc import raw_norm
    cal, raw = load_calibration(stage.CAL), raw_norm.load_calibration(stage.RAW_CAL)
    c = dict(batch='synthetic', attack='gaussian_noise', seed=44, defense=stage.COMBINED, run_id='combined',
             attack_implementation_hash='synthetic', attack_contract_hash='synthetic')
    lock = dict(cells=[c], calibration=cal, raw_calibration=raw, calibration_hash=calibration_hash(cal),
                raw_calibration_hash=raw_norm.calibration_hash(raw))
    batch, rid = tmp_path / c['batch'], c['run_id']
    ids = list(map(str, range(10)))
    data = {'synthetic': True}; data['sha256'] = stage.canonical(data)
    trial = {'rounds': [dict(partition_ids=ids, fit_seed_digest=str(r)) for r in range(1, 61)]}
    trial['trial_plan_hash'] = stage.canonical(trial); c['trial_plan_hash'] = trial['trial_plan_hash']
    write_json(batch / 'trial.json', trial)
    pair = dict(trial_plan_path=str(batch / 'trial.json'), trial_plan_hash=trial['trial_plan_hash'],
                data_manifest_sha256=data['sha256'], initial_model_sha256='initial', malicious_identity_sha256='identity',
                malicious_partition_ids=[7, 8, 9], pairing_mode='strict', sampling_protocol='principal_uniform', deterministic_client_training=True)
    cumulative = dict(scale_lower=1, scale_upper=1, kappa=1, threshold=1, eta=1, q_min=.5)
    write_json(batch / 'cal.json', {'cumulative': {'resolutions': {'full': cumulative}}})
    cfg = {'security': {'defense': {'custom_params': {'calibration_path': str(batch / 'cal.json')}}}}
    for suffix, value in [('_pairing_manifest', pair), ('_data_manifest', data), ('_config', cfg)]:
        write_json(batch / 'raw' / (rid + suffix + '.json'), value)
    write_json(batch / 'resolved' / (rid + '.json'), cfg)
    write_json(batch / 'status' / (rid + '.json'), dict(state='completed', exit_code=0, last_round=60))
    write_csv(batch / 'quality_gates.csv', [dict(gate=g, passed=True) for g in validation.RUNNER_GATES | {'attack_execution:combined'}])
    rr = [dict(round=0, server_accuracy=.1, server_loss=2)]; cc = []
    for r in range(1, 61):
        active = r >= 11
        x = np.column_stack([np.array([1]*7 + ([-1, 1, -1] if active else [1]*3)), np.eye(10)*.2])
        raw_norms = np.linalg.norm(x, axis=1) * np.array([1]*8 + ([30]*2 if active else [1]*2))
        measured = measure([[v] for v in x], [0], cal, 'cap')
        raw_rows = raw_norm.score(raw_norms, raw, 'cap')
        direction_q = [v['q'] for v in measured['rows']]
        weights = np.array([.1 * min(q, n['rtc_r2_q']) for q, n in zip(direction_q, raw_rows)])
        amass = .51 * (1 - sum(weights)); zero = .49 * (1 - sum(weights))
        anchor = np.ones(x.shape[1]) * .1
        clients, geometry = diagnostics(measured, [[v] for v in x], [anchor], [weights @ x + amass * anchor], weights, amass, ids, ['trainable'])
        rr.append(dict(round=r, server_accuracy=.82, server_loss=1, trial_plan_hash=c['trial_plan_hash'],
            attack_implementation_hash='synthetic', attack_contract_hash='synthetic',
            fit_completed_partition_ids_json=json.dumps(ids), fit_fit_seed_digest=str(r), fit_defense_random_seed=str(r),
            fit_received_updates_finite=True, fit_coordinated_updates_finite=True, fit_aggregate_parameters_finite=True,
            fit_rtc_v3_semantic_intervention_risk_floor=.5, fit_rtc_v3_cumulative_q_cap_power=1,
            fit_rtc_v3_anchor_recycle_fraction=.51, fit_rtc_v3_norm_clip_mad_k=2.5,
            fit_rtc_v3_anchor_recycle_weighting='accepted', fit_rtc_v3_max_constraint_violation=0,
            fit_rtc_v3_anchor_recycle_mass=amass, fit_rtc_v3_zero_update_mass=zero,
            fit_rtc_v3_effective_update_mass=sum(weights)+amass, fit_aggregation_time_seconds=.1,
            **{'fit_'+k: v for k, v in geometry.items()},
            **{'fit_'+k: v for k, v in raw_norm.summary(raw, 'cap', .01).items()}))
        for i, obs in enumerate(clients):
            cc.append(dict(round=r, cid=str(i), principal_id=str(i), is_malicious=i >= 7,
                attack_active=active and i >= 7, aggregation_weight=weights[i], num_examples=10, local_epochs=5,
                nominal_mass=.1, rtc_v3_residual_norm=0., rtc_v3_cumulative_q_full=1.,
                rtc_r1_existing_q=1., rtc_r1_applied=direction_q[i] < 1., rtc_r1_nominal_mass=.1,
                rtc_r2_existing_q=direction_q[i], rtc_r2_applied=raw_rows[i]['rtc_r2_q'] < direction_q[i],
                rtc_r2_nominal_mass=.1, rtc_r2_clip_factor=float(np.linalg.norm(x[i]) / raw_norms[i]),
                **obs, **raw_rows[i]))
    write_csv(batch / 'rounds' / (rid + '.csv'), rr)
    write_csv(batch / 'raw' / (rid + '_clients.csv'), cc)
    return tmp_path, c, lock


def test_combined_artifacts_pass(combined_artifacts):
    root, cell, lock = combined_artifacts
    assert len(validation.verify_cell(root, cell, lock)['clients']) == 600


@pytest.mark.parametrize('field,value', [('rtc_r1_q', 1), ('rtc_r2_existing_q', 1),
    ('aggregation_weight', .1), ('rtc_v3_cumulative_q_full', .5), ('rtc_r2_ratio', 999)])
def test_invalid_joint_evidence_rejected(combined_artifacts, field, value):
    root, c, lock = combined_artifacts
    path = root / c['batch'] / 'raw' / (c['run_id'] + '_clients.csv')
    rows = validation.read_csv(path)
    next(r for r in rows if r['round'] == '11' and r['cid'] == '9')[field] = value
    write_csv(path, rows)
    with pytest.raises(AssertionError):
        validation.verify_cell(root, c, lock)


def synthetic_decision_runs():
    runs = []
    for name, attack, seed, ds in stage.batches():
        for d in ds:
            gain = .02 if attack == 'sign_flip' and d in (stage.DIRECTION, stage.COMBINED) else 0
            weight = .2 if d in (stage.BASE, stage.DIRECTION) else .02
            runs.append({'cell': dict(batch=name, attack=attack, seed=seed, defense=d), 'rounds': [dict(round=0)],
                         'computed': dict(attack=attack, seed=seed, defense=d, active_accuracy=.8+gain,
                                          final_accuracy=.85+gain, benign_union_flag_rate=0., malicious_weight_per_round=weight)})
    return runs


def test_increment_regression_and_standalone_status_are_distinct(monkeypatch):
    monkeypatch.setattr(stage, 'metrics', lambda r: r['computed'])
    monkeypatch.setattr(stage, 'paired', lambda a, b: True)
    runs = synthetic_decision_runs(); protocol = stage.read_json(stage.PROTOCOL)
    assert stage.decide(runs, protocol)['candidate_accepted']
    changed = copy.deepcopy(runs)
    for r in changed:
        if r['cell']['defense'] == stage.NORM:
            r['computed']['active_accuracy'] = .5
    result = stage.decide(changed, protocol)
    assert result['candidate_accepted'] and not result['standalone_norm_screen_accepted']
    for attack in ('none', 'sign_flip', 'gaussian_noise', 'lie'):
        changed = copy.deepcopy(runs)
        next(r for r in changed if r['cell']['attack'] == attack and r['cell']['defense'] == stage.COMBINED)['computed']['active_accuracy'] = .1
        assert not stage.decide(changed, protocol)['candidate_accepted']
    with pytest.raises(AssertionError, match='All preregistered'):
        stage.decide(runs[:-1], protocol)


def test_execute_rejects_incomplete_before_subprocess(tmp_path, monkeypatch):
    monkeypatch.setattr(stage, 'verify_lock', lambda root: {'cells': [dict(batch='b', run_id='r')]})
    monkeypatch.setattr(stage, 'ensure_no_training_process', lambda: None)
    monkeypatch.setattr(stage.subprocess, 'run', lambda *a, **kw: pytest.fail('must not start training'))
    write_json(tmp_path / 'b/status/r.json', dict(state='failed', exit_code=1, last_round=12))
    with pytest.raises(ValueError, match='Incomplete/failed'):
        stage.execute(tmp_path)
