"""R2 handoff/analyzer gates using synthetic files; never real training."""
import copy
import csv
import json
from pathlib import Path
import numpy as np
import pytest
from experiments import rtc_r2_raw_norm as stage
from defenses.rtc.spectral_direction import measure, diagnostics, load_calibration, calibration_hash


def write_json(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj), encoding='utf-8')


def write_csv(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


@pytest.fixture
def completed(tmp_path):
    calibration = load_calibration(stage.CALIBRATION)
    cells = [{'batch': name, 'attack': attack, 'defense': defense, 'run_id': defense, 'seed': 42,
              'attack_implementation_hash': 'synthetic-impl', 'attack_contract_hash': 'synthetic-contract'} for name, attack, ds in stage.BATCHES for defense in ds]
    raw_cal = stage.raw_norm.load_calibration(stage.RAW_CALIBRATION)
    lock = {'cells': cells, 'calibration': calibration, 'calibration_hash': calibration_hash(calibration),
            'raw_calibration': raw_cal, 'raw_calibration_hash': stage.raw_norm.calibration_hash(raw_cal)}
    ids = [str(i) for i in range(10)]
    for c in cells:
        batch, rid = tmp_path / c['batch'], c['run_id']
        data = {'synthetic': True}; data['sha256'] = stage.canonical(data)
        trial = {'rounds': [{'partition_ids': ids, 'fit_seed_digest': str(r)} for r in range(1, 61)]}
        trial['trial_plan_hash'] = stage.canonical(trial)
        c['trial_plan_hash'] = trial['trial_plan_hash']
        trial_path = batch / 'trial.json'; write_json(trial_path, trial)
        pair = dict(trial_plan_path=str(trial_path), trial_plan_hash=trial['trial_plan_hash'],
            data_manifest_sha256=data['sha256'], initial_model_sha256='synthetic-initial', malicious_identity_sha256='identity',
            malicious_partition_ids=[7, 8, 9], pairing_mode='strict', sampling_protocol='principal_uniform', deterministic_client_training=True)
        write_json(batch / 'status' / (rid + '.json'), dict(state='completed', exit_code=0, last_round=60))
        for suffix, value in [('_pairing_manifest', pair), ('_data_manifest', data), ('_config', {'synthetic': True})]:
            write_json(batch / 'raw' / (rid + suffix + '.json'), value)
        write_json(batch / 'resolved' / (rid + '.json'), {'synthetic': True})
        gates = stage.RUNNER_GATES | ({'attack_execution:' + x['run_id'] for x in cells if x['batch'] == c['batch']} if c['attack'] != 'none' else set())
        write_csv(batch / 'quality_gates.csv', [{'gate': g, 'passed': True} for g in sorted(gates)])
        rr = [dict(round=0, server_accuracy=.1, server_loss=2)]
        cc = []
        for r in range(1, 61):
            attacking = c['attack'] != 'none' and r >= 11
            mode = 'cap' if c['defense'] == stage.CAP else 'observe'
            x = np.eye(10) * .2
            x = np.column_stack([np.array([1] * 7 + ([-1] * 3 if attacking else [1] * 3)), x])
            if c['defense'] in (stage.MK, stage.RFA) and attacking:
                x[7:] *= 30
            delta = [[v] for v in x]
            measured = measure(delta, [0], calibration, 'observe')
            norms = np.linalg.norm(x,axis=1) * np.array([1.]*7 + ([30.]*3 if attacking else [1.]*3))
            if c['defense'] in (stage.MK, stage.RFA): norms = np.linalg.norm(x,axis=1)
            raw_rows = stage.raw_norm.score(norms, raw_cal, mode)
            for i, raw in enumerate(raw_rows): raw['rtc_r2_clip_factor'] = float(np.linalg.norm(x[i])/norms[i])
            weights = np.array([.1 * row['rtc_r2_q'] for row in raw_rows])
            if c['defense'] == stage.MK:
                weights = np.array([.2] * 5 + [0.] * 5)
            anchor = np.zeros(11)
            clients, metrics = diagnostics(measured, delta, [anchor], [weights @ x], weights, 0., ids, ['trainable'])
            row = {'round': r, 'server_accuracy': .82 if c['defense'] == stage.CAP and attacking else .8,
                'server_loss': 1., 'trial_plan_hash': c['trial_plan_hash'], 'attack_implementation_hash': c['attack_implementation_hash'],
                'attack_contract_hash': 'synthetic-contract', 'fit_completed_partition_ids_json': json.dumps(ids), 'fit_fit_seed_digest': str(r),
                'fit_received_updates_finite': True, 'fit_coordinated_updates_finite': True, 'fit_aggregate_parameters_finite': True,
                'fit_rtc_v3_semantic_intervention_risk_floor': .5, 'fit_rtc_v3_cumulative_q_cap_power': 1,
                'fit_rtc_v3_anchor_recycle_fraction': .51, 'fit_rtc_v3_norm_clip_mad_k': 2.5,
                'fit_rtc_v3_anchor_recycle_weighting': 'accepted', 'fit_rtc_v3_max_constraint_violation': 0.,
                'fit_aggregation_time_seconds': .1, 'fit_defense_random_seed': str(r), 'fit_aggregate_update_sketch_json': json.dumps((weights @ x).tolist()),
                **{'fit_' + k: v for k, v in metrics.items()},
                **{'fit_' + k:v for k,v in stage.raw_norm.summary(raw_cal, mode, .01).items()}}
            rr.append(row)
            for i, obs in enumerate(clients):
                cc.append(dict(round=r, cid=str(i), is_malicious=i >= 7, attack_active=attacking and i >= 7,
                    aggregation_weight=weights[i], principal_id=str(i), num_examples=10, local_epochs=5,
                    rtc_r2_existing_q=1., rtc_r2_applied=raw_rows[i]['rtc_r2_q']<1., rtc_r2_nominal_mass=.1,
                    **raw_rows[i], rtc_r1_existing_q=1., rtc_r1_applied=obs['rtc_r1_q'] < 1., rtc_r1_nominal_mass=.1, **obs))
        write_csv(batch / 'rounds' / (rid + '.csv'), rr)
        write_csv(batch / 'raw' / (rid + '_clients.csv'), cc)
    return tmp_path, lock


def test_complete_synthetic_results_accept_and_failure_gates_reject(completed):
    root, lock = completed
    runs = [stage.verify_cell(root, c, lock) for c in lock['cells']]
    decision = stage.decide(runs)
    assert decision['candidate_accepted'] and all(decision['checks'].values())
    assert decision['negative_proxy']['candidate'] == 0
    for attack, metric in [('gaussian_noise', 'gaussian_active_gain'), ('none', 'clean_active')]:
        changed = copy.deepcopy(runs)
        target = next(r for r in changed if r['cell']['attack'] == attack and r['cell']['defense'] == stage.CAP)
        for row in target['rounds'][1:]:
            row['server_accuracy'] = .7
        rejected = stage.decide(changed)
        assert not rejected['candidate_accepted'] and not rejected['checks'][metric]
    changed = copy.deepcopy(runs)
    target = next(r for r in changed if r['cell']['attack'] == 'none' and r['cell']['defense'] == stage.CAP)
    for row in target['clients'][:7]:
        row['rtc_r2_flagged'] = True
    assert not stage.decide(changed)['checks']['clean_benign_flags']


@pytest.mark.parametrize('problem', ['missing_round', 'failed', 'missing_gate', 'nonfinite', 'plan_drift', 'cap_violation', 'wrong_score', 'label_drift', 'raw_ratio_drift', 'raw_q_drift', 'budget'])
def test_corrupt_results_refused(completed, problem):
    root, lock = completed
    c = next(c for c in lock['cells'] if c['defense'] == stage.CAP)
    batch, rid = root / c['batch'], c['run_id']
    if problem == 'failed':
        write_json(batch / 'status' / (rid + '.json'), dict(state='failed', exit_code=1, last_round=60))
    elif problem == 'missing_gate':
        write_csv(batch / 'quality_gates.csv', [{'gate': 'unrelated', 'passed': True}])
    elif problem in ('cap_violation', 'wrong_score', 'label_drift', 'raw_ratio_drift', 'raw_q_drift'):
        path = batch / 'raw' / (rid + '_clients.csv')
        cc = stage.read_csv(path)
        row = next(r for r in cc if r['round'] == '11' and r['cid'] == '7')
        row[{'cap_violation': 'aggregation_weight', 'wrong_score': 'rtc_r1_cosine', 'label_drift': 'attack_active', 'raw_ratio_drift': 'rtc_r2_ratio', 'raw_q_drift': 'rtc_r2_q'}[problem]] = .1 if problem != 'label_drift' else False
        write_csv(path, cc)
    else:
        path = batch / 'rounds' / (rid + '.csv')
        rr = stage.read_csv(path)
        if problem == 'missing_round':
            rr.pop()
        else:
            key, value = {'nonfinite': ('server_accuracy', 'nan'), 'plan_drift': ('fit_fit_seed_digest', 'wrong'),
                          'budget': ('fit_rtc_v3_max_constraint_violation', .1)}[problem]
            rr[1][key] = value
        write_csv(path, rr)
    with pytest.raises(AssertionError):
        stage.verify_cell(root, c, lock)


def test_default_commands_are_exactly_six_dry_run_units():
    assert sum(len(ds) for _, _, ds in stage.BATCHES) == 6
    for name, attack, ds in stage.BATCHES:
        dry = stage.command(stage.OUTPUT, name, attack, ds)
        assert '--dry-run' in dry and '--smoke' not in dry
        assert dry[dry.index('--max-spec-retries') + 1] == '0'
        assert '--dry-run' not in stage.command(stage.OUTPUT, name, attack, ds, execute=True)


def test_launch_refuses_drift_or_failed_state_before_subprocess(tmp_path, monkeypatch):
    monkeypatch.setattr(stage.subprocess, 'run', lambda *a, **kw: pytest.fail('must not launch'))
    with pytest.raises(FileNotFoundError):
        stage.execute(tmp_path)
    write_json(tmp_path / 'r2_lock.json', {'sources': {}})
    with pytest.raises(ValueError, match='source contract'):
        stage.execute(tmp_path)
    monkeypatch.setattr(stage, 'verify_lock', lambda root: {'cells': [{'batch': 'b', 'run_id': 'r'}]})
    write_json(tmp_path / 'b/status/r.json', dict(state='failed', exit_code=1, last_round=15))
    with pytest.raises(ValueError, match='Incomplete'):
        stage.execute(tmp_path)


def test_analyzer_does_not_create_acceptance_with_missing_results(tmp_path, monkeypatch):
    monkeypatch.setattr(stage, 'verify_lock', lambda root: {'cells': [{'batch': 'b', 'run_id': 'r'}]})
    with pytest.raises(FileNotFoundError):
        stage.analyze(tmp_path)
    assert not (tmp_path / 'analysis').exists()
