"""Manual R0b geometric-reference observation: no cap; default is dry-run."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUTPUT = ROOT / 'logs/rtc_r0b_geometric_observation'
BASELINE = ROOT / 'logs/rtc_r0_direction_observation'
REVIEW = ROOT / 'analysis/rtc_r0_direction_review'
BATCHES = (('clean_calibration_seed101', 'none', 101), ('sign_flip_diagnostic_seed42', 'sign_flip', 42))
DEFENSE = 'rtc_r0b_geometric_observe'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def truth(value):
    return str(value).strip().lower() in ('true', '1', '1.0')


def snapshot():
    files = [ROOT / 'main.py']
    for directory in ('attacks', 'client', 'config', 'data', 'datasets', 'defenses', 'experiments', 'models', 'server', 'strategies', 'utils'):
        folder = ROOT / directory
        if folder.exists():
            files += [p for p in folder.rglob('*') if p.suffix in ('.py', '.ps1', '.json', '.yaml', '.yml')]
    return {str(p.relative_to(ROOT)).replace('\\', '/'): digest(p) for p in sorted(set(files))}


def command(root, name, attack, seed, *, execute=False):
    args = [sys.executable, '-m', 'experiments.core.run', '--profile', 'rtc-byzantine',
            '--attacks', attack, '--defenses', DEFENSE, '--seeds', str(seed),
            '--byzantine-attack-freeze', str(root / 'attack.freeze.json'),
            '--attack-start-round', '11', '--attack-end-round', '-1', '--malicious-fractions', '.3',
            '--rounds', '60', '--num-clients', '20', '--participation-rate', '.5', '--batch-size', '48',
            '--partition', 'iid', '--skip-clean', '--max-spec-retries', '0',
            '--ray-client-num-cpus', '1', '--ray-client-num-gpus', '.25',
            '--ray-object-store-memory-mb', '3072', '--ray-min-available-memory-mb', '10240',
            '--ray-memory-wait-seconds', '120', '--output', str(root / name)]
    if not execute:
        args.append('--dry-run')
    return args


def verify_lock(root):
    lock = read_json(root / 'r0b_lock.json')
    current = snapshot()
    if current != lock['sources']:
        changed = sorted(k for k in set(current) | set(lock['sources']) if current.get(k) != lock['sources'].get(k))
        raise ValueError(f'R0b source contract changed; do not overwrite old outputs: {changed}')
    for relative, expected in lock['artifacts'].items():
        if digest(root / relative) != expected:
            raise ValueError(f'R0b artifact drift: {relative}')
    for relative, expected in lock['baseline_evidence'].items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f'R0 baseline evidence drift: {relative}')
    return lock


def prepare(root=OUTPUT):
    from attacks.spec import attack_source_hash
    from experiments import periodic_attack
    if (root / 'r0b_lock.json').exists():
        lock = verify_lock(root)
        print('Existing dry-run lock verified; no manifests rewritten.')
        return lock
    if root.exists() and any(root.glob('*/status/*.json')):
        raise ValueError('Refuse to prepare over existing training outputs')
    prior_review = read_json(REVIEW / 'review.json')
    assert prior_review['observation_quality_accepted'] and not prior_review['mk_reference_cap_accepted']
    prior_lock = read_json(BASELINE / 'r0_lock.json')
    baseline_evidence = dict(prior_review['evidence'])
    for path in (BASELINE / 'r0_lock.json', REVIEW / 'review.json'):
        baseline_evidence[str(path.relative_to(ROOT)).replace('\\', '/')] = digest(path)
    for relative, expected in baseline_evidence.items():
        assert digest(ROOT / relative) == expected, relative
    bridge = verify_attack_source_bridge()
    root.mkdir(parents=True, exist_ok=True)
    freeze = {'schema_version': 'RTCByzantineAttackFreezeV1',
              'implementation_source_sha256': attack_source_hash(),
              'selection_policy': 'R0 observation only; fixed Sign-flip trainable v2 scale1, no strength promotion',
              'attacks': {'sign_flip': {'strength_level': 'weak', 'parameters': {'sign_flip_scale': 1.0}}}}
    (root / 'attack.freeze.json').write_text(json.dumps(freeze, indent=2), encoding='utf-8')
    cells, artifacts = [], {'attack.freeze.json': digest(root / 'attack.freeze.json')}
    for name, attack, seed in BATCHES:
        subprocess.run(command(root, name, attack, seed), cwd=ROOT, check=True)
        manifest_path = root / name / 'experiment_manifest.json'
        specs = read_json(manifest_path)['specs']
        assert len(specs) == 1, 'R0 must have one observation cell per batch'
        spec = specs[0]
        assert spec['defense'] == DEFENSE and spec['attack'] == attack and spec['seed'] == seed
        assert spec['custom_params']['direction_observe_only'] is True
        assert spec['custom_params']['direction_observe_reference'] == 'geometric_median'
        assert spec['custom_params']['anchor_recycle_fraction'] == .51
        from experiments.run import parse_args
        args = parse_args(command(root, name, attack, seed)[3:])
        cfg = periodic_attack._build_spec_config(spec, args, root / name)
        assert cfg.client.local_epochs > 0 and cfg.security.defense.custom_params['direction_observe_only'] is True
        (root / name / 'resolved_config.json').write_text(json.dumps(asdict(cfg), indent=2), encoding='utf-8')
        cells.append({'batch': name, 'attack': attack, 'seed': seed, 'run_id': periodic_attack.run_id(spec),
                      'trial_plan_hash': spec['trial_plan_hash'], 'attack_implementation_hash': spec['attack_implementation_hash']})
        for path in (root / name).rglob('*'):
            if path.is_file() and path.suffix in ('.json', '.csv'):
                artifacts[str(path.relative_to(root)).replace('\\', '/')] = digest(path)
    lock = {'schema': 'RTCR0bObservationLockV1', 'stage': 'R0b', 'training_units': 2,
            'baseline': 'B3R-F0.51', 'effect': 'geometric-reference observation only, no new defense cap',
            'baseline_evidence': baseline_evidence, 'baseline_cells': prior_lock['cells'],
            'attack_source_bridge': bridge,
            'reference_contract': {'name': 'smoothed_geometric_median', 'max_iterations': 200,
                'relative_tolerance': 1e-7, 'relative_smoothing': 1e-6, 'relative_min_norm': 1e-6, 'uniform_input_weights': True,
                'primary_diagnostic': 'leave_one_out_cosine', 'threshold': 'min(0, clean lower 1% order statistic)'},
            'calibration_policy': 'clean seed101 only; sign_flip seed42 diagnostic only; no joint threshold tuning',
            'gates': {'rounds_per_cell': 61, 'client_rows_per_cell': 600,
                      'reference_valid_fraction_min': .95, 'reconstruction_relative_error_max': 1e-4, 'loo_valid_fraction_min': .95,
                      'baseline_accuracy_abs_delta_max': 1e-12, 'baseline_weight_abs_delta_max': 1e-12,
                      'baseline_sketch_abs_delta_max': 1e-10, 'sign_benign_flag_rate_max': .01,
                      'clean_flag_rate_max': .01, 'sign_attacker_flag_rate_min': .5},
            'cells': cells, 'sources': snapshot(), 'artifacts': artifacts}
    (root / 'r0b_lock.json').write_text(json.dumps(lock, indent=2), encoding='utf-8')
    print(json.dumps({'stage': 'R0b', 'training_units': 2, 'cells': cells}, indent=2))
    return lock


def execute(root=OUTPUT):
    verify_lock(root)  # Crucially, never generate/update a lock during execution.
    for name, attack, seed in BATCHES:
        verify_lock(root)
        statuses = list((root / name / 'status').glob('*.json'))
        if any(read_json(p).get('state') not in ('completed', 'completed_cached') for p in statuses):
            raise ValueError('Existing incomplete/failed status requires manual review; no automatic restart')
        subprocess.run(command(root, name, attack, seed, execute=True), cwd=ROOT, check=True)


def verify_attack_source_bridge():
    """Prove why the attack-source hash changes although the attack does not.

    This permits a diagnostic trajectory check only, not reuse as a future
    algorithm efficacy baseline. Every attack source except the observer call
    is byte-identical to the archived R0 contract.
    """
    paths = ('attacks/spec.py', 'attacks/attack_client.py', 'attacks/coordinator.py',
             'client/fl_client.py', 'strategies/fed_strategy.py', 'server/fl_server.py', 'utils/numerical_failure.py')
    insertion = '                reference_mode=(self.defense.cfg.custom_params or {}).get("direction_observe_reference", "multi_krum"),\n'
    with zipfile.ZipFile(REVIEW / 'r0_sources.zip') as archive:
        for relative in paths:
            before = archive.read(relative)
            after = (ROOT / relative).read_bytes()
            if relative == 'strategies/fed_strategy.py':
                new_text = after.decode().replace('\r\n', '\n')
                assert new_text.count(insertion) == 1
                assert new_text.replace(insertion, '') == before.decode().replace('\r\n', '\n')
            else:
                assert before == after, f'Unreviewed attack source change: {relative}'
    return {'verified': True, 'only_attack_source_change': 'strategy passes observation reference_mode after aggregation',
            'usage': 'R0/R0b observation invariance check only; not an efficacy baseline for R1'}


def verify_baseline_trajectory(batch, rid, rounds, baseline_batch, baseline_rid, gates):
    """Compare observation-only trajectories; not evidence of cap efficacy."""
    baseline_rounds = {r['round']: r for r in read_csv(baseline_batch / 'rounds' / (baseline_rid + '.csv'))}
    old_pair = read_json(baseline_batch / 'raw' / (baseline_rid + '_pairing_manifest.json'))
    pair = read_json(batch / 'raw' / (rid + '_pairing_manifest.json'))
    for field in ('trial_plan_hash', 'data_manifest_sha256', 'initial_model_sha256',
                  'malicious_identity_sha256', 'malicious_partition_ids', 'deterministic_client_training',
                  'pairing_mode', 'sampling_protocol'):
        assert pair[field] == old_pair[field], f'baseline pairing: {field}'
    accuracy_delta = sketch_delta = weight_delta = 0.0
    for r in rounds:
        old = baseline_rounds[r['round']]
        accuracy_delta = max(accuracy_delta, abs(float(r['server_accuracy']) - float(old['server_accuracy'])))
        if int(r['round']) == 0:
            continue
        for field in ('trial_plan_hash', 'attack_contract_hash',
                      'fit_completed_partition_ids_json', 'fit_fit_seed_digest', 'fit_defense_random_seed'):
            assert r[field] == old[field], f'baseline runtime: {field}'
        current_sketch, old_sketch = json.loads(r['fit_aggregate_update_sketch_json']), json.loads(old['fit_aggregate_update_sketch_json'])
        assert len(current_sketch) == len(old_sketch)
        sketch_delta = max(sketch_delta, max(abs(float(a) - float(b)) for a, b in zip(current_sketch, old_sketch)))
    old_clients = {(c['round'], c['cid']): c for c in read_csv(baseline_batch / 'raw' / (baseline_rid + '_clients.csv'))}
    current_clients = read_csv(batch / 'raw' / (rid + '_clients.csv'))
    assert len(current_clients) == len(old_clients) == 600
    for c in current_clients:
        old = old_clients[(c['round'], c['cid'])]
        for field in ('is_malicious', 'attack_active', 'principal_id', 'num_examples', 'local_epochs'):
            assert c[field] == old[field], field
        weight_delta = max(weight_delta, abs(float(c['aggregation_weight']) - float(old['aggregation_weight'])))
    assert accuracy_delta <= gates['baseline_accuracy_abs_delta_max'], 'observation changed ACC trajectory'
    assert sketch_delta <= gates['baseline_sketch_abs_delta_max'], 'observation changed update sketch'
    assert weight_delta <= gates['baseline_weight_abs_delta_max'], 'observation changed RTC weight'
    return {'accuracy_max_abs_delta': accuracy_delta, 'sketch_max_abs_delta': sketch_delta,
            'weight_max_abs_delta': weight_delta, 'baseline_run_id': baseline_rid,
            'comparison_scope': 'observation invariance under the frozen reviewed source bridge'}


def geometry_audit(rounds, clients):
    import numpy as np
    from defenses.rtc.geometric_reference import geometric_reference, RELATIVE_MIN_NORM
    for row in rounds:
        if int(row['round']) == 0:
            continue
        ids = json.loads(row['fit_rtc_r0_geometry_client_ids_json'])
        gram = np.asarray(json.loads(row['fit_rtc_r0_gram_json']), dtype=float)
        assert len(ids) == len(set(ids)) == 10 and gram.shape == (10, 10)
        assert np.isfinite(gram).all() and np.allclose(gram, gram.T, atol=1e-10)
        assert np.linalg.eigvalsh(gram).min() >= -1e-8 * max(1, np.diag(gram).max())
        alpha, info = geometric_reference(gram)
        assert truth(row['fit_rtc_r0_reference_converged']) == info['converged']
        assert row['fit_rtc_r0_reference_mode'] == 'geometric_median'
        assert np.allclose(alpha, json.loads(row['fit_rtc_r0_reference_weights_json']), atol=1e-10, rtol=0)
        cs = {c['cid']: c for c in clients if c['round'] == row['round']}
        assert set(cs) == set(ids)
        for i, cid in enumerate(ids):
            c = cs[cid]
            assert abs(float(c['rtc_r0_trainable_clipped_norm']) ** 2 - gram[i, i]) <= 1e-8 * max(1, gram[i, i])
            loo, convergence = geometric_reference(gram, omit=i)
            norm_sq = max(0, loo @ gram @ loo)
            valid = convergence['converged'] and norm_sq > max(1e-12, RELATIVE_MIN_NORM * convergence['scale']) ** 2 and gram[i, i] > 1e-24
            value = c.get('rtc_r0_loo_cosine', '')
            assert bool(value != '') == bool(valid)
            if valid:
                expected = float(np.clip((gram @ loo)[i] / np.sqrt(gram[i, i] * norm_sq), -1, 1))
                assert abs(float(value) - expected) < 1e-8


def screen_reference(clients, threshold, gates):
    """Pre-registered descriptive screening; does not accept an unrun cap."""
    results = []
    for condition, attacker in (('none', False), ('sign_flip', False), ('sign_flip', True)):
        selected = [c for c in clients if c['condition'] == condition and
                    int(c['round']) >= (1 if condition == 'none' else 11) and truth(c['attack_active']) == attacker]
        assert selected
        flagged = sum(c.get('rtc_r0_loo_cosine', '') != '' and float(c['rtc_r0_loo_cosine']) < threshold for c in selected)
        results.append({'condition': condition, 'active_attacker': attacker, 'rows': len(selected),
                        'flagged': flagged, 'flag_rate': flagged / len(selected),
                        'abstentions': sum(c.get('rtc_r0_loo_cosine', '') == '' for c in selected)})
    checks = {'clean_flag_rate': results[0]['flag_rate'] <= gates['clean_flag_rate_max'],
              'sign_benign_flag_rate': results[1]['flag_rate'] <= gates['sign_benign_flag_rate_max'],
              'sign_attacker_flag_rate': results[2]['flag_rate'] >= gates['sign_attacker_flag_rate_min']}
    return {'threshold': threshold, 'groups': results, 'checks': checks,
            'eligible_for_r1_design_review': all(checks.values()), 'cap_accepted': False}


def analyze(root=OUTPUT):
    lock = verify_lock(root)
    assert verify_attack_source_bridge() == lock['attack_source_bridge']
    reports, clean_cosines, group_reports, all_clients = [], [], [], []
    for cell in lock['cells']:
        batch, rid = root / cell['batch'], cell['run_id']
        baseline_cell = next(c for c in lock['baseline_cells'] if c['attack'] == cell['attack'] and c['seed'] == cell['seed'])
        baseline_batch = BASELINE / baseline_cell['batch']
        baseline_rid = baseline_cell['run_id']
        assert cell['trial_plan_hash'] == baseline_cell['trial_plan_hash']
        assert read_json(batch / 'resolved_config.json') == read_json(batch / 'raw' / (rid + '_config.json'))
        status = read_json(batch / 'status' / (rid + '.json'))
        assert status['state'] in ('completed', 'completed_cached') and status['exit_code'] == 0 and status['last_round'] == 60
        rounds = read_csv(batch / 'rounds' / (rid + '.csv'))
        assert len(rounds) == 61 and {int(r['round']) for r in rounds} == set(range(61))
        fit = [r for r in rounds if int(r['round']) > 0]
        paired = verify_baseline_trajectory(batch, rid, rounds, baseline_batch, baseline_rid, lock['gates'])
        active = [r for r in fit if int(r['round']) >= (1 if cell['attack'] == 'none' else 11)]
        gates = read_csv(batch / 'quality_gates.csv')
        assert gates and all(truth(g['passed']) for g in gates), 'Runner quality gate failure'
        for field in ('trial_plan_hash', 'attack_implementation_hash'):
            assert {r[field] for r in fit} == {cell[field]}, field
        for field in ('server_accuracy', 'server_loss'):
            assert all(math.isfinite(float(r[field])) for r in rounds), field
        for field in ('fit_received_updates_finite', 'fit_coordinated_updates_finite', 'fit_aggregate_parameters_finite'):
            assert all(truth(r[field]) for r in fit), field
        assert all(truth(r['fit_rtc_r0_observe_only']) for r in fit)
        assert {r['fit_rtc_r0_version'] for r in fit} == {'rtc_r0b_geometric_geometry.v1'}
        for field, expected in {'fit_rtc_v3_semantic_intervention_risk_floor': .5, 'fit_rtc_v3_cumulative_q_cap_power': 1,
                                'fit_rtc_v3_anchor_recycle_fraction': .51, 'fit_rtc_v3_norm_clip_mad_k': 2.5}.items():
            assert all(float(r[field]) == expected for r in fit), field
        assert {r['fit_rtc_v3_anchor_recycle_weighting'] for r in fit} == {'accepted'}
        assert all(float(r['fit_rtc_v3_max_constraint_violation']) <= 1e-8 for r in fit), 'hard budget violation'
        valid = sum(truth(r['fit_rtc_r0_reference_valid']) for r in fit) / 60
        reconstruction = max(float(r['fit_rtc_r0_reconstruction_error']) / max(float(r['fit_rtc_r0_actual_aggregate_norm']), 1e-8) for r in fit)
        assert valid >= lock['gates']['reference_valid_fraction_min'], 'Reference availability insufficient'
        assert reconstruction <= lock['gates']['reconstruction_relative_error_max'], 'Reconstruction failed'
        clients = [c for c in read_csv(batch / 'raw' / (rid + '_clients.csv')) if 1 <= int(c['round']) <= 60]
        assert len(clients) == 600 and len({(c['round'], c['cid']) for c in clients}) == 600
        all_clients.extend({**c, 'condition': cell['attack']} for c in clients)
        assert {int(c['round']) for c in clients} == set(range(1, 61))
        for number in range(1, 61):
            assert sum(int(c['round']) == number for c in clients) == 10
        for c in clients:
            assert c['rtc_r0_version'] == 'rtc_r0b_geometric_geometry.v1'
            layers = json.loads(c['rtc_r0_layers_json'])
            assert layers and not any(name.endswith(('running_mean', 'running_var', 'num_batches_tracked')) for name in layers)
            for field in ('rtc_r0_trainable_raw_norm', 'rtc_r0_trainable_clipped_norm', 'rtc_r0_clip_factor'):
                assert math.isfinite(float(c[field])) and float(c[field]) >= 0
            if truth(c['rtc_r0_reference_valid']) and c['rtc_r0_cosine']:
                assert -1.0000001 <= float(c['rtc_r0_cosine']) <= 1.0000001
                assert abs(float(c['rtc_r0_weighted_projection']) - float(c['aggregation_weight']) * float(c['rtc_r0_projection'])) < 1e-8
            if cell['attack'] == 'none' and c.get('rtc_r0_loo_cosine', '') != '':
                clean_cosines.append(float(c['rtc_r0_loo_cosine']))
        loo_valid = sum(c.get('rtc_r0_loo_cosine', '') != '' for c in clients) / 600
        assert loo_valid >= lock['gates']['loo_valid_fraction_min']
        geometry_audit(rounds, clients)
        for attacker in (False, True):
            selected = [c for c in clients if int(c['round']) >= (1 if cell['attack'] == 'none' else 11) and truth(c['attack_active']) == attacker]
            if selected:
                def avg(field):
                    values = [float(c[field]) for c in selected if c.get(field, '') != '']
                    return sum(values) / len(values) if values else None
                group_reports.append({'attack': cell['attack'], 'seed': cell['seed'], 'active_attacker': attacker,
                    'rows': len(selected), **{field: avg(field) for field in ('aggregation_weight', 'rtc_r0_cosine', 'rtc_r0_negative_contribution', 'rtc_r0_clip_factor')}})
        def round_mean(field):
            return sum(float(r[field]) for r in active) / len(active)
        reports.append({**cell, 'paired_to_r0': paired, 'loo_valid_fraction': loo_valid, 'quality_gates': len(gates), 'reference_valid_fraction': valid,
                        'reconstruction_relative_error': reconstruction,
                        'mean_accuracy': sum(float(r['server_accuracy']) for r in active) / len(active),
                        'final_accuracy': float(next(r['server_accuracy'] for r in rounds if int(r['round']) == 60)),
                        **{field: round_mean(field) for field in ('fit_malicious_aggregation_weight_share',
                            'fit_malicious_impact_share', 'fit_rtc_v3_zero_update_mass', 'fit_rtc_v3_effective_update_mass',
                            'fit_rtc_v3_anchor_recycle_mass', 'fit_rtc_r0_observation_seconds')}})
    assert clean_cosines
    clean_cosines.sort()
    out = root / 'analysis'
    out.mkdir(exist_ok=True)
    threshold = min(0.0, clean_cosines[int(.01 * (len(clean_cosines) - 1))])
    screening = screen_reference(all_clients, threshold, lock['gates'])
    result = {'stage': 'R0b', 'complete': True, 'defense_improvement_claim': False, 'runs': reports, 'groups': group_reports,
              'clean_lower_1pct_loo_cosine': threshold, 'reference_screening': screening,
              'scope': 'Calibration/diagnostic only; no R1 cap was run; no defense improvement claim.',
              'next': 'Review reference screening and abstention reliability before R1; failed screening must not promote a cap; never auto launch'}
    (out / 'decision.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--analyze', action='store_true')
    args = parser.parse_args()
    if args.execute:
        execute()
    elif args.analyze:
        analyze()
    else:
        prepare()


if __name__ == '__main__':
    main()
