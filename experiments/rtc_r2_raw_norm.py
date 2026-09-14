"""R2 standalone raw-norm candidate, six MANUAL training units. Default prepares dry-run only."""
from __future__ import annotations
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.rtc_r0b_observation import read_json, read_csv, digest, snapshot, truth
from defenses.rtc.spectral_direction import load_calibration, calibration_hash, score_gram, VERSION

OUTPUT = ROOT / 'logs/rtc_r2_raw_norm'
CALIBRATION = ROOT / 'config/rtc_r1c_pairwise_calibration.json'
BASE, CAP, MK, RFA = ('rtc_r2_raw_' + s for s in ('baseline', 'cap', 'multikrum', 'rfa'))
RAW_CALIBRATION = ROOT / 'config/rtc_r2_raw_norm_calibration.json'
PROTOCOL = ROOT / 'analysis/rtc_r2_raw_norm/protocol.json'
from defenses.rtc import raw_norm
BATCHES = (('gaussian_seed42', 'gaussian_noise', (BASE, CAP, MK, RFA)), ('clean_seed42', 'none', (BASE, CAP)))
GATES = {'gaussian_active_gain_min': .002, 'gaussian_final_gain_min': 0., 'clean_gain_min': -.002,
         'benign_flag_rate_max': .01, 'malicious_weight_ratio_max': .5, 'budget_violation_max': 1e-8,
         'reconstruction_relative_error_max': 1e-4}
RUNNER_GATES = {'selected_partition_ids_match', 'data_manifest_hash_match', 'trial_plan_hash_match',
    'trial_plan_sequence_match', 'completed_sequence_equals_plan', 'client_random_stream_match',
    'malicious_identity_match', 'paired_data_manifest_match', 'initial_model_match'}


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def command(root, name, attack, defenses, *, execute=False):
    args = [sys.executable, '-m', 'experiments.core.run', '--profile', 'rtc-byzantine',
        '--attacks', attack, '--defenses', ','.join(defenses), '--seeds', '42',
        '--byzantine-attack-freeze', str(root / 'attack.freeze.json'),
        '--attack-start-round', '11', '--attack-end-round', '-1', '--malicious-fractions', '.3',
        '--rounds', '60', '--num-clients', '20', '--participation-rate', '.5', '--batch-size', '48',
        '--partition', 'iid', '--skip-clean', '--max-spec-retries', '0',
        '--ray-client-num-cpus', '1', '--ray-client-num-gpus', '.25',
        '--ray-object-store-memory-mb', '3072', '--ray-min-available-memory-mb', '10240',
        '--ray-memory-wait-seconds', '120', '--output', str(root / name)]
    return args if execute else args + ['--dry-run']


def verify_lock(root=OUTPUT):
    lock = read_json(root / 'r2_lock.json')
    if lock['sources'] != snapshot():
        raise ValueError('R2 source contract changed; do not overwrite or launch')
    for relative, expected in lock['artifacts'].items():
        if digest(root / relative) != expected:
            raise ValueError(f'R2 artifact drift: {relative}')
    for relative, expected in lock['evidence'].items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f'R2 evidence drift: {relative}')
    if lock['gates'] != GATES or lock['training_units'] != 6:
        raise ValueError('R2 preregistration drift')
    return lock


def prepare(root=OUTPUT):
    from attacks.spec import attack_source_hash
    from experiments import periodic_attack
    from experiments.run import parse_args
    if (root / 'r2_lock.json').exists():
        return verify_lock(root)
    if root.exists() and any(root.glob('*/status/*.json')):
        raise ValueError('Refuse to prepare over training results')
    review = read_json(ROOT / 'analysis/rtc_r1c_review/review.json')
    assert review['quality_accepted'] and review['candidate_accepted'] and all(review['independent_checks'].values())
    assert read_json(PROTOCOL)['gates'] == GATES
    calibration = load_calibration(CALIBRATION)
    raw_cal = raw_norm.load_calibration(RAW_CALIBRATION)
    assert digest(PROTOCOL) == raw_cal['provenance']['protocol_sha256']
    evidence = dict(review['evidence'])
    for relative in ('analysis/rtc_r1c_review/review.json', 'analysis/rtc_r1c_review/verify.py',
                     'analysis/rtc_r1c_review/r1c_sources.zip', 'logs/rtc_r1c_pairwise_direction/r1c_lock.json',
                     'analysis/rtc_r2_raw_norm/protocol.json', 'analysis/rtc_r2_raw_norm/calibrate.py',
                     raw_cal['provenance']['clean_source']):
        evidence[relative] = digest(ROOT / relative)
    assert evidence[raw_cal['provenance']['clean_source']] == raw_cal['provenance']['clean_source_sha256']
    assert digest(ROOT/'analysis/rtc_r2_raw_norm/calibrate.py') == raw_cal['provenance']['calibrator_sha256']
    for relative, expected in evidence.items():
        assert digest(ROOT / relative) == expected, relative
    root.mkdir(parents=True, exist_ok=True)
    freeze = {'schema_version': 'RTCByzantineAttackFreezeV1', 'implementation_source_sha256': attack_source_hash(),
        'selection_policy': 'R2 fixed Gaussian strong mean0 std0.1; no strength tuning',
        'attacks': {'gaussian_noise': {'strength_level': 'strong', 'parameters': {'gaussian_noise_mean': 0.0, 'gaussian_noise_std': 0.1}}}}
    (root / 'attack.freeze.json').write_text(json.dumps(freeze, indent=2), encoding='utf-8')
    cells = []
    for name, attack, defenses in BATCHES:
        args_list = command(root, name, attack, defenses)
        subprocess.run(args_list, cwd=ROOT, check=True)
        specs = read_json(root / name / 'experiment_manifest.json')['specs']
        assert len(specs) == len(defenses) and {s['defense'] for s in specs} == set(defenses)
        assert len({s['trial_plan_hash'] for s in specs}) == len({s['attack_implementation_hash'] for s in specs}) == 1
        args = parse_args(args_list[3:])
        configs = {}
        for spec in specs:
            assert spec['seed'] == 42 and spec['attack'] == attack
            rid = periodic_attack.run_id(spec)
            cfg = asdict(periodic_attack._build_spec_config(spec, args, root / name))
            assert cfg['client'] == {'local_epochs': 5, 'batch_size': 48, 'optimizer': 'sgd', 'learning_rate': .01,
                                    'momentum': .9, 'weight_decay': .0001, 'lr_scheduler': 'cosine'}
            assert cfg['federation']['deterministic_client_training'] and cfg['federation']['num_rounds'] == 60
            assert cfg['model']['architecture'] == 'resnet18' and not cfg['model']['pretrained']
            configs[spec['defense']] = cfg
            dest = root / name / 'resolved' / (rid + '.json')
            dest.parent.mkdir(exist_ok=True)
            dest.write_text(json.dumps(cfg, indent=2), encoding='utf-8')
            cells.append({'batch': name, 'attack': attack, 'defense': spec['defense'], 'seed': 42, 'run_id': rid,
                'trial_plan_hash': spec['trial_plan_hash'], 'attack_implementation_hash': spec['attack_implementation_hash'],
                'attack_contract_hash': spec['attack_contract_hash']})
        a, b = copy.deepcopy(configs[BASE]), copy.deepcopy(configs[CAP])
        assert a['security']['defense']['custom_params']['raw_norm_mode'] == 'observe'
        assert b['security']['defense']['custom_params']['raw_norm_mode'] == 'cap'
        b['security']['defense']['custom_params']['raw_norm_mode'] = 'observe'
        assert a == b, 'Baseline and candidate must differ only in raw norm mode'
        assert all(c['security']['defense']['custom_params']['spectral_direction_mode'] == 'observe' for c in configs.values())
        if MK in configs:
            assert configs[MK]['security']['defense']['krum_num_to_select'] == 5
            assert configs[MK]['security']['defense']['krum_num_malicious'] == 3
        if RFA in configs:
            custom = configs[RFA]['security']['defense']['custom_params']
            assert custom['num_iterations'] == 3 and custom['smoothing'] == 1e-6 and custom['use_num_examples']
    artifacts = {str(p.relative_to(root)).replace('\\', '/'): digest(p) for p in root.rglob('*')
                 if p.is_file() and p.suffix in ('.json', '.csv')}
    lock = {'schema': 'RTCR2RawNormLockV1', 'stage': 'R2', 'training_units': 6, 'reused_training_units': 0,
        'baseline': 'B3R-F0.51', 'candidate': 'Standalone preclip trainable raw norm cap; spectral direction observe-only; B3R-F0.51 comparator',
        'calibration': calibration, 'calibration_hash': calibration_hash(calibration),
        'raw_calibration': raw_cal, 'raw_calibration_hash': raw_norm.calibration_hash(raw_cal),
        'gates': GATES, 'cells': cells, 'sources': snapshot(), 'artifacts': artifacts, 'evidence': evidence,
        'projection_policy': 'Paired client/round available in both trajectories; each uses its own LOO axis. Diagnostic proxy, not net attack vector. Coverage reported; no direction-specific gate.',
        'failure_policy': 'Reject on any gate failure; no retuning on returned seed42; no Random-v2 transfer before acceptance.',
        'scope': 'seed42 development screen, original random pressure protocol; no theoretical guarantee or significance claim'}
    (root / 'r2_lock.json').write_text(json.dumps(lock, indent=2), encoding='utf-8')
    print(json.dumps({'ready': True, 'training_units': 6, 'cells': cells}, indent=2))
    return lock


def execute(root=OUTPUT):
    lock = verify_lock(root)
    # Check every run before launching any. Failed/incomplete results require a new manual review.
    for c in lock['cells']:
        path = root / c['batch'] / 'status' / (c['run_id'] + '.json')
        if path.exists():
            s = read_json(path)
            if s['state'] not in ('completed', 'completed_cached') or s['exit_code'] != 0 or s['last_round'] != 60:
                raise ValueError('Incomplete/failed run: no automatic restart')
        elif any((root / c['batch'] / 'raw').glob(c['run_id'] + '*')):
            raise ValueError('Orphaned raw results: inspect before restarting')
    for name, attack, defenses in BATCHES:
        verify_lock(root)
        subprocess.run(command(root, name, attack, defenses, execute=True), cwd=ROOT, check=True)


def verify_cell(root, c, lock):
    batch, rid = root / c['batch'], c['run_id']
    s = read_json(batch / 'status' / (rid + '.json'))
    assert s['state'] in ('completed', 'completed_cached') and s['exit_code'] == 0 and s['last_round'] == 60
    assert read_json(batch / 'resolved' / (rid + '.json')) == read_json(batch / 'raw' / (rid + '_config.json'))
    rr, cc = read_csv(batch / 'rounds' / (rid + '.csv')), read_csv(batch / 'raw' / (rid + '_clients.csv'))
    assert len(rr) == 61 and {int(r['round']) for r in rr} == set(range(61))
    rr.sort(key=lambda r: int(r['round']))
    assert len(cc) == 600 and len({(r['round'], r['cid']) for r in cc}) == 600
    q = read_csv(batch / 'quality_gates.csv')
    expected_gates = RUNNER_GATES | ({'attack_execution:' + x['run_id'] for x in lock['cells'] if x['batch'] == c['batch']} if c['attack'] != 'none' else set())
    assert expected_gates <= {r['gate'] for r in q} and all(truth(r['passed']) for r in q)
    pair = read_json(batch / 'raw' / (rid + '_pairing_manifest.json'))
    data = read_json(batch / 'raw' / (rid + '_data_manifest.json'))
    trial = read_json(Path(pair['trial_plan_path']))
    assert canonical({k: v for k, v in data.items() if k != 'sha256'}) == data['sha256'] == pair['data_manifest_sha256']
    assert canonical({k: v for k, v in trial.items() if k != 'trial_plan_hash'}) == pair['trial_plan_hash'] == c['trial_plan_hash']
    assert pair['pairing_mode'] == 'strict' and pair['sampling_protocol'] == 'principal_uniform' and pair['deterministic_client_training'] is True
    for r in rr:
        for key in ('server_accuracy', 'server_loss'):
            assert math.isfinite(float(r[key]))
        if int(r['round']) == 0:
            continue
        for key in ('trial_plan_hash', 'attack_implementation_hash', 'attack_contract_hash'):
            assert r[key] == c[key]
        for key in ('fit_received_updates_finite', 'fit_coordinated_updates_finite', 'fit_aggregate_parameters_finite'):
            assert truth(r[key])
        assert r['fit_rtc_r1_version'] == VERSION and r['fit_rtc_r1_calibration_hash'] == lock['calibration_hash']
        assert r['fit_rtc_r1_mode'] == 'observe'
        for key in ('fit_rtc_r1_reconstruction_error', 'fit_rtc_r1_actual_aggregate_norm',
                    'fit_rtc_r1_anchor_contribution_norm', 'fit_rtc_r1_client_sum_norm', 'fit_aggregation_time_seconds'):
            assert math.isfinite(float(r[key])) and float(r[key]) >= 0, key
        assert float(r['fit_rtc_r1_reconstruction_error']) / max(float(r['fit_rtc_r1_actual_aggregate_norm']), 1e-8) <= GATES['reconstruction_relative_error_max']
        ids = json.loads(r['fit_rtc_r1_client_ids_json'])
        plan = trial['rounds'][int(r['round']) - 1]
        assert json.loads(r['fit_completed_partition_ids_json']) == plan['partition_ids']
        assert r['fit_fit_seed_digest'] == plan['fit_seed_digest']
        clients = {row['cid']: row for row in cc if row['round'] == r['round']}
        assert len(ids) == len(set(ids)) == len(clients) == 10 and set(ids) == set(clients) == set(plan['partition_ids'])
        gram = np.asarray(json.loads(r['fit_rtc_r1_gram_json']))
        scores = score_gram(gram, lock['calibration']['f'])
        assert r['fit_rtc_r2_version'] == raw_norm.VERSION
        assert r['fit_rtc_r2_calibration_hash'] == lock['raw_calibration_hash']
        raw_mode = 'cap' if c['defense'] == CAP else 'observe'
        assert r['fit_rtc_r2_mode'] == raw_mode
        assert math.isfinite(float(r['fit_rtc_r2_scoring_seconds'])) and float(r['fit_rtc_r2_scoring_seconds']) >= 0
        norms = [float(clients[cid]['rtc_r2_raw_norm']) for cid in ids]
        assert all(math.isfinite(v) and v >= 0 for v in norms)
        # Recompute leave-one-out medians without production score().
        import statistics
        for i, cid in enumerate(ids):
            row = clients[cid]
            clip_factor = float(row['rtc_r2_clip_factor'])
            assert math.isfinite(clip_factor) and 0 <= clip_factor <= 1
            assert abs(norms[i]*clip_factor - float(row['rtc_r1_trainable_clipped_norm'])) <= 1e-8*max(1,norms[i]*clip_factor)
            if c['defense'] in (MK, RFA):
                assert clip_factor == 1.
            others = [v for j,v in enumerate(norms) if j != i and v > 1e-12]
            valid = len(others) >= 9
            median = statistics.median(others) if valid else None
            ratio = norms[i]/median if valid else None
            flag = valid and ratio > lock['raw_calibration']['ratio_threshold']
            assert int(row['rtc_r2_reference_count']) == len(others)
            assert truth(row['rtc_r2_valid']) == valid and truth(row['rtc_r2_flagged']) == flag
            for key, value in [('rtc_r2_reference_median',median), ('rtc_r2_ratio',ratio)]:
                if value is None:
                    assert row[key] == ''
                else:
                    assert math.isfinite(float(row[key])) and abs(float(row[key])-value) <= 1e-10*max(1,abs(value))
            q = 0. if flag and raw_mode == 'cap' else 1.
            assert float(row['rtc_r2_q']) == q
            if c['defense'] in (BASE, CAP):
                previous = min(float(row['rtc_r1_existing_q']), float(row['rtc_r1_q']))
                assert float(row['rtc_r2_existing_q']) == previous
                assert truth(row['rtc_r2_applied']) == (q < previous)
                assert float(row['rtc_r2_nominal_mass']) == float(row['rtc_r1_nominal_mass'])
                assert float(row['aggregation_weight']) <= float(row['rtc_r2_nominal_mass'])*min(q,previous)+1e-8

        for i, cid in enumerate(ids):
            row, score = clients[cid], scores[i]
            identity = int(cid) in pair['malicious_partition_ids']
            assert truth(row['is_malicious']) == identity
            assert truth(row['attack_active']) == (identity and c['attack'] != 'none' and int(r['round']) >= 11)
            assert math.isfinite(float(row['aggregation_weight'])) and 0 <= float(row['aggregation_weight']) <= 1
            assert abs(float(row['rtc_r1_trainable_clipped_norm']) ** 2 - gram[i, i]) <= 1e-8 * max(1, gram[i, i])
            valid = score['valid'] and score['gap'] >= lock['calibration']['minimum_gap']
            flag = valid and score['cosine'] < lock['calibration']['cosine_threshold']
            from defenses.rtc.corroboration import corroboration
            guard = corroboration(gram, i, score['reference_indices'], lock['calibration']['corroboration']['pairwise_threshold'])
            assert truth(row['rtc_r1_base_flagged']) == flag
            assert int(row['rtc_r1_corroboration_votes']) == guard['votes']
            assert int(row['rtc_r1_corroboration_required']) == guard['required']
            assert truth(row['rtc_r1_corroboration_passes']) == guard['passes']
            flag = flag and guard['passes']
            assert truth(row['rtc_r1_valid']) == score['valid'] and truth(row['rtc_r1_usable']) == valid
            assert truth(row['rtc_r1_flagged']) == flag
            assert json.loads(row['rtc_r1_reference_ids_json']) == [ids[j] for j in score['reference_indices']]
            assert np.allclose(json.loads(row['rtc_r1_reference_weights_json']), score['reference_weights'], atol=1e-10, rtol=0)
            assert float(row['rtc_r1_q']) == 1.
            if score['valid']:
                assert abs(float(row['rtc_r1_cosine']) - score['cosine']) < 1e-8
            if valid:
                assert all(math.isfinite(float(row[k])) for k in ('rtc_r1_anchor_projection', 'rtc_r1_actual_projection', 'rtc_r1_weighted_projection'))
                projection = math.sqrt(max(0, gram[i, i])) * score['cosine']
                assert abs(float(row['rtc_r1_projection']) - projection) <= 1e-8 * max(1., abs(projection))
                negative = float(row['aggregation_weight']) * max(0, -projection)
                assert abs(float(row['rtc_r1_negative_contribution']) - negative) <= 1e-8 * max(1., negative)
            else:
                assert row['rtc_r1_projection'] == row['rtc_r1_negative_contribution'] == ''
            if c['defense'] in (BASE, CAP):
                assert float(row['aggregation_weight']) <= float(row['rtc_r1_nominal_mass']) * min(float(row['rtc_r1_existing_q']), float(row['rtc_r1_q'])) + 1e-8
                assert truth(row['rtc_r1_applied']) == (float(row['rtc_r1_q']) < float(row['rtc_r1_existing_q']))
        if c['defense'] in (BASE, CAP):
            for key, value in {'semantic_intervention_risk_floor': .5, 'cumulative_q_cap_power': 1,
                               'anchor_recycle_fraction': .51, 'norm_clip_mad_k': 2.5}.items():
                assert float(r['fit_rtc_v3_' + key]) == value
            assert r['fit_rtc_v3_anchor_recycle_weighting'] == 'accepted'
            assert float(r['fit_rtc_v3_max_constraint_violation']) <= GATES['budget_violation_max']
    return {'cell': c, 'rounds': rr, 'clients': cc, 'pair': pair}


def paired(a, b):
    for key in ('trial_plan_hash', 'data_manifest_sha256', 'initial_model_sha256', 'malicious_identity_sha256',
                'malicious_partition_ids', 'deterministic_client_training', 'pairing_mode', 'sampling_protocol'):
        assert a['pair'][key] == b['pair'][key], key
    assert a['cell']['attack_implementation_hash'] == b['cell']['attack_implementation_hash']
    for ar, br in zip(a['rounds'][1:], b['rounds'][1:]):
        for key in ('round', 'attack_contract_hash', 'fit_completed_partition_ids_json', 'fit_fit_seed_digest'):
            assert ar[key] == br[key], key
        if a['cell']['defense'] in (BASE, CAP) and b['cell']['defense'] in (BASE, CAP):
            assert ar['fit_defense_random_seed'] == br['fit_defense_random_seed']
    ac = {(c['round'], c['cid']): c for c in a['clients']}
    for c in b['clients']:
        for key in ('is_malicious', 'attack_active', 'principal_id', 'num_examples', 'local_epochs'):
            assert ac[(c['round'], c['cid'])][key] == c[key], key
    return True


def summarize(run):
    start = 1 if run['cell']['attack'] == 'none' else 11
    rr = [r for r in run['rounds'] if int(r['round']) >= start]
    cc = [c for c in run['clients'] if int(c['round']) >= start]
    benign = [c for c in cc if not truth(c['attack_active'])]
    bad = [c for c in cc if truth(c['attack_active'])]
    def average(key):
        values = [float(r[key]) for r in rr if r.get(key, '') != '']
        return float(np.mean(values)) if values else None
    return {**run['cell'], 'active_accuracy': average('server_accuracy'),
        'final_accuracy': float(run['rounds'][-1]['server_accuracy']), 'asr': None,
        'asr_reason': 'untargeted/no attack; targeted safety is reserved for later registered stages',
        'malicious_weight_per_round': sum(float(c['aggregation_weight']) for c in bad) / len(rr) if bad else None,
        'benign_weight_per_round': sum(float(c['aggregation_weight']) for c in benign) / len(rr),
        'benign_flag_rate': sum(truth(c['rtc_r2_flagged']) for c in benign) / len(benign),
        'malicious_flag_rate': sum(truth(c['rtc_r2_flagged']) for c in bad) / len(bad) if bad else None,
        'benign_projection_availability': sum(truth(c['rtc_r1_usable']) for c in benign) / len(benign),
        'malicious_projection_availability': sum(truth(c['rtc_r1_usable']) for c in bad) / len(bad) if bad else None,
        'zero_mass': average('fit_rtc_v3_zero_update_mass'), 'effective_mass': average('fit_rtc_v3_effective_update_mass'),
        'anchor_mass': average('fit_rtc_v3_anchor_recycle_mass'),
        'anchor_norm': average('fit_rtc_r1_anchor_contribution_norm'), 'aggregate_norm': average('fit_rtc_r1_actual_aggregate_norm'),
        'aggregation_seconds': average('fit_aggregation_time_seconds'),
        'raw_scoring_seconds': average('fit_rtc_r2_scoring_seconds'),
        'spectral_scoring_seconds': average('fit_rtc_r1_scoring_seconds'), 'observation_seconds': average('fit_rtc_r1_observation_seconds'),
        'rounds_exceeding_f3': sum(sum(truth(c['attack_active']) for c in bad if c['round'] == r['round']) > 3 for r in rr),
        'first_flagged_attacker_round': min((int(c['round']) for c in bad if truth(c['rtc_r2_flagged'])), default=None),
        'first_applied_round': min((int(c['round']) for c in run['clients'] if truth(c.get('rtc_r2_applied'))), default=None)}


def decide(runs):
    summaries = [summarize(r) for r in runs]
    index = {(r['cell']['attack'], r['cell']['defense']): r for r in runs}
    summary = {(r['attack'], r['defense']): r for r in summaries}
    pairings = [paired(index[attack, BASE], index[attack, d]) for _, attack, ds in BATCHES for d in ds if d != BASE]
    b, c = summary['gaussian_noise', BASE], summary['gaussian_noise', CAP]
    clean_b, clean_c = summary['none', BASE], summary['none', CAP]
    old = {(r['round'], r['cid']): r for r in index['gaussian_noise', BASE]['clients']}
    matched = [(old[r['round'], r['cid']], r) for r in index['gaussian_noise', CAP]['clients']
               if int(r['round']) >= 11 and truth(r['attack_active']) and truth(r['rtc_r1_usable'])
               and truth(old[r['round'], r['cid']]['rtc_r1_usable'])]
    total_bad = sum(truth(r['attack_active']) for r in index['gaussian_noise', CAP]['clients'] if int(r['round']) >= 11)
    assert total_bad > 0
    negative_b = sum(float(a['rtc_r1_negative_contribution']) for a, _ in matched) / 50 if matched else None
    negative_c = sum(float(a['rtc_r1_negative_contribution']) for _, a in matched) / 50 if matched else None
    checks = {'strict_pairing': all(pairings),
        'gaussian_active_gain': c['active_accuracy'] - b['active_accuracy'] >= GATES['gaussian_active_gain_min'],
        'gaussian_final_gain': c['final_accuracy'] - b['final_accuracy'] >= GATES['gaussian_final_gain_min'],
        'clean_active': clean_c['active_accuracy'] - clean_b['active_accuracy'] >= GATES['clean_gain_min'],
        'clean_final': clean_c['final_accuracy'] - clean_b['final_accuracy'] >= GATES['clean_gain_min'],
        'gaussian_benign_flags': c['benign_flag_rate'] <= GATES['benign_flag_rate_max'],
        'clean_benign_flags': clean_c['benign_flag_rate'] <= GATES['benign_flag_rate_max'],
        'malicious_weight_decreased': c['malicious_weight_per_round'] <= b['malicious_weight_per_round'] * GATES['malicious_weight_ratio_max'] and c['malicious_weight_per_round'] < b['malicious_weight_per_round']}
    divergences = []
    for _, attack, _ in BATCHES:
        baseline = index[attack, BASE]
        candidate = index[attack, CAP]
        old_c = {(x['round'], x['cid']): x for x in baseline['clients']}
        first = next(({'round': int(r['round']), 'cid': r['cid']} for r in sorted(candidate['clients'], key=lambda x: (int(x['round']), x['cid']))
                     if abs(float(r['aggregation_weight']) - float(old_c[r['round'], r['cid']]['aggregation_weight'])) > 1e-10), None)
        update = next((int(a['round']) for a, b in zip(baseline['rounds'][1:], candidate['rounds'][1:])
                       if a['fit_aggregate_update_sketch_json'] != b['fit_aggregate_update_sketch_json']), None)
        divergences.append({'attack': attack, 'first_weight_difference': first, 'first_sketch_difference_round': update})
    return {'stage': 'R2', 'quality_accepted': True, 'checks': checks, 'candidate_accepted': all(checks.values()),
        'summaries': summaries, 'first_divergences': divergences,
        'negative_proxy': {'baseline': negative_b, 'candidate': negative_c, 'matched_rows': len(matched),
                           'total_attacker_rows': total_bad, 'scope': 'paired available client/round; per-client LOO axes, not net vector'},
        'next': 'Review before accepting standalone R2; Random-v2 transfer requires a new manual batch', 'development_only': True}


def analyze(root=OUTPUT):
    lock = verify_lock(root)
    runs = [verify_cell(root, c, lock) for c in lock['cells']]
    decision = decide(runs)
    destination = root / 'analysis'
    destination.mkdir(exist_ok=True)
    decision['source_lock_sha256'] = digest(root / 'r2_lock.json')
    decision['evidence'] = {str(p.relative_to(root)).replace('\\', '/'): digest(p)
        for p in root.glob('*/*/*') if p.is_file() and p.suffix in ('.csv', '.json') and 'analysis' not in p.parts}
    (destination / 'decision.json').write_text(json.dumps(decision, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(decision, indent=2, allow_nan=False))
    return decision


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--execute', action='store_true')
    group.add_argument('--analyze', action='store_true')
    args = parser.parse_args()
    analyze() if args.analyze else execute() if args.execute else prepare()
