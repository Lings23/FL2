"""Manual MNIST calibration, clean validation and 108-cell M3 comparison.

The default only prepares the calibration matrix. No mode chains into training.
"""
import argparse
import copy
import csv
from dataclasses import asdict
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import sys
from datetime import datetime, timezone
import zipfile

# Load Torch before pandas/NumPy-backed experiment modules on Windows (OpenMP).
import torch

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'config/rtc_mnist_all_attacks_protocol.json'
OUTPUT = ROOT / 'logs/rtc_mnist_all_attacks_seed42'
M3 = 'rtc_i12_eligibility_confirmed'
TARGETED = {'label_flip_targeted', 'label_flip_all_reverse', 'scaling_backdoor', 'dba'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def integer(value):
    number = Decimal(str(value))
    require(number.is_finite() and number == number.to_integral_value(), f'Not an integer: {value}')
    return int(number)


def truth(value):
    return str(value).strip().lower() in ('true', '1', '1.0')


def inventory(paths, base):
    return {p.relative_to(base).as_posix(): sha(p) for p in sorted(paths) if p.is_file()}


def sources():
    paths = [ROOT / 'main.py', ROOT / 'requirements.txt', ROOT / 'requirements-l20.txt']
    for folder in ('experiments', 'defenses', 'attacks', 'client', 'server', 'strategies', 'models', 'data', 'config', 'utils'):
        paths.extend(p for p in (ROOT / folder).rglob('*') if p.suffix in ('.py', '.json', '.yaml', '.sh', '.ps1'))
    return inventory(paths, ROOT)


def environment():
    from experiments.rtc_i12_bridge import environment as current
    return current()


def runtime_config(base, data_dir, gpu):
    cfg = copy.deepcopy(base)
    cfg['client'] = read(PROTOCOL)['training']
    cfg['dataset'].update(name='mnist', data_dir=str(data_dir), num_classes=10,
                          partition='iid', val_split=.1, download_source='torchvision')
    cfg['model'].update(architecture='mnist_cnn', num_classes=10, pretrained=False)
    cfg['security']['defense']['custom_params'] = {}
    cfg['differential_privacy']['enabled'] = False
    cfg.setdefault('ray', {})['client_num_gpus'] = gpu
    return cfg


def runner_args(root, output):
    from experiments.run import parse_args
    settings = read(root / 'settings.json')
    return parse_args(['--profile', 'rtc-byzantine', '--config', str(root / 'runtime_config.yaml'),
        '--output', str(output), '--rounds', '60', '--num-clients', '20', '--participation-rate', '.5',
        '--malicious-fractions', '.3', '--partition', 'iid', '--batch-size', '96',
        '--attack-start-round', '11', '--attack-end-round', '60', '--pairing-mode', 'strict',
        '--ray-client-num-cpus', '1', '--ray-client-num-gpus', str(settings['client_num_gpus']),
        '--ray-object-store-memory-mb', '3072', '--ray-min-available-memory-mb', '10240',
        '--ray-memory-wait-seconds', '120', '--max-spec-retries', '0', '--skip-clean', '--dry-run'])


def verify_config(cfg, stage):
    require(cfg['dataset']['name'] == 'mnist' and cfg['dataset']['num_classes'] == 10, 'Dataset drift')
    require(cfg['model']['architecture'] == 'mnist_cnn' and not cfg['model']['pretrained'], 'Model drift')
    require(cfg['client'] == read(PROTOCOL)['training'], 'Training recipe drift')
    require(cfg['federation']['num_rounds'] == 60 and cfg['federation']['clients_per_round'] == 10, 'FL recipe drift')
    require(cfg['federation']['deterministic_client_training'], 'Strict fit RNG required')
    require(not cfg['differential_privacy']['enabled'], 'DP must be disabled')


def observer_params():
    return dict(spectral_direction_mode='observe',
        spectral_direction_calibration=str(ROOT / 'config/rtc_r1c_pairwise_calibration.json'),
        raw_norm_mode='observe', raw_norm_calibration=str(ROOT / 'config/rtc_r2_raw_norm_calibration.json'))


def make_specs(root, stage, condition, seed, defenses):
    from experiments.rtc_v3 import byzantine
    from experiments import rtc_v3_formal_calibration as calibration
    from experiments.trial_plan import attach_trial_plans
    batch = root / stage / condition['name']
    args = runner_args(root, batch)
    if stage == 'calibration':
        specs = calibration.build_clean_specs(seeds=[seed], bootstrap_manifest=root / 'bootstrap.json',
            num_clients=20, participation_rate=.5, partition='iid', dirichlet_alpha=.5)
        for spec in specs:
            spec['custom_params'].update(observer_params())
    else:
        args.attacks = condition['attack']
        args.defenses = ','.join(defenses)
        args.seeds = str(seed)
        args.malicious_fraction = .3
        args.byzantine_attack_freeze = str(root / 'freezes' / (condition['name'] + '.json'))
        args.byzantine_evaluation_only = True
        args.rtc_v3_manifest = str(root / 'calibrated/manifest.json')
        specs = byzantine.build_matrix(args)
        for spec in specs:
            if spec['defense'] == M3:
                spec['custom_params'].update(
                    spectral_direction_calibration=str(root / 'calibrated/direction.json'),
                    raw_norm_calibration=str(root / 'calibrated/raw_norm.json'))
    for spec in specs:
        spec.update(dataset='mnist', architecture='mnist_cnn', strength_validation='unverified_on_mnist')
    args.sampling_protocol = 'principal_uniform'
    return attach_trial_plans(specs, args, batch), args


def save_batch(root, stage, name, specs, args):
    from experiments import periodic_attack as runner
    from experiments.trial_plan import TrialPlanV1
    batch = root / stage / name
    runner.write_experiment_manifest(specs, batch)
    require(len({s['trial_plan_hash'] for s in specs}) == 1, 'Unpaired defenses')
    cells = []
    for spec in specs:
        cfg = asdict(runner._build_spec_config(spec, args, batch))
        verify_config(cfg, stage)
        if stage != 'calibration':
            from attacks.spec import attack_contract_payload
            contract = attack_contract_payload(spec['attack'], cfg['security']['attack'])
            require(contract['contract_hash'] == spec['attack_contract_hash'] and
                    contract['implementation_hash'] == spec['attack_implementation_hash'], 'Resolved attack contract differs from spec')
        plan = TrialPlanV1.load(spec['trial_plan_path']).payload
        require(plan['data_config']['dataset'] == 'mnist' and
                plan['initial_model_config']['architecture'] == 'mnist_cnn', 'TrialPlan model drift')
        rid = runner.run_id(spec)
        write(batch / 'resolved' / (rid + '.json'), cfg)
        cells.append(dict(batch=name, run_id=rid, spec=spec))
    return cells


def lock_stage(root, stage, cells):
    stage_dir = root / stage
    files = inventory(stage_dir.rglob('*'), root)
    lock = dict(schema='RTCMNISTStageLockV1', stage=stage, cells=cells, protocol=read(PROTOCOL),
        environment=environment(), sources=sources(), artifacts=files,
        root_artifacts=inventory([root / 'runtime_config.yaml', root / 'settings.json', root / 'bootstrap.json',
            *list((root / 'freezes').glob('*.json')), *list((root / 'calibrated').glob('*.json'))], root))
    write(stage_dir / 'lock.json', lock)
    with zipfile.ZipFile(stage_dir / 'sources.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for name in lock['sources']:
            z.write(ROOT / name, name)
    return lock


def verify_lock(root, stage, training=False):
    lock = read(root / stage / 'lock.json')
    require(lock['protocol'] == read(PROTOCOL), 'Protocol changed after preparation')
    require(lock['sources'] == sources(), 'Source drift: preserve batch; restore frozen source before continuing')
    if training:
        require(lock['environment'] == environment(), 'Execute on the host/environment used to prepare this stage')
    for name, digest in {**lock['artifacts'], **lock['root_artifacts']}.items():
        require(sha(root / name) == digest, 'Frozen artifact changed: ' + name)
    return lock


def prepare(root, stage, data_dir=None, gpu=None):
    if (root / stage / 'lock.json').exists():
        settings = read(root / 'settings.json')
        require(data_dir is None or str(Path(data_dir).resolve()) == settings['data_dir'], 'Frozen data-dir differs')
        require(gpu is None or gpu == settings['client_num_gpus'], 'Frozen GPU quota differs')
        return verify_lock(root, stage)
    stage_dir = root / stage
    require(not stage_dir.exists() or not any(stage_dir.iterdir()), 'Nonempty unfrozen stage; preserve it and use a new output')
    p = read(PROTOCOL)
    if stage == 'calibration':
        import yaml
        from experiments.rtc_v3_formal_calibration import build_semantic_collection_bootstrap
        from experiments.rtc_v3_manifest import build_bootstrap_manifest
        from defenses.rtc.calibration import content_hash
        from attacks.spec import attack_source_hash
        require(not root.exists() or not any(root.iterdir()), 'Output exists without calibration lock')
        root.mkdir(parents=True, exist_ok=True)
        gpu = p['ray_client_num_gpus'] if gpu is None else gpu
        require(math.isfinite(gpu) and gpu > 0, 'GPU quota must be finite and positive')
        data_dir = Path(data_dir or ROOT / 'data').resolve()
        write(root / 'settings.json', dict(data_dir=str(data_dir), client_num_gpus=gpu))
        cfg = runtime_config(yaml.safe_load((ROOT / 'config/config.yaml').read_text(encoding='utf-8')), data_dir, gpu)
        (root / 'runtime_config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8', newline='\n')
        bootstrap = build_bootstrap_manifest(root / 'runtime_config.yaml')
        bootstrap['training_phases'] = [{'start_round': n, 'profile': v} for n, v in [(1, 'warmup'), (11, 'steady'), (31, 'late')]]
        bootstrap['profile'] = 'warmup'
        bootstrap['content_hash'] = content_hash(bootstrap)
        write(root / 'bootstrap_base.json', bootstrap)
        build_semantic_collection_bootstrap(base_manifest=root / 'bootstrap_base.json',
            config_path=root / 'runtime_config.yaml', output_path=root / 'bootstrap.json')
        for condition in p['conditions']:
            attack = condition['attack']
            entry = copy.deepcopy(p['attacks'].get(attack, {}))
            if 'parameters' in condition:
                entry['parameters'] = condition['parameters']
            write(root / 'freezes' / (condition['name'] + '.json'), dict(
                schema_version='RTCByzantineAttackFreezeV1', implementation_source_sha256=attack_source_hash(),
                purpose='fixed_strength_evaluation_only', selection_policy=p['attack_strength_policy'],
                attacks={} if attack == 'none' else {attack: entry}))
        cells = []
        for seed in p['calibration_seeds']:
            c = dict(name=f'clean_seed{seed}', attack='none')
            specs, args = make_specs(root, stage, c, seed, ())
            cells += save_batch(root, stage, c['name'], specs, args)
    else:
        verify_lock(root, 'calibration')
        receipt = read(root / 'calibrated/receipt.json')
        require(receipt['passed'], 'Clean calibration not accepted')
        for name, h in receipt['files'].items():
            require(sha(root / name) == h, 'Calibration evidence changed: ' + name)
        if stage == 'evaluation':
            verify_lock(root, 'validation')
            validation = read(root / 'validation/analysis/decision.json')
            require(validation['passed'], 'Independent clean validation failed; preserve results, do not launch evaluation')
            for name, h in validation['evidence'].items():
                require(sha(root / name) == h, 'Validation evidence changed')
        cells = []
        conditions = [p['conditions'][0]] if stage == 'validation' else p['conditions']
        seed = 43 if stage == 'validation' else 42
        defenses = [M3, 'fedavg'] if stage == 'validation' else p['defenses']
        for c in conditions:
            specs, args = make_specs(root, stage, c, seed, defenses)
            cells += save_batch(root, stage, c['name'], specs, args)
    require(len(cells) == p[stage + '_units'], 'Matrix cardinality mismatch')
    return lock_stage(root, stage, cells)


def cell_paths(root, stage, cell):
    batch = root / stage / cell['batch']; rid = cell['run_id']
    return batch, batch / 'status' / (rid + '.json'), batch / 'rounds' / (rid + '.csv')


def complete(root, stage, cell):
    from experiments import periodic_attack as runner
    _, status, round_path = cell_paths(root, stage, cell)
    if not status.exists():
        return False
    s = read(status)
    if stage == 'evaluation' and runner.valid_numerical_terminal(s, cell['spec']):
        return True
    return (s.get('state') in ('completed', 'completed_cached') and s.get('exit_code') == 0
        and s.get('last_round') == 60 and runner.validate_round_cache(round_path, 60, cell['spec'])[0])


def execute(root, stage, restart=False):
    from experiments import periodic_attack as runner
    from experiments.rtc_m3_multikrum import ensure_no_other_training
    lock = verify_lock(root, stage, training=True)
    ensure_no_other_training()
    for cell in lock['cells']:
        batch, status, round_path = cell_paths(root, stage, cell)
        if complete(root, stage, cell):
            continue
        remnants = [p for p in [status, round_path, *list((batch / 'raw').glob(cell['run_id'] + '*'))] if p.exists()]
        if remnants:
            require(restart, f'Incomplete cell {cell["run_id"]}; explicit --restart-incomplete archives evidence and restarts round0')
            archive = batch / 'interrupted' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') / cell['run_id']
            archive.mkdir(parents=True)
            for path in remnants:
                require(path.resolve().is_relative_to(root.resolve()), 'Restart path escapes output')
                destination = archive / path.parent.name / path.name
                destination.parent.mkdir(exist_ok=True)
                shutil.move(str(path), str(destination))
            write(archive / 'restart_receipt.json', dict(run_id=cell['run_id'], restart_from_round=0))
        verify_lock(root, stage, training=True)
        args = runner_args(root, batch); args.dry_run = False
        (batch / 'rounds').mkdir(exist_ok=True)
        runner._run_specs([cell['spec']], args, batch, batch / 'rounds')
        require(complete(root, stage, cell), 'Incomplete training; no automatic retry')
    print('Stage execution finished. Run --analyze manually; no next stage was launched.')


def validate_metric_rows(round_rows, attack):
    require([integer(r['round']) for r in round_rows] == list(range(61)), 'Expected exactly rounds0..60')
    class_totals = None
    for r in round_rows:
        matrix = json.loads(r['server_confusion_matrix_json'])
        require(len(matrix) == 10 and all(len(v) == 10 for v in matrix), 'Confusion shape')
        require(all(type(v) is int and v >= 0 for row in matrix for v in row), 'Confusion counts')
        totals = [sum(row) for row in matrix]
        require(sum(totals) == 10000 and all(n > 0 for n in totals), 'Full MNIST test set required')
        class_totals = totals if class_totals is None else class_totals
        require(totals == class_totals, 'Test population changed between rounds')
        require(abs(sum(matrix[i][i] for i in range(10)) / 10000 - float(r['server_accuracy'])) < 1e-12, 'ACC reconstruction failed')
        for k in ('server_model_state_valid', 'server_logits_valid', 'server_loss_valid'):
            require(truth(r[k]), 'Numerically invalid model: ' + k)
        if attack in TARGETED:
            value = float(r['server_asr'])
            require(math.isfinite(value) and 0 <= value <= 1 and truth(r['server_asr_valid']), 'Invalid ASR')
            if attack == 'label_flip_targeted':
                total, success = totals[5], matrix[5][3]
            elif attack == 'label_flip_all_reverse':
                total, success = 10000, sum(matrix[i][9-i] for i in range(10))
            else:
                total, success = 10000 - totals[0], value * (10000 - totals[0])
                require(abs(success - round(success)) < 1e-8, 'Nonintegral triggered successes')
            require(integer(r['server_asr_total']) == total and abs(value - success/total) < 1e-12, 'ASR denominator/reconstruction mismatch')
    return class_totals


def load_runs(root, stage, lock):
    from experiments import periodic_attack as runner
    from experiments.rtc_v3.byzantine import validate_attack_execution
    runs, evidence, gates = [], {}, []
    for cell in lock['cells']:
        require(complete(root, stage, cell), 'Missing/failed/incomplete cell: ' + cell['run_id'])
        batch, status, round_path = cell_paths(root, stage, cell)
        raw = batch / 'raw'; rid = cell['run_id']; spec = cell['spec']
        cfg_path = raw / (rid + '_config.json')
        cfg = read(cfg_path); verify_config(cfg, stage)
        require(cfg == read(batch / 'resolved' / (rid + '.json')), 'Runtime config differs from frozen resolved config')
        if runner.valid_numerical_terminal(read(status), spec):
            require(stage == 'evaluation', 'Numerically invalid clean calibration/validation')
            result = dict(condition=cell['batch'], defense=spec['defense'], seed=spec['seed'],
                execution_state='terminated_numerical', failure_round=read(status).get('failure_round', read(status).get('last_round')),
                active_accuracy=None, final_accuracy=None, tail_accuracy=None,
                asr_mean=None, asr_peak=None, asr_tail10=None, asr_final=None, benign_union_cap_rate=None)
            runs.append(dict(cell=cell, rounds=[], clients=[], summary=result))
            evidence.update(inventory([status, *raw.glob(rid+'*')], root))
            continue
        rr = rows(round_path); cr = rows(raw / (rid + '_clients.csv'))
        validate_metric_rows(rr, spec['attack'])
        require(len(cr) == 600 and len({(integer(c['round']), c['cid']) for c in cr}) == 600, 'Client coverage')
        require(all(sum(integer(c['round']) == n for c in cr) == 10 for n in range(1, 61)), 'Ten clients per round required')
        if spec['defense'] == M3 or stage == 'calibration':
            from defenses.rtc.calibration import CalibrationManifest
            from defenses.rtc.spectral_direction import load_calibration, calibration_hash
            from defenses.rtc.raw_norm import load_calibration as load_raw, calibration_hash as raw_hash
            expected = CalibrationManifest.load(spec['custom_params']['calibration_path']).hash
            direction_hash = calibration_hash(load_calibration(spec['custom_params']['spectral_direction_calibration']))
            norm_hash = raw_hash(load_raw(spec['custom_params']['raw_norm_calibration']))
            mode = 'cap' if spec['defense'] == M3 else 'observe'
            for row in rr[1:]:
                require(row['fit_rtc_v3_calibration_hash'] == expected, 'RTC calibration hash drift')
                require(row['fit_rtc_r1_calibration_hash'] == direction_hash and row['fit_rtc_r2_calibration_hash'] == norm_hash,
                        'R1/R2 calibration hash drift')
                require(row['fit_rtc_r1_mode'] == row['fit_rtc_r2_mode'] == mode, 'R1/R2 mode drift')
                require(float(row['fit_rtc_v3_max_constraint_violation']) <= 1e-8 and
                        0 <= float(row['fit_rtc_v3_weight_sum']) <= 1 + 1e-8, 'RTC constraint/weight violation')
            if spec['defense'] == M3:
                require(all(integer(r['fit_rtc_r1e_quorum_numerator']) == 2 and
                            integer(r['fit_rtc_r1e_quorum_denominator']) == 3 for r in rr[1:]), 'H2b quorum drift')
        start = 1 if spec['attack'] == 'none' else 11
        active = rr[start:]; tail = rr[51:]
        result = dict(condition=cell['batch'], defense=spec['defense'], seed=spec['seed'],
            execution_state='completed', failure_round=None,
            active_accuracy=statistics.mean(float(r['server_accuracy']) for r in active),
            final_accuracy=float(rr[-1]['server_accuracy']), tail_accuracy=statistics.mean(float(r['server_accuracy']) for r in tail))
        for label, series in [('mean', active), ('peak', active), ('tail10', tail), ('final', rr[-1:])]:
            result['asr_' + label] = ((max if label == 'peak' else statistics.mean)(float(r['server_asr']) for r in series)
                if spec['attack'] in TARGETED else None)
        good = [r for r in cr if integer(r['round']) >= start and not truth(r['attack_active'])]
        result['benign_union_cap_rate'] = (sum(float(r['rtc_r1_q']) < 1 or float(r['rtc_r2_q']) < 1 for r in good)/len(good)
            if spec['defense'] == M3 else None)
        runs.append(dict(cell=cell, rounds=rr, clients=cr, summary=result))
        evidence.update(inventory([status, round_path, *raw.glob(rid + '*.json'), raw / (rid + '_clients.csv')], root))
    for name in sorted({c['batch'] for c in lock['cells']}):
        selected = [c for c in lock['cells'] if c['batch'] == name]; batch = root / stage / name
        finished = {r['cell']['run_id'] for r in runs if r['cell']['batch'] == name and r['summary']['execution_state'] == 'completed'}
        checks = runner.validate_sampling_manifests(batch / 'rounds', batch / 'raw', expected_ids=finished) if finished else []
        if stage != 'calibration':
            checks += validate_attack_execution([c['spec'] for c in selected], batch / 'rounds')
        for check in checks:
            gates.append(dict(batch=name, **check))
    write(root / stage / 'analysis/quality.json', gates)
    require(gates and all(g['passed'] for g in gates), 'Quality gates failed; inspect analysis/quality.json')
    return runs, evidence


def calibrate_filters(runs):
    import numpy as np
    from defenses.rtc.spectral_direction import score_gram
    scores, pairs, ratios = [], [], []
    for run in runs:
        for row in run['rounds'][1:]:
            gram = np.asarray(json.loads(row['fit_rtc_r1_gram_json']), dtype=float)
            scores.extend(s for s in score_gram(gram) if s['valid'])
            norms = np.sqrt(np.maximum(np.diag(gram), 0))
            require((norms > 1e-12).all(), 'Degenerate clean updates; inspect calibration')
            cosine = gram / np.outer(norms, norms)
            pairs.extend(cosine[np.triu_indices(10, 1)].tolist())
            # R2 uses raw (unclipped) trainable norms exported by its observer.
            clients = [c for c in run['clients'] if integer(c['round']) == integer(row['round'])]
            raw = [float(c['rtc_r2_raw_norm']) for c in clients]
            ratios.extend(v/statistics.median(raw[:i]+raw[i+1:]) for i, v in enumerate(raw))
    quantile = lambda xs, q: sorted(xs)[int(q*(len(xs)-1))]
    require(scores and len(pairs) == 5400 and len(ratios) == 1200, 'Incomplete clean filter observations')
    require(all(math.isfinite(x) for x in pairs + ratios), 'Nonfinite clean filter observations')
    gap = quantile([s['gap'] for s in scores], .01)
    direction = read(ROOT / 'config/rtc_r1c_pairwise_calibration.json')
    direction.update(minimum_gap=gap, cosine_threshold=min(0., quantile([s['cosine'] for s in scores if s['gap'] >= gap], .01)),
        provenance=dict(dataset='mnist', clean_seeds=[40,41], rule='lower1% order statistics; no attack results'))
    direction['corroboration'].update(pairwise_threshold=min(-1e-12, quantile(pairs, .01)),
        provenance=dict(dataset='mnist', clean_pairs=len(pairs), clean_seeds=[40,41],
            negative_epsilon=1e-12, reason='Strictly negative threshold required by corroboration contract'))
    raw = read(ROOT / 'config/rtc_r2_raw_norm_calibration.json')
    raw.update(ratio_threshold=max(3., quantile(ratios, .99)),
        provenance=dict(dataset='mnist', clean_seeds=[40,41], clean_upper99_ratio=quantile(ratios, .99)))
    return direction, raw


def analyze(root, stage):
    lock = verify_lock(root, stage)
    runs, evidence = load_runs(root, stage, lock)
    if stage == 'calibration':
        from experiments import rtc_v3_formal_calibration as calibration
        from defenses.rtc.calibration import content_hash, CalibrationManifest
        from defenses.rtc.spectral_direction import load_calibration
        from defenses.rtc.raw_norm import load_calibration as load_raw
        # The calibrator consumes a single flat matrix. Copy immutable observations;
        # never rewrite or merge the original training files.
        flat = root / 'calibration_observations'; flat.mkdir(exist_ok=True)
        bootstrap = read(root / 'bootstrap.json')
        archive = flat / f'rtc_v3_semantic_collection_bootstrap_{bootstrap["content_hash"][:12]}.json'
        if archive.exists():
            require(read(archive) == bootstrap, 'Calibration archive changed')
        else:
            write(archive, bootstrap)
        specs = []
        for run in runs:
            cell = run['cell']; specs.append(cell['spec'])
            batch = root / stage / cell['batch']; rid = cell['run_id']
            for folder, names in [('rounds', [rid+'.csv']), ('raw', [rid+'_clients.csv', rid+'_config.json', rid+'_data_manifest.json'])]:
                (flat / folder).mkdir(exist_ok=True)
                for name in names:
                    source, target = batch / folder / name, flat / folder / name
                    if target.exists():
                        require(sha(source) == sha(target), 'Calibration copy changed')
                    else:
                        shutil.copyfile(source, target)
        args = calibration.parse_args(['--config', str(root/'runtime_config.yaml'), '--output', str(flat),
            '--bootstrap-manifest', str(root/'bootstrap.json'), '--batch-size', '96'])
        payload, validation = calibration.calibrate_manifest(args=args, specs=specs, inherit_parent_budget_floor=False)
        require(validation['passed'], 'Clean calibration validation failed')
        payload['metadata'].update(dataset='mnist', architecture='mnist_cnn',
            artifact_kind='rtc_m3_mnist_clean_calibrated', formal_evaluation_ready=False,
            warning='MNIST calibration only; independent clean gate and attack evaluation pending.')
        payload['content_hash'] = content_hash(payload)
        validation['content_hash'] = payload['content_hash']
        CalibrationManifest.load(payload)
        direction, raw = calibrate_filters(runs)
        for name, value in [('manifest',payload), ('direction',direction), ('raw_norm',raw), ('validation',validation)]:
            path = root / 'calibrated' / (name+'.json')
            if path.exists():
                require(read(path) == value, 'Refusing calibration drift')
            else:
                write(path, value)
        load_calibration(root / 'calibrated/direction.json'); load_raw(root / 'calibrated/raw_norm.json')
        write(root / 'calibrated/receipt.json', dict(passed=True, evidence=evidence,
            files={**evidence, **inventory([root/'calibrated'/f'{n}.json' for n in ('manifest','direction','raw_norm','validation')], root)}))
        return {'stage': stage, 'passed': True, 'next': 'Manually prepare validation; no training started'}
    summaries = [r['summary'] for r in runs]
    out = root / stage / 'analysis'; out.mkdir(exist_ok=True)
    with (out/'results.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0])); writer.writeheader(); writer.writerows(summaries)
    if stage == 'validation':
        m3 = next(r for r in summaries if r['defense'] == M3)
        fa = next(r for r in summaries if r['defense'] == 'fedavg')
        gates = {k+'_noninferior': 100*(m3[k]-fa[k]) >= -.2 for k in ('active_accuracy','final_accuracy','tail_accuracy')}
        gates['benign_union_cap_limit'] = m3['benign_union_cap_rate'] <= .01
        decision = dict(passed=all(gates.values()), checks=gates, summaries=summaries, evidence=evidence)
    else:
        # Record all conditions, including ineffective attacks and poor RTC outcomes.
        fedavg = {r['condition']: r for r in summaries if r['defense'] == 'fedavg'}
        clean_run = next(r for r in runs if r['summary']['condition'] == 'clean' and r['summary']['defense'] == 'fedavg')
        clean_active = statistics.mean(float(r['server_accuracy']) for r in clean_run['rounds'][11:])
        damage = {c: dict(fedavg_acc_drop_pp=None if v['active_accuracy'] is None else 100*(clean_active-v['active_accuracy']),
            execution_state=v['execution_state'],
            asr_mean=v['asr_mean'], note='Descriptive cross-condition comparison; different TrialPlans; not a strength-screen certificate')
            for c,v in fedavg.items() if c != 'clean'}
        decision = dict(passed=True, meaning='Data quality complete, not all-attack superiority or M3 promotion',
            training_units=len(summaries), seed_count=1, fedavg_damage=damage, evidence=evidence)
        lines = ['# MNIST full comparison', '', 'ACC/ASR are percentages. One seed; no significance claim.',
            'RTC-M3 mechanisms with MNIST clean calibration; no pooling with CIFAR results.', '']
        methods = lock['protocol']['defenses']
        for metric in ('active_accuracy','final_accuracy','tail_accuracy','asr_mean','asr_peak','asr_tail10','asr_final'):
            lines += ['## '+metric, '', '|Condition|'+'|'.join(methods)+'|', '|---|'+'---:|'*len(methods)]
            for condition in lock['protocol']['conditions']:
                matching = {r['defense']: r for r in summaries if r['condition'] == condition['name']}
                def display(d):
                    row = matching[d]
                    if row['execution_state'] == 'terminated_numerical':
                        return 'DIV'
                    return 'N/A' if row[metric] is None else f'{100*row[metric]:.3f}'
                lines.append('|'+condition['name']+'|'+'|'.join(display(d) for d in methods)+'|')
            lines.append('')
        (out/'comparison_tables.md').write_text('\n'.join(lines), encoding='utf-8')
    write(out / 'decision.json', decision)
    return decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('calibration','validation','evaluation'), default='calibration')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--analyze', action='store_true')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--client-num-gpus', type=float)
    parser.add_argument('--restart-incomplete', action='store_true')
    args = parser.parse_args()
    if (args.data_dir or args.client_num_gpus is not None) and (args.execute or args.analyze or args.stage != 'calibration'):
        parser.error('Data/GPU options only apply when preparing calibration')
    if args.restart_incomplete and not args.execute:
        parser.error('--restart-incomplete requires --execute')
    root = args.output.resolve()
    if args.execute:
        execute(root, args.stage, args.restart_incomplete)
    elif args.analyze:
        result = analyze(root, args.stage)
        print(json.dumps({k:v for k,v in result.items() if k != 'evidence'}, indent=2))
        if not result.get('passed', False):
            raise SystemExit(2)
    else:
        lock = prepare(root, args.stage, args.data_dir, args.client_num_gpus)
        print(f'Prepared {args.stage}: {len(lock["cells"])} units. WAITING_FOR_MANUAL_EXPERIMENT. No training executed.')


if __name__ == '__main__':
    main()
