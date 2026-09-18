"""Frozen M3-only descriptive evaluation. Default prepares; training requires manual --execute."""
import argparse
import copy
import csv
from dataclasses import asdict
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import types
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import rtc_i12_bridge as prior
from experiments import rtc_g2_integration as recipe
from experiments import rtc_reference_memory_validation as validator

OUTPUT = ROOT / 'logs/rtc_m3_all_attacks_seed42'
PROTOCOL = ROOT / 'config/rtc_m3_all_attacks_protocol.json'
RECEIPT = ROOT / 'config/rtc_reference_memory_accepted.json'
M3 = 'rtc_i12_eligibility_confirmed'
TARGETED = {'label_flip_targeted', 'label_flip_all_reverse', 'scaling_backdoor', 'dba'}
INTEGER_FIELDS = {'round', 'rtc_r2_reference_count', 'rtc_r1_corroboration_votes',
    'rtc_r1_corroboration_required', 'rtc_r1e_known_count', 'rtc_r1e_eligible_count',
    'rtc_r1e_required', 'fit_rtc_r1e_quorum_numerator', 'fit_rtc_r1e_quorum_denominator'}


def numeric_csv(path):
    """Allow CSV 2.0 integers without silently truncating fractions or nonfinite values."""
    rows = validator.read_csv(path)
    for row in rows:
        for key in INTEGER_FIELDS.intersection(row):
            if row[key] == '':
                continue
            value = Decimal(row[key])
            if not value.is_finite() or value != value.to_integral_value():
                raise ValueError(f'Nonintegral metadata {key}: {row[key]}')
            row[key] = str(int(value))
    return rows


def conditions():
    return prior.read_json(PROTOCOL)['conditions']


def command(root, condition, execute=False):
    argv = recipe.command(root, condition['name'], condition['attack'], 42, (M3,), execute)
    argv[argv.index('--byzantine-attack-freeze') + 1] = str(root / 'freezes' / (condition['name'] + '.json'))
    if condition['attack'] == 'random_noise':
        argv.insert(argv.index('--dry-run') if '--dry-run' in argv else len(argv), '--byzantine-evaluation-only')
    return argv


def normalized_custom(custom):
    result = copy.deepcopy(custom)
    for k, v in result.items():
        if isinstance(v, str) and '/config/' in v.replace('\\', '/'):
            result[k] = 'config/' + v.replace('\\', '/').split('/config/', 1)[1]
    return result


def verify_config(cfg, protocol):
    assert cfg['client'] == protocol['training']
    assert cfg['ray']['client_num_gpus'] == .125
    assert cfg['dataset']['name'] == 'cifar10' and cfg['dataset']['partition'] == 'iid'
    assert cfg['model']['architecture'] == 'resnet18' and not cfg['model']['pretrained']
    assert cfg['federation']['num_rounds'] == 60 and cfg['federation']['num_clients'] == 20
    assert cfg['federation']['deterministic_client_training']
    assert normalized_custom(cfg['security']['defense']['custom_params']) == protocol['accepted_custom_params'], 'M3 mechanism drift'


def verify_lock(root):
    lock = prior.read_json(root / 'm3_lock.json')
    assert lock['protocol'] == prior.read_json(PROTOCOL)
    assert lock['accepted_parent'] == prior.read_json(RECEIPT)
    assert prior.canonical(prior.read_json(RECEIPT)) == lock['protocol']['accepted_parent_receipt_canonical_sha256']
    for path, expected in lock['protocol']['accepted_calibration_canonical_sha256'].items():
        assert prior.canonical(prior.read_json(ROOT / path)) == expected, 'Accepted M3 calibration changed: ' + path
    assert lock['sources'] == prior.sources(), 'Source drift; retain frozen batch and inspect'
    assert lock['environment'] == prior.environment(), 'Use the original training host/environment'
    for name, h in lock['artifacts'].items():
        assert prior.digest(root / name) == h, name
    assert len(lock['cells']) == lock['training_units'] == 12
    assert {(c['batch'], c['attack'], c['seed'], c['defense']) for c in lock['cells']} == {
        (c['name'], c['attack'], 42, M3) for c in conditions()}
    return lock


def prepare(root, data_dir=None):
    import yaml
    from attacks.spec import attack_source_hash
    from experiments import periodic_attack
    from experiments.run import parse_args
    if (root / 'm3_lock.json').exists():
        lock = verify_lock(root)
        if data_dir is not None:
            assert str(Path(data_dir).resolve()) == lock['data_dir']
        return lock
    if root.exists() and any(root.iterdir()):
        raise ValueError('Nonempty unfrozen output; preserve it and choose a new directory')
    protocol = prior.read_json(PROTOCOL)
    receipt = prior.read_json(RECEIPT)
    assert receipt['candidate_accepted'] and receipt['candidate'] == M3 and receipt['version'] == 'M3'
    assert protocol['seeds'] == [42] and protocol['training_units'] == len(conditions()) == 12
    assert protocol['defense'] == M3 and protocol['training'] == recipe.CLIENT_CONTRACT
    for path, expected in protocol['accepted_calibration_canonical_sha256'].items():
        assert prior.canonical(prior.read_json(ROOT / path)) == expected, 'Accepted M3 calibration changed: ' + path
    root.mkdir(parents=True, exist_ok=True)
    data_dir = str(Path(data_dir or ROOT / 'data').resolve())
    base = yaml.safe_load((ROOT / 'config/config.yaml').read_text(encoding='utf-8'))
    cfg = recipe.runtime_config(base, data_dir)
    cfg['ray']['client_num_gpus'] = .125
    cfg['dataset'].update(name='cifar10', partition='iid')
    cfg['model'].update(architecture='resnet18', pretrained=False)
    (root / 'runtime_config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    prior.write(root / 'preparation_client_parameters.json', dict(source_client=base['client'], registered_client=protocol['training']))
    cells = []
    for condition in conditions():
        attack = condition['attack']
        entry = copy.deepcopy(protocol['attacks'].get(attack, {}))
        if 'parameters' in condition:
            entry['parameters'] = condition['parameters']
        prior.write(root / 'freezes' / (condition['name'] + '.json'), dict(
            schema_version='RTCByzantineAttackFreezeV1', implementation_source_sha256=attack_source_hash(),
            purpose='fixed_strength_evaluation_only', selection_policy='Fixed predeclared parameters; no selection on M3 results',
            attacks={} if attack == 'none' else {attack: entry}))
        argv = command(root, condition)
        assert argv[-1] == '--dry-run' and argv[argv.index('--defenses') + 1] == M3 and '--skip-clean' in argv
        subprocess.run(argv, cwd=ROOT, check=True)
        specs = prior.read_json(root / condition['name'] / 'experiment_manifest.json')['specs']
        assert len(specs) == 1 and specs[0]['defense'] == M3
        spec = specs[0]
        assert spec['seed'] == 42 and spec['attack'] == attack
        if attack == 'random_noise':
            assert spec['attack_parameter_status'] == 'unverified_evaluation_only'
        args = parse_args(argv[3:])
        resolved = asdict(periodic_attack._build_spec_config(spec, args, root / condition['name']))
        verify_config(resolved, protocol)
        rid = periodic_attack.run_id(spec)
        prior.write(root / condition['name'] / 'resolved' / (rid + '.json'), resolved)
        cells.append(dict(batch=condition['name'], attack=attack, defense=M3, seed=42, run_id=rid,
            **{k: spec[k] for k in ('trial_plan_hash', 'attack_implementation_hash', 'attack_contract_hash')}))
    sources = prior.sources()
    with zipfile.ZipFile(root / 'm3_sources.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, expected in sorted(sources.items()):
            content = (ROOT / name).read_bytes()
            assert hashlib.sha256(content).hexdigest() == expected
            archive.writestr(name, content)
    artifacts = {p.relative_to(root).as_posix(): prior.digest(p) for p in root.rglob('*') if p.is_file()}
    cal = prior.load_calibration(prior.CAL)
    raw = prior.raw_norm.load_calibration(prior.RAW_CAL)
    prior.write(root / 'm3_lock.json', dict(schema='RTCM3EvaluationLockV1', stage=protocol['stage'],
        training_units=12, reused_training_units=0, protocol=protocol, accepted_parent=receipt, cells=cells,
        sources=sources, artifacts=artifacts, environment=prior.environment(), data_dir=data_dir,
        calibration=cal, calibration_hash=prior.calibration_hash(cal),
        raw_calibration=raw, raw_calibration_hash=prior.raw_norm.calibration_hash(raw)))
    return verify_lock(root)


def preflight(root, lock):
    for c in lock['cells']:
        batch, rid = root / c['batch'], c['run_id']
        status = batch / 'status' / (rid + '.json')
        if status.exists():
            s = prior.read_json(status)
            if s['state'] not in ('completed', 'completed_cached') or s['exit_code'] != 0 or s['last_round'] != 60:
                raise ValueError('Incomplete/failed run: inspect manually; no automatic restart or mid-round resume')
        elif any((batch / 'raw').glob(rid + '*')) or (batch / 'rounds' / (rid + '.csv')).exists():
            raise ValueError('Orphaned results: inspect manually')


def execute(root):
    lock = verify_lock(root)
    prior.ensure_no_training_process()
    preflight(root, lock)
    for condition in conditions():
        verify_lock(root)
        subprocess.run(command(root, condition, execute=True), cwd=ROOT, check=True)


def verify_run(root, cell, lock):
    verify_config(prior.read_json(root / cell['batch'] / 'resolved' / (cell['run_id'] + '.json')), lock['protocol'])
    scope = dict(validator.verify_cell.__globals__, MODES={M3: ('cap', 'cap')}, RTC=(M3,), read_csv=numeric_csv)
    verify = types.FunctionType(validator.verify_cell.__code__, scope)
    return verify(root, cell, lock)


def metrics(run):
    result = prior.summarize(run)
    result.pop('asr'); result.pop('asr_reason')
    targeted = run['cell']['attack'] in TARGETED
    start = 1 if run['cell']['attack'] == 'none' else 11
    rows = [r for r in run['rounds'] if int(r['round']) >= start]
    for r in run['rounds']:
        assert 0 <= float(r['server_accuracy']) <= 1
    good = [c for c in run['clients'] if int(c['round']) >= start and not prior.truth(c['attack_active'])]
    flags = sum(float(c['rtc_r1_q']) < 1 or float(c['rtc_r2_q']) < 1 for c in good)
    result.update(version='M3', condition=run['cell']['batch'], seed_count=1,
        evidence_scope='single_seed_descriptive', strength_validation='unverified' if run['cell']['attack']=='random_noise' else 'inherited_fixed_parameters',
        benign_union_flags=flags, benign_records=len(good), benign_union_flag_rate=flags/len(good),
        tail_accuracy=statistics.mean(float(r['server_accuracy']) for r in rows if int(r['round']) >= 51),
        asr_mean=None, asr_peak=None, asr_tail10=None, asr_final=None,
        asr_scope='source-class 5 to 3' if run['cell']['attack']=='label_flip_targeted' else
            'all test examples mapped y to 9-y' if run['cell']['attack']=='label_flip_all_reverse' else
            'triggered non-target examples' if targeted else 'not_applicable')
    if targeted:
        values = [float(r['server_asr']) for r in rows]
        assert len(values) == 50 and all(math.isfinite(x) and 0 <= x <= 1 for x in values)
        result.update(asr_mean=statistics.mean(values), asr_peak=max(values),
            asr_tail10=statistics.mean(values[-10:]), asr_final=values[-1])
    return result


def write_table(root, summaries):
    out = root / 'analysis'; out.mkdir(exist_ok=True)
    columns = ('condition', 'version', 'seed', 'active_accuracy', 'final_accuracy', 'tail_accuracy',
        'asr_mean', 'asr_peak', 'asr_tail10', 'asr_final', 'benign_union_flag_rate',
        'malicious_weight_per_round', 'rounds_exceeding_f3', 'strength_validation', 'run_id')
    with (out / 'rtc_m3_results.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader(); writer.writerows(summaries)
    text = ['# M3 / RTC, seed42', '', 'Single-seed descriptive evaluation. Values below are percentages. '
        'No comparator results have been merged. Random-v2 strength is unverified.', '',
        '| Condition | Mean ACC | Final ACC | Tail10 ACC | Mean ASR | Peak ASR | Tail10 ASR | Final ASR |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        vals = [s[k] for k in ('active_accuracy','final_accuracy','tail_accuracy','asr_mean','asr_peak','asr_tail10','asr_final')]
        text.append('| ' + s['condition'] + ' | ' + ' | '.join('N/A' if v is None else f'{100*v:.4f}' for v in vals) + ' |')
    (out / 'rtc_m3_table.md').write_text('\n'.join(text) + '\n', encoding='utf-8')


def analyze(root):
    lock = verify_lock(root)
    runs = [verify_run(root, c, lock) for c in lock['cells']]
    summaries = [metrics(r) for r in runs]
    # Accuracy, ASR, and false positives are outcomes, not a reason to drop a cell.
    result = dict(stage=lock['stage'], quality_accepted=True, training_units=12,
        all_attack_superiority_established=False, final_goal_achieved=False,
        source_lock_sha256=prior.digest(root / 'm3_lock.json'), summaries=summaries,
        comparator_results_merged=0, scope=lock['protocol']['scope'],
        evidence={p.relative_to(root).as_posix(): prior.digest(p) for p in root.glob('*/*/*')
            if p.is_file() and p.suffix in ('.json', '.csv')})
    write_table(root, summaries)
    prior.write(root / 'analysis/decision.json', result)
    return result


def main():
    if not __debug__:
        raise RuntimeError('Assertions must remain enabled')
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group()
    g.add_argument('--execute', action='store_true'); g.add_argument('--analyze', action='store_true')
    p.add_argument('--output', type=Path, default=OUTPUT); p.add_argument('--data-dir', type=Path)
    args = p.parse_args()
    if args.data_dir is not None and (args.execute or args.analyze):
        p.error('--data-dir is preparation-only')
    root = args.output.expanduser().resolve()
    result = execute(root) if args.execute else analyze(root) if args.analyze else prepare(root, args.data_dir)
    if result:
        print(json.dumps({k: result[k] for k in ('stage','training_units','quality_accepted') if k in result}))


if __name__ == '__main__':
    main()
