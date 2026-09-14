"""I12 direction/norm bridge. Default prepares only; real runs require --execute."""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.rtc_i12_validation import BASE, DIRECTION, NORM, COMBINED, MK, RFA, RTC, MODES, verify_cell
from experiments.rtc_r0b_observation import read_json, digest, snapshot, truth
from experiments.rtc_r2_raw_norm import paired, summarize, canonical
from defenses.rtc.spectral_direction import load_calibration, calibration_hash
from defenses.rtc import raw_norm

OUTPUT = ROOT / 'logs/rtc_i12_bridge_v2'
PROTOCOL = ROOT / 'config/rtc_i12_protocol.json'
SEEDS = (44, 45)
CONDITIONS = (('none', RTC), ('sign_flip', (*RTC, MK)),
              ('gaussian_noise', (*RTC, MK, RFA)), ('lie', (*RTC, MK)))
CAL = ROOT / 'config/rtc_r1c_pairwise_calibration.json'
RAW_CAL = ROOT / 'config/rtc_r2_raw_norm_calibration.json'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8', newline='\n')


def batches():
    return [(f'{attack}_seed{seed}', attack, seed, ds) for seed in SEEDS for attack, ds in CONDITIONS]


def sources():
    result = snapshot()
    for folder in ('experiments', 'tests'):
        for path in (ROOT / folder).rglob('*'):
            if path.is_file() and (path.suffix == '.sh' or folder == 'tests' and path.suffix == '.py'):
                result[path.relative_to(ROOT).as_posix()] = digest(path)
    for name in ('requirements.txt', 'requirements-l20.txt',
                 'analysis/rtc_r3_diagnostics/replay.py'):
        result[name] = digest(ROOT / name)
    return result


def environment():
    import torch
    import numpy
    return {'os': platform.system(), 'machine': platform.machine(), 'python': platform.python_version(),
            'python_executable': str(Path(sys.executable).resolve()), 'torch': torch.__version__,
            'numpy': numpy.__version__, 'cuda': torch.version.cuda,
            'devices': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            'deterministic_client_training': True, 'pairing_mode': 'strict',
            'sampling_protocol': 'principal_uniform'}


def command(root, name, attack, seed, defenses, *, execute=False):
    args = [sys.executable, '-m', 'experiments.core.run', '--profile', 'rtc-byzantine',
            '--config', str(root / 'runtime_config.yaml'), '--attacks', attack,
            '--defenses', ','.join(defenses), '--seeds', str(seed),
            '--byzantine-attack-freeze', str(root / 'attack.freeze.json'),
            '--attack-start-round', '11', '--attack-end-round', '-1', '--malicious-fractions', '.3',
            '--rounds', '60', '--num-clients', '20', '--participation-rate', '.5', '--batch-size', '48',
            '--partition', 'iid', '--pairing-mode', 'strict', '--skip-clean', '--max-spec-retries', '0',
            '--ray-client-num-cpus', '1', '--ray-client-num-gpus', '.25',
            '--ray-object-store-memory-mb', '3072', '--ray-min-available-memory-mb', '10240',
            '--ray-memory-wait-seconds', '120', '--output', str(root / name)]
    return args if execute else args + ['--dry-run']


def verify_lock(root=OUTPUT):
    lock = read_json(root / 'i12_lock.json')
    if lock['sources'] != sources():
        raise ValueError('I12 source contract changed; preserve outputs, prepare a new batch after review')
    if lock['protocol'] != read_json(PROTOCOL) or lock['environment'] != environment():
        raise ValueError('I12 protocol/environment drift; no cross-platform cache reuse')
    for p, h in lock['artifacts'].items():
        if digest(root / p) != h:
            raise ValueError('I12 frozen artifact drift: ' + p)
    expected = {(name, attack, seed, d) for name, attack, seed, ds in batches() for d in ds}
    assert {(c['batch'], c['attack'], c['seed'], c['defense']) for c in lock['cells']} == expected
    assert len(lock['cells']) == lock['training_units'] == 40
    assert lock['parent'] == DIRECTION and lock['candidate'] == COMBINED
    return lock


def prepare(root=OUTPUT, data_dir=None):
    import yaml
    from attacks.spec import attack_source_hash
    from experiments import periodic_attack
    from experiments.run import parse_args
    if (root / 'i12_lock.json').exists():
        lock = verify_lock(root)
        if data_dir is not None and str(Path(data_dir).resolve()) != lock['data_dir']:
            raise ValueError('Cannot change data directory of frozen batch')
        return lock
    if root.exists() and any(root.iterdir()):
        raise ValueError('Output is not empty and has no lock; inspect interrupted preparation, use a new output directory')
    protocol = read_json(PROTOCOL)
    assert protocol['seeds'] == list(SEEDS) and protocol['training_units'] == 40
    assert protocol['parent'] == DIRECTION and protocol['candidate'] == COMBINED
    assert canonical(read_json(CAL)) == protocol['direction_calibration_canonical_sha256']
    assert canonical(read_json(RAW_CAL)) == protocol['raw_calibration_canonical_sha256']
    root.mkdir(parents=True, exist_ok=True)
    data_dir = Path(data_dir or ROOT / 'data').resolve()
    config = yaml.safe_load((ROOT / 'config/config.yaml').read_text(encoding='utf-8'))
    config['dataset']['data_dir'] = str(data_dir)
    (root / 'runtime_config.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8', newline='\n')
    write(root / 'attack.freeze.json', {'schema_version': 'RTCByzantineAttackFreezeV1',
          'implementation_source_sha256': attack_source_hash(),
          'selection_policy': 'I12 fixed direction/norm interaction, no attack-strength selection',
          'attacks': protocol['attacks']})
    cells = []
    for name, attack, seed, ds in batches():
        argv = command(root, name, attack, seed, ds)
        subprocess.run(argv, cwd=ROOT, check=True)
        specs = read_json(root / name / 'experiment_manifest.json')['specs']
        assert len(specs) == len(ds) and {s['defense'] for s in specs} == set(ds)
        assert len({s['trial_plan_hash'] for s in specs}) == 1
        args = parse_args(argv[3:])
        configs = {}
        for spec in specs:
            assert spec['seed'] == seed and spec['attack'] == attack
            rid = periodic_attack.run_id(spec)
            cfg = asdict(periodic_attack._build_spec_config(spec, args, root / name))
            custom = cfg['security']['defense']['custom_params']
            assert (custom['spectral_direction_mode'], custom['raw_norm_mode']) == MODES[spec['defense']]
            assert cfg['client'] == dict(local_epochs=5, batch_size=48, optimizer='sgd', learning_rate=.01,
                                         momentum=.9, weight_decay=.0001, lr_scheduler='cosine')
            assert cfg['federation']['num_rounds'] == 60 and cfg['federation']['deterministic_client_training']
            assert cfg['model']['architecture'] == 'resnet18' and not cfg['model']['pretrained']
            if spec['defense'] in RTC:
                for k, v in dict(semantic_intervention_risk_floor=.5, cumulative_q_cap_power=1,
                                 anchor_recycle_fraction=.51, anchor_recycle_weighting='accepted').items():
                    assert custom[k] == v
            if spec['defense'] == MK:
                assert cfg['security']['defense']['krum_num_malicious'] == 3
                assert cfg['security']['defense']['krum_num_to_select'] == 5
            configs[spec['defense']] = cfg
            write(root / name / 'resolved' / (rid + '.json'), cfg)
            cells.append({'batch': name, 'attack': attack, 'defense': spec['defense'], 'seed': seed, 'run_id': rid,
                          'parent': DIRECTION, 'modes': MODES[spec['defense']],
                          **{k: spec[k] for k in ('trial_plan_hash', 'attack_implementation_hash', 'attack_contract_hash')}})
        baseline = copy.deepcopy(configs[BASE])
        for d in RTC:
            cfg = copy.deepcopy(configs[d])
            for key in ('spectral_direction_mode', 'raw_norm_mode'):
                cfg['security']['defense']['custom_params'][key] = 'observe'
            assert cfg == baseline, f'Unexpected training difference: {d}'
    artifacts = {p.relative_to(root).as_posix(): digest(p) for p in root.rglob('*')
                 if p.is_file() and p.suffix in ('.json', '.csv', '.yaml')}
    cal, raw = load_calibration(CAL), raw_norm.load_calibration(RAW_CAL)
    lock = dict(schema='RTCI12BridgeV1', stage='I12', training_units=40, reused_training_units=0,
                parent=DIRECTION, candidate=COMBINED, protocol=protocol, cells=cells,
                data_dir=str(data_dir), environment=environment(), sources=sources(), artifacts=artifacts,
                calibration=cal, calibration_hash=calibration_hash(cal),
                raw_calibration=raw, raw_calibration_hash=raw_norm.calibration_hash(raw))
    write(root / 'i12_lock.json', lock)
    verify_lock(root)
    return lock


def ensure_no_training_process():
    import psutil
    for p in psutil.process_iter(['pid', 'name', 'cmdline']):
        if p.info['pid'] == os.getpid():
            continue
        name, argv = (p.info['name'] or '').lower(), p.info['cmdline'] or []
        if name.startswith('raylet') or name.startswith(('python', 'ray')) and any(
                a in ('main.py', '--execute') or a.endswith('/main.py') or a.endswith('\\main.py') for a in argv):
            raise ValueError(f'Related training process exists (PID {p.info["pid"]}); inspect manually')


def execute(root=OUTPUT):
    lock = verify_lock(root)
    ensure_no_training_process()
    for c in lock['cells']:
        batch, rid = root / c['batch'], c['run_id']
        status = batch / 'status' / (rid + '.json')
        if status.exists():
            s = read_json(status)
            if s['state'] not in ('completed', 'completed_cached') or s['exit_code'] != 0 or s['last_round'] != 60:
                raise ValueError('Incomplete/failed run: manual review required; no checkpoint resume or automatic restart')
        elif any((batch / 'raw').glob(rid + '*')):
            raise ValueError('Orphaned raw results: manual review required')
    for name, attack, seed, ds in batches():
        verify_lock(root)
        subprocess.run(command(root, name, attack, seed, ds, execute=True), cwd=ROOT, check=True)


def metrics(run):
    result = summarize(run)
    start = 1 if run['cell']['attack'] == 'none' else 11
    window = [c for c in run['clients'] if int(c['round']) >= start]
    good = [c for c in window if not truth(c['attack_active'])]
    bad = [c for c in window if truth(c['attack_active'])]
    dmode, nmode = MODES[run['cell']['defense']]
    union = lambda c: (dmode == 'cap' and truth(c['rtc_r1_flagged'])) or (nmode == 'cap' and truth(c['rtc_r2_flagged']))
    result.update(benign_union_flag_rate=sum(union(c) for c in good) / len(good),
                  malicious_union_flag_rate=sum(union(c) for c in bad) / len(bad) if bad else None,
                  direction_benign_flags=sum(truth(c['rtc_r1_flagged']) for c in good),
                  direction_malicious_flags=sum(truth(c['rtc_r1_flagged']) for c in bad),
                  first_direction_flag_round=min((int(c['round']) for c in bad if truth(c['rtc_r1_flagged'])), default=None))
    return result


def decide(runs, protocol):
    expected = {(a, seed, d) for _, a, seed, ds in batches() for d in ds}
    index = {(r['cell']['attack'], r['cell']['seed'], r['cell']['defense']): r for r in runs}
    assert set(index) == expected and len(runs) == len(expected), 'All preregistered cells required'
    summaries = [metrics(r) for r in runs]
    table = {(r['attack'], r['seed'], r['defense']): r for r in summaries}
    gates, standalone_checks, effects = {}, {}, []
    g = protocol['gates']
    for _, attack, seed, ds in batches():
        b, d, n, c = [table[attack, seed, method] for method in RTC]
        prefix = f'{attack}/seed{seed}'
        for method in ds:
            paired(index[attack, seed, BASE], index[attack, seed, method])
            if method in RTC:
                for ar, br in zip(index[attack, seed, BASE]['rounds'][1:], index[attack, seed, method]['rounds'][1:]):
                    assert ar['fit_defense_random_seed'] == br['fit_defense_random_seed']
                destination = standalone_checks if method == NORM else gates
                destination[f'{prefix}/{method}/benign_union'] = table[attack, seed, method]['benign_union_flag_rate'] <= g['benign_union_flag_rate_max']
        for key in ('active_accuracy', 'final_accuracy'):
            gates[f'{prefix}/combined_vs_parent/{key}'] = c[key] - d[key] >= g['accuracy_noninferiority_min']
            gates[f'{prefix}/parent_vs_b0/{key}'] = d[key] - b[key] >= g['accuracy_noninferiority_min']
            standalone_checks[f'{prefix}/norm_vs_b0/{key}'] = n[key] - b[key] >= g['accuracy_noninferiority_min']
            effects.append(dict(attack=attack, seed=seed, metric=key, direction=d[key]-b[key],
                                norm_alone=n[key]-b[key], norm_after_direction=c[key]-d[key],
                                interaction=(c[key]-d[key])-(n[key]-b[key]), total=c[key]-b[key]))
        if attack != 'none':
            gates[f'{prefix}/malicious_weight_regression'] = c['malicious_weight_per_round'] <= d['malicious_weight_per_round'] + g['malicious_weight_tolerance']
        if attack == 'sign_flip':
            for label, item in [('parent', d), ('combined', c)]:
                gates[f'{prefix}/{label}/retain_direction_gain'] = item['active_accuracy']-b['active_accuracy'] >= g['sign_active_gain_min']
                gates[f'{prefix}/{label}/sign_final'] = item['final_accuracy'] >= b['final_accuracy']
        if attack == 'gaussian_noise':
            standalone_checks[f'{prefix}/norm_suppression'] = n['malicious_weight_per_round'] <= b['malicious_weight_per_round'] * g['gaussian_weight_ratio_max'] and n['malicious_weight_per_round'] < b['malicious_weight_per_round']
            gates[f'{prefix}/combined_suppression'] = c['malicious_weight_per_round'] <= max(g['malicious_weight_zero_tolerance'], d['malicious_weight_per_round'] * g['gaussian_weight_ratio_max'])
    return dict(stage='I12', quality_accepted=True, candidate_accepted=all(gates.values()), checks=gates,
                summaries=summaries, effects=effects, standalone_checks=standalone_checks,
                standalone_norm_screen_accepted=all(standalone_checks.values()), parent=DIRECTION, candidate=COMBINED,
                scope='Engineering screen: IID seeds44/45 clean, Sign-flip, Gaussian, LIE .5 only; no significance claim',
                pending='DBA/scaling, Random-v2, LIE .25, Min-Max/Min-Sum, label reversal and final new-seed/non-IID validation',
                historical_r2_decision='Unchanged: original +0.2pp utility gate failed',
                final_goal_achieved=False)


def analyze(root=OUTPUT):
    lock = verify_lock(root)
    runs = [verify_cell(root, c, lock) for c in lock['cells']]
    result = decide(runs, lock['protocol'])
    index = {(r['cell']['attack'], r['cell']['seed'], r['cell']['defense']): r for r in runs}
    divergences = []
    for _, attack, seed, _ in batches():
        for old, new in ((BASE, DIRECTION), (BASE, NORM), (DIRECTION, COMBINED), (BASE, COMBINED)):
            a, b = index[attack, seed, old], index[attack, seed, new]
            original = {(c['round'], c['cid']): c for c in a['clients']}
            first = next((dict(round=int(c['round']), cid=c['cid']) for c in sorted(b['clients'], key=lambda c: (int(c['round']), c['cid']))
                          if abs(float(c['aggregation_weight']) - float(original[c['round'], c['cid']]['aggregation_weight'])) > 1e-10), None)
            sketch = next((int(x['round']) for x, y in zip(a['rounds'][1:], b['rounds'][1:])
                           if x['fit_aggregate_update_sketch_json'] != y['fit_aggregate_update_sketch_json']), None)
            matched = [(original[c['round'], c['cid']], c) for c in b['clients'] if int(c['round']) >= 11
                       and truth(c['attack_active']) and truth(c['rtc_r1_usable'])
                       and truth(original[c['round'], c['cid']]['rtc_r1_usable'])]
            divergences.append(dict(attack=attack, seed=seed, old=old, new=new, first_weight_difference=first,
                first_sketch_difference_round=sketch, matched_projection_records=len(matched),
                old_negative_proxy=sum(float(x['rtc_r1_negative_contribution']) for x, _ in matched)/50 if matched else None,
                new_negative_proxy=sum(float(y['rtc_r1_negative_contribution']) for _, y in matched)/50 if matched else None,
                proxy_scope='Same client/round with usable reference in both trajectories; different LOO axes, not net attack vector'))
    result['divergences_and_proxies'] = divergences
    result['source_lock_sha256'] = digest(root / 'i12_lock.json')
    result['evidence'] = {p.relative_to(root).as_posix(): digest(p) for p in root.glob('*/*/*')
                          if p.is_file() and p.suffix in ('.json', '.csv') and 'analysis' not in p.parts}
    write(root / 'analysis/decision.json', result)
    return result


def main():
    if not __debug__:
        raise RuntimeError('Optimization disables acceptance assertions; run without -O/PYTHONOPTIMIZE')
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--execute', action='store_true')
    group.add_argument('--analyze', action='store_true')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--data-dir', type=Path)
    args = parser.parse_args()
    root = args.output.expanduser().resolve()
    if args.data_dir is not None and (args.execute or args.analyze):
        parser.error('--data-dir is a preparation-only option; frozen paths cannot be changed at execution')
    result = analyze(root) if args.analyze else execute(root) if args.execute else prepare(root, args.data_dir)
    if result is not None:
        print(json.dumps({k: v for k, v in result.items() if k in ('stage', 'training_units', 'reused_training_units',
              'parent', 'candidate', 'candidate_accepted', 'quality_accepted', 'checks')}, indent=2))


if __name__ == '__main__':
    main()
