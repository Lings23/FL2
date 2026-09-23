"""Manual Multi-Krum supplement, executed with the exact archived M3 sources.

Preparation never trains. Results are a development-seed comparison, not a
new RTC version or a certificate of all-attack safety.
"""
import argparse
import copy
import csv
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import statistics
import subprocess
import sys
import types
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
PROTOCOL = ROOT / 'config/rtc_m3_multikrum_protocol.json'
PARENT = ROOT / 'logs/rtc_m3_all_attacks_seed42'
OUTPUT = ROOT / 'logs/rtc_m3_multikrum_seed42'
MK = 'rtc_i12_multikrum'
M3 = 'rtc_i12_eligibility_confirmed'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def source_members(parent, protocol):
    require(sha(parent / 'm3_lock.json') == protocol['parent_lock_sha256'], 'Wrong M3 parent lock')
    require(sha(parent / 'm3_sources.zip') == protocol['parent_sources_sha256'], 'Wrong M3 source archive')
    lock = read(parent / 'm3_lock.json')
    for name, expected in lock['artifacts'].items():
        require(sha(parent / name) == expected, 'M3 artifact drift: ' + name)
    with zipfile.ZipFile(parent / 'm3_sources.zip') as z:
        require(len(z.namelist()) == len(set(z.namelist())) and set(z.namelist()) == set(lock['sources']), 'Source member mismatch')
        members = {}
        for name, expected in lock['sources'].items():
            p = PurePosixPath(name)
            require(not p.is_absolute() and '..' not in p.parts and '\\' not in name and ':' not in name, 'Unsafe source member')
            content = z.read(name)
            require(hashlib.sha256(content).hexdigest() == expected, 'Source corruption: ' + name)
            members[name] = content
    return lock, members


def evidence_inventory(parent, lock):
    files = {}
    for c in lock['cells']:
        base = parent / c['batch']; rid = c['run_id']
        paths = [base / 'quality_gates.csv', base / 'status' / (rid + '.json'),
                 base / 'rounds' / (rid + '.csv'), *sorted((base / 'raw').glob(rid + '*.json')),
                 base / 'raw' / (rid + '_clients.csv')]
        for p in paths:
            require(p.is_file(), 'Missing M3 evidence: ' + str(p))
            files[p.relative_to(parent).as_posix()] = sha(p)
    return files


def verify_files(root, lock):
    require(sha(SCRIPT) == lock['controller_sha256'], 'Supplement controller changed; preserve frozen batch')
    require(read(PROTOCOL) == lock['protocol'], 'Supplement protocol changed')
    parent = Path(lock['parent_root'])
    parent_lock, _ = source_members(parent, lock['protocol'])
    require(evidence_inventory(parent, parent_lock) == lock['parent_evidence'], 'M3 evidence changed')
    for name, expected in lock['artifacts'].items():
        require(sha(root / name) == expected, 'Supplement artifact drift: ' + name)
    require(len(lock['cells']) == lock['training_units'] == 12, 'Expected twelve MK units')
    require({(c['batch'], c['attack'], c['seed'], c['defense']) for c in lock['cells']} ==
            {(c['name'], c['attack'], 42, MK) for c in parent_lock['protocol']['conditions']}, 'Wrong MK matrix')
    return parent_lock


def require_training_environment(actual, parent_environment):
    require(actual == parent_environment, 'Training requires the original M3 Linux/Python/CUDA/GPU environment; dry-run is allowed here')


def ensure_no_other_training(processes=None):
    import psutil
    processes = psutil.process_iter(['pid','name','cmdline']) if processes is None else processes
    for p in processes:
        info = p.info
        argv = info['cmdline'] or []
        if info['pid'] == os.getpid():
            continue
        # The waiting controller is our own launcher, not another experiment.
        if info['pid'] == os.getppid() and str(SCRIPT) in argv:
            continue
        name = (info['name'] or '').lower()
        busy = name.startswith('raylet') or name.startswith(('python','ray')) and any(
            a in ('main.py','--execute') or a.endswith('/main.py') or a.endswith('\\main.py') for a in argv)
        require(not busy, f'Related training process exists (PID {info["pid"]}); inspect manually')


def command(stage, root, condition, execute=False):
    args = stage.command(root, condition, execute=execute)
    args[args.index('--defenses') + 1] = MK
    return args


def check_config(parent, candidate):
    """Permit only the defense and location changes, never training changes."""
    a, b = copy.deepcopy(parent), copy.deepcopy(candidate)
    defense = b['security']['defense']
    require(defense['type'] == 'krum' and defense['krum_num_malicious'] == 3 and
            defense['krum_num_to_select'] == 5 and defense['enabled'], 'Expected Multi-Krum f3, select5')
    custom = defense['custom_params']
    require(custom['spectral_direction_mode'] == custom['raw_norm_mode'] == 'observe', 'MK observers must not cap')
    require(set(custom) == {'spectral_direction_mode', 'spectral_direction_calibration', 'raw_norm_mode', 'raw_norm_calibration'}, 'Unexpected MK mechanism')
    for cfg in (a, b):
        cfg['security']['defense'].pop('custom_params')
        cfg['security']['defense']['type'] = 'DEFENSE'
        cfg['security']['defense']['krum_num_to_select'] = 0
        cfg['project']['log_dir'] = 'OUTPUT'
        cfg['dataset']['data_dir'] = 'DATA'
        cfg['federation']['trial_plan_path'] = 'TRIAL'
    require(a == b, 'Training/evaluation configuration differs from frozen M3')


def validate_metrics(run, truth):
    """Independent ACC/label ASR reconstruction; triggered ASR has no image export."""
    for row in run['rounds']:
        matrix = json.loads(row['server_confusion_matrix_json'])
        require(len(matrix) == 10 and all(len(r) == 10 and sum(r) == 1000 for r in matrix), 'Invalid confusion matrix')
        require(all(type(x) is int and x >= 0 for r in matrix for x in r), 'Invalid confusion counts')
        require(abs(sum(matrix[i][i] for i in range(10))/10000 - float(row['server_accuracy'])) <= 1e-12, 'ACC mismatch')
        require(all(truth(row[k]) for k in ('server_model_state_valid','server_logits_valid','server_loss_valid')), 'Numerical validity failed')
        attack = run['cell']['attack']
        if attack in ('label_flip_targeted', 'label_flip_all_reverse'):
            success, total = (matrix[5][3], 1000) if attack == 'label_flip_targeted' else (sum(matrix[i][9-i] for i in range(10)), 10000)
            require(float(row['server_asr_total']) == total and abs(float(row['server_asr'])-success/total) <= 1e-12, 'Label ASR mismatch')
        elif attack in ('dba', 'scaling_backdoor'):
            value = float(row['server_asr'])
            require(truth(row['server_asr_valid']) and float(row['server_asr_total']) == 9000 and math.isfinite(value) and 0 <= value <= 1 and abs(value*9000-round(value*9000)) < 1e-8, 'Invalid triggered ASR')
            if attack == 'dba':
                require(value == float(row['server_dba_full_trigger_asr']), 'DBA full-trigger ASR mismatch')


def comparison(m3, mk, protocol):
    fields = ('active_accuracy','final_accuracy','tail_accuracy','asr_mean','asr_peak','asr_tail10','asr_final')
    delta = {k: None if m3[k] is None else 100*(m3[k]-mk[k]) for k in fields}
    gates = {'acc_mean_noninferior': delta['active_accuracy'] >= -protocol['acc_noninferiority_pp'],
             'acc_final_noninferior': delta['final_accuracy'] >= -protocol['acc_noninferiority_pp'],
             'm3_benign_cap_limit': m3['benign_union_flag_rate'] <= protocol['m3_benign_cap_limit']}
    for k in ('asr_mean','asr_peak','asr_tail10','asr_final'):
        if delta[k] is not None:
            gates[k + '_no_increase'] = delta[k] <= 1e-10
    return dict(condition=m3['condition'], m3=m3, multikrum=mk, delta_pp=delta,
                descriptive_engineering_checks=gates, statistical_superiority_established=False)


def worker(root, mode):
    """Only frozen imports are visible in this fresh subprocess."""
    source = root / 'frozen_source'
    sys.path.insert(0, str(source))
    os.chdir(source)
    from experiments import rtc_m3_all_attacks as stage
    from experiments import periodic_attack
    from experiments.run import parse_args
    from experiments import rtc_reference_memory_validation as validator
    from experiments.rtc_v3 import byzantine
    prior = stage.prior
    bootstrap = read(root / 'bootstrap.json')
    parent = Path(bootstrap['parent_root'])
    protocol = bootstrap['protocol']
    parent_lock, _ = source_members(parent, protocol)

    def mapped_read(path):
        p = Path(path)
        if p.is_file() and (p.resolve().is_relative_to(parent) or p.resolve().is_relative_to(root)):
            return read(p)
        s = p.as_posix()
        marker = '/logs/rtc_m3_all_attacks_seed42/'
        if marker in s:
            rel = s.split(marker, 1)[1]
            require(rel in parent_lock['artifacts'], 'Unfrozen parent path')
            return read(parent / rel)
        if '/config/' in s:
            rel = 'config/' + s.split('/config/', 1)[1]
            require(rel in parent_lock['sources'], 'Unfrozen calibration path')
            return read(source / rel)
        raise ValueError('Unmapped artifact path: ' + s)

    scope = dict(validator.verify_cell.__globals__, read_json=mapped_read, read_csv=stage.numeric_csv,
                 MODES={M3: ('cap','cap'), MK: ('observe','observe')}, RTC=(M3,))
    verify = types.FunctionType(validator.verify_cell.__code__, scope)

    def verified_run(batch_root, cell, lock):
        run = verify(batch_root, cell, lock)
        validate_metrics(run, prior.truth)
        base = batch_root / cell['batch']
        specs = read(base / 'experiment_manifest.json')['specs']
        require(len(specs) == 1 and specs[0]['defense'] == cell['defense'] and periodic_attack.run_id(specs[0]) == cell['run_id'], 'Manifest/run identity mismatch')
        imported = copy.deepcopy(specs)
        if cell['attack'] == 'dba':
            trial = next((base / 'trial_plans').glob('*.json'))
            require(read(trial)['trial_plan_hash'] == cell['trial_plan_hash'], 'DBA trial plan mismatch')
            imported[0]['trial_plan_path'] = str(trial)
        gates = byzantine.validate_attack_execution(imported, base / 'rounds')
        require(all(g['passed'] for g in gates), 'Recomputed attack execution failed')
        if cell['defense'] == MK:
            for r in run['rounds'][1:]:
                weights = [float(c['aggregation_weight']) for c in run['clients'] if c['round'] == r['round']]
                require(sum(w > 0 for w in weights) == 5 and all(abs(w-.2) < 1e-8 or w == 0 for w in weights), 'Multi-Krum must select five equal-weight clients')
        return run

    if mode == 'prepare':
        # The frozen attack hasher used str(relative_path), so path separators
        # enter its hash. Preserve the original Linux hash on the training host;
        # Windows gets an explicitly non-training preflight contract. No source
        # byte is rewritten and no freeze check is disabled.
        from attacks.spec import attack_source_hash
        linux_hash = hashlib.sha256()
        for name in ('attacks/spec.py','attacks/attack_client.py','attacks/coordinator.py',
                     'client/fl_client.py','strategies/fed_strategy.py','server/fl_server.py',
                     'utils/numerical_failure.py'):
            linux_hash.update(name.encode('utf-8')); linux_hash.update((source/name).read_bytes())
        for condition in parent_lock['protocol']['conditions']:
            path = root / 'freezes' / (condition['name']+'.json')
            freeze = read(path)
            require(freeze['implementation_source_sha256'] == linux_hash.hexdigest(), 'Archived attack sources do not reproduce M3 Linux hash')
            if platform.system() == 'Windows':
                freeze['implementation_source_sha256'] = attack_source_hash()
                freeze['preflight_only_parent_linux_source_sha256'] = linux_hash.hexdigest()
                write(path, freeze)
            else:
                require(attack_source_hash() == linux_hash.hexdigest(), 'Original attack source hash mismatch')
        parent_runs = [verified_run(parent, c, parent_lock) for c in parent_lock['cells']]
        for condition in parent_lock['protocol']['conditions']:
            args = command(stage, root, condition)
            require(args[-1] == '--dry-run', 'Preparation may not train')
            subprocess.run(args, cwd=source, check=True)
            specs = read(root / condition['name'] / 'experiment_manifest.json')['specs']
            require(len(specs) == 1 and specs[0]['defense'] == MK, 'Expected one MK cell')
            spec = specs[0]
            pc = next(c for c in parent_lock['cells'] if c['batch'] == condition['name'])
            for key in ('trial_plan_hash','attack_contract_hash','attack_implementation_hash'):
                if key == 'attack_implementation_hash' and platform.system() == 'Windows':
                    continue  # Preflight only; execution still requires exact Linux environment.
                require(spec[key] == pc[key], 'Parent contract mismatch: ' + key)
            cfg = asdict(periodic_attack._build_spec_config(spec, parse_args(args[3:]), root / condition['name']))
            check_config(read(parent / pc['batch'] / 'resolved' / (pc['run_id']+'.json')), cfg)
            rid = periodic_attack.run_id(spec)
            write(root / condition['name'] / 'resolved' / (rid+'.json'), cfg)
        cells = []
        for condition in parent_lock['protocol']['conditions']:
            spec = read(root / condition['name'] / 'experiment_manifest.json')['specs'][0]
            cells.append(dict(batch=condition['name'], run_id=periodic_attack.run_id(spec),
                **{k: spec[k] for k in ('attack','defense','seed','trial_plan_hash','attack_contract_hash','attack_implementation_hash')}))
        write(root / 'parent_review.json', dict(quality_accepted=True, runs=len(parent_runs),
              parent_lock_sha256=sha(parent / 'm3_lock.json'), summaries=[stage.metrics(r) for r in parent_runs]))
        env = prior.environment()
        write(root / 'prepared.json', dict(cells=cells, environment=env,
            training_host_compatible=env == parent_lock['environment']))
        return

    lock = read(root / 'multikrum_lock.json')
    verify_files(root, lock)
    if mode == 'execute':
        require_training_environment(prior.environment(), parent_lock['environment'])
        require(prior.environment() == lock['environment'], 'Preparation environment changed')
        ensure_no_other_training()
        stage.preflight(root, lock)
        # Audit every already-completed cell before allowing any new training.
        completed = set()
        for c in lock['cells']:
            if (root / c['batch'] / 'status' / (c['run_id']+'.json')).exists():
                verified_run(root, c, lock)
                completed.add(c['batch'])
        for condition in parent_lock['protocol']['conditions']:
            if condition['name'] not in completed:
                verify_files(root, lock)
                subprocess.run(command(stage, root, condition, execute=True), cwd=source, check=True)
        return

    require(mode == 'analyze', 'Unknown worker mode')
    require(lock['environment'] == parent_lock['environment'], 'Cannot compare cross-host training')
    pairs = []
    for c in lock['cells']:
        pc = next(x for x in parent_lock['cells'] if x['batch'] == c['batch'])
        a, b = verified_run(parent, pc, parent_lock), verified_run(root, c, lock)
        prior.paired(a, b)
        check_config(read(parent / pc['batch'] / 'resolved' / (pc['run_id']+'.json')),
                     read(root / c['batch'] / 'resolved' / (c['run_id']+'.json')))
        am, bm = stage.metrics(a), stage.metrics(b)
        bm['version'] = 'Multi-Krum'
        # Observer flags on MK are diagnostic, not a detector false-positive rate.
        bm['benign_union_flag_rate'] = None
        pairs.append(comparison(am, bm, protocol))
    out = root / 'analysis'; out.mkdir(exist_ok=True)
    decision = dict(quality_accepted=True, paired_conditions=12, new_training_units=12, reused_m3_units=12,
        research_snapshot_comparison_complete=True, broad_release_accepted=False, final_goal_achieved=False,
        all_attack_superiority_established=False, scope=protocol['scope'], pairs=pairs,
        limitations=protocol['limitations'], source_lock_sha256=sha(root / 'multikrum_lock.json'))
    write(out / 'decision.json', decision)
    fields = ('active_accuracy','final_accuracy','tail_accuracy','asr_mean','asr_peak','asr_tail10','asr_final')
    records = [dict(condition=p['condition'], method=method, **{k: p[method][k] for k in fields}) for p in pairs for method in ('m3','multikrum')]
    with (out / 'comparison.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['condition','method',*fields]); writer.writeheader(); writer.writerows(records)
    lines = ['# Frozen M3 vs Multi-Krum, seed42', '', 'Percentages; single development seed. No broad release or statistical superiority claim.', '',
             '| Condition | M3 mean/final ACC | MK mean/final ACC | M3 mean/peak ASR | MK mean/peak ASR |', '|---|---:|---:|---:|---:|']
    def fmt(a, keys):
        return ' / '.join('N/A' if a[k] is None else f'{100*a[k]:.4f}' for k in keys)
    for p in pairs:
        lines.append('| '+p['condition']+' | '+' | '.join(fmt(p[m], ks) for ks in (('active_accuracy','final_accuracy'),('asr_mean','asr_peak')) for m in ('m3','multikrum'))+' |')
    (out / 'comparison.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def launch_worker(root, mode):
    subprocess.run([sys.executable, '-B', str(SCRIPT), '--worker', mode, '--output', str(root)],
                   cwd=root / 'frozen_source', check=True)


def prepare(root, parent, data_dir):
    if (root / 'multikrum_lock.json').exists():
        lock = read(root / 'multikrum_lock.json'); verify_files(root, lock)
        require(str(parent) == lock['parent_root'], 'Parent path changed')
        if data_dir is not None:
            require(str(data_dir) == lock['data_dir'], 'Data path changed')
        return lock
    require(not root.exists() or not any(root.iterdir()), 'Nonempty unfrozen output; preserve and inspect before retry')
    protocol = read(PROTOCOL)
    parent_lock, members = source_members(parent, protocol)
    require(len(parent_lock['cells']) == 12 and protocol['training_units'] == 12, 'Wrong parent matrix')
    root.mkdir(parents=True, exist_ok=True)
    for name, content in members.items():
        dest = root / 'frozen_source' / name
        dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(content)
    data_dir = data_dir or ROOT / 'data'
    # YAML is data only. All training, attack and evaluation imports use the archive.
    import yaml
    cfg = yaml.safe_load((parent / 'runtime_config.yaml').read_text(encoding='utf-8'))
    cfg['dataset']['data_dir'] = str(data_dir)
    (root / 'runtime_config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8', newline='\n')
    shutil.copytree(parent / 'freezes', root / 'freezes')
    write(root / 'bootstrap.json', dict(parent_root=str(parent), protocol=protocol))
    launch_worker(root, 'prepare')
    prepared = read(root / 'prepared.json')
    artifacts = {p.relative_to(root).as_posix(): sha(p) for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    lock = dict(schema='RTCM3MultiKrumSupplementV1', stage='M3-MultiKrum-all-attacks', protocol=protocol,
        training_units=12, reused_training_units=12, parent_root=str(parent), data_dir=str(data_dir),
        parent_evidence=evidence_inventory(parent, parent_lock), controller_sha256=sha(SCRIPT),
        artifacts=artifacts, calibration=parent_lock['calibration'], calibration_hash=parent_lock['calibration_hash'],
        raw_calibration=parent_lock['raw_calibration'], raw_calibration_hash=parent_lock['raw_calibration_hash'], **prepared)
    write(root / 'multikrum_lock.json', lock)
    verify_files(root, lock)
    return lock


def main():
    require(__debug__, 'Python assertions must remain enabled')
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--execute', action='store_true'); modes.add_argument('--analyze', action='store_true')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--parent', type=Path, default=PARENT)
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--worker', choices=('prepare','execute','analyze'), help=argparse.SUPPRESS)
    args = parser.parse_args(); root = args.output.expanduser().resolve()
    if args.worker:
        worker(root, args.worker); return
    if args.data_dir is not None and (args.execute or args.analyze):
        parser.error('--data-dir is preparation-only')
    if args.execute or args.analyze:
        lock = read(root / 'multikrum_lock.json'); verify_files(root, lock)
        launch_worker(root, 'execute' if args.execute else 'analyze')
    else:
        lock = prepare(root, args.parent.expanduser().resolve(), args.data_dir.expanduser().resolve() if args.data_dir else None)
        print(json.dumps({k: lock[k] for k in ('stage','training_units','reused_training_units','training_host_compatible')}))


if __name__ == '__main__':
    main()
