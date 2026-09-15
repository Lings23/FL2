"""R3 trainable temporal evidence on accepted I12 parent. All training is manual."""
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import rtc_i12_bridge as prior
from experiments import rtc_i12_validation as validator
from defenses.rtc.temporal_observe import metadata, DIMENSION

PARENT = 'rtc_i12_combined'
OBSERVER = 'rtc_i12_temporal_observe'
MK = 'rtc_i12_multikrum'
OUTPUT = ROOT / 'logs/rtc_r3_temporal_observation'
PROTOCOL = ROOT / 'config/rtc_r3_temporal_observation.json'
RECEIPT = ROOT / 'config/rtc_i12_accepted.json'
MODES = {PARENT: ('cap', 'cap'), OBSERVER: ('cap', 'cap'), MK: ('observe', 'observe')}


def batches():
    return [('clean_seed103', 'none', 103, (PARENT, OBSERVER)),
            ('clean_seed104', 'none', 104, (PARENT, OBSERVER)),
            ('lie_seed44', 'lie', 44, (PARENT, OBSERVER, MK)),
            ('lie_seed45', 'lie', 45, (PARENT, OBSERVER, MK))]


def command(root, name, attack, seed, defenses, execute=False):
    return prior.command(root, name, attack, seed, defenses, execute=execute)


def verify_lock(root):
    lock = prior.read_json(root / 'r3t_lock.json')
    assert lock['protocol'] == prior.read_json(PROTOCOL)
    assert lock['accepted_parent'] == prior.read_json(RECEIPT)
    assert lock['sources'] == prior.sources(), 'Source drift; preserve old batch'
    assert lock['environment'] == prior.environment(), 'Use original training host/environment'
    for name, h in lock['artifacts'].items():
        assert prior.digest(root / name) == h, name
    assert {(c['batch'], c['attack'], c['seed'], c['defense']) for c in lock['cells']} == {
        (name, a, s, d) for name, a, s, ds in batches() for d in ds}
    assert len(lock['cells']) == lock['training_units'] == 10
    return lock


def prepare(root, data_dir=None):
    import yaml
    from attacks.spec import attack_source_hash
    from experiments import periodic_attack
    from experiments.run import parse_args
    if (root / 'r3t_lock.json').exists():
        lock = verify_lock(root)
        if data_dir is not None:
            assert str(Path(data_dir).resolve()) == lock['data_dir']
        return lock
    if root.exists() and any(root.iterdir()):
        raise ValueError('Nonempty unfrozen output; inspect before choosing a new directory')
    receipt = prior.read_json(RECEIPT)
    assert receipt['candidate_accepted'] and receipt['candidate'] == PARENT
    protocol = prior.read_json(PROTOCOL)
    assert protocol['training_units'] == 10 and protocol['observer_metadata'] == metadata()
    root.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((ROOT / 'config/config.yaml').read_text(encoding='utf-8'))
    data_dir = str(Path(data_dir or ROOT / 'data').resolve())
    cfg['dataset']['data_dir'] = data_dir
    (root / 'runtime_config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    prior.write(root / 'attack.freeze.json', {'schema_version': 'RTCByzantineAttackFreezeV1',
        'implementation_source_sha256': attack_source_hash(), 'selection_policy': 'Fixed LIE .5 temporal observation; no attack-strength tuning',
        'attacks': {'lie': {'strength_level': 'medium', 'parameters': {'lie_z': .5, 'coordinated_attack_knowledge': 'all_updates'}}}})
    cells = []
    for name, attack, seed, defenses in batches():
        argv = command(root, name, attack, seed, defenses)
        assert argv[-1] == '--dry-run'
        subprocess.run(argv, cwd=ROOT, check=True)
        args = parse_args(argv[3:])
        specs = prior.read_json(root / name / 'experiment_manifest.json')['specs']
        assert {s['defense'] for s in specs} == set(defenses) and len(specs) == len(defenses)
        assert len({s['trial_plan_hash'] for s in specs}) == 1
        configs = {}
        for s in specs:
            rid = periodic_attack.run_id(s)
            config = asdict(periodic_attack._build_spec_config(s, args, root / name))
            custom = config['security']['defense']['custom_params']
            assert (custom['spectral_direction_mode'], custom['raw_norm_mode']) == MODES[s['defense']]
            assert config['client'] == dict(local_epochs=5, batch_size=48, optimizer='sgd', learning_rate=.01,
                momentum=.9, weight_decay=.0001, lr_scheduler='cosine')
            assert config['federation']['num_rounds'] == 60 and config['federation']['deterministic_client_training']
            assert config['model']['architecture'] == 'resnet18' and not config['model']['pretrained']
            configs[s['defense']] = config
            prior.write(root / name / 'resolved' / (rid + '.json'), config)
            cells.append(dict(batch=name, attack=attack, seed=seed, defense=s['defense'], run_id=rid,
                **{k: s[k] for k in ('trial_plan_hash', 'attack_implementation_hash', 'attack_contract_hash')}))
        observed = copy.deepcopy(configs[OBSERVER])
        assert observed['security']['defense']['custom_params'].pop('temporal_residual_observe_only') is True
        assert observed == configs[PARENT], 'Observer must be the only configuration change'
    cal = prior.load_calibration(prior.CAL)
    raw = prior.raw_norm.load_calibration(prior.RAW_CAL)
    source_hashes = prior.sources()
    with zipfile.ZipFile(root / 'r3t_sources.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, expected in sorted(source_hashes.items()):
            content = (ROOT / name).read_bytes()
            assert hashlib.sha256(content).hexdigest() == expected, name
            archive.writestr(name, content)
    artifacts = {p.relative_to(root).as_posix(): prior.digest(p) for p in root.rglob('*') if p.is_file()}
    lock = dict(schema='RTCR3TemporalObservationV1', stage='R3-temporal-observe', training_units=10,
        reused_training_units=0, accepted_parent=receipt, protocol=protocol, sources=source_hashes,
        artifacts=artifacts, environment=prior.environment(), data_dir=data_dir, cells=cells,
        calibration=cal, calibration_hash=prior.calibration_hash(cal), raw_calibration=raw,
        raw_calibration_hash=prior.raw_norm.calibration_hash(raw))
    prior.write(root / 'r3t_lock.json', lock)
    return verify_lock(root)


def execute(root):
    lock = verify_lock(root)
    prior.ensure_no_training_process()
    for c in lock['cells']:
        b, rid = root / c['batch'], c['run_id']
        s = b / 'status' / (rid + '.json')
        if s.exists():
            value = prior.read_json(s)
            if value['state'] not in ('completed', 'completed_cached') or value['exit_code'] != 0 or value['last_round'] != 60:
                raise ValueError('Incomplete/failed run; manual inspection required, no automatic restart')
        elif any((b / 'raw').glob(rid + '*')):
            raise ValueError('Orphaned evidence; manual inspection required')
    for name, attack, seed, defenses in batches():
        verify_lock(root)
        subprocess.run(command(root, name, attack, seed, defenses, execute=True), cwd=ROOT, check=True)


def coherence(records):
    history, scores = {}, []
    for row in sorted(records, key=lambda r: (int(r['round']), r['cid'])):
        x = np.asarray(json.loads(row['rtc_r3t_sketch_json']), dtype=float)
        assert x.shape == (DIMENSION,) and np.isfinite(x).all()
        norm = float(row['rtc_r3t_residual_norm'])
        assert np.isfinite(norm) and norm >= 0
        assert int(row['rtc_r3t_trainable_dimension']) > 0
        pid = row['principal_id']
        h = history.setdefault(pid, [])
        length = float(np.linalg.norm(x))
        # Every visit occupies a slot. Zero residuals do not receive a direction.
        h.append(x / length if length > 1e-12 else None)
        del h[:-5]
        valid = len(h) == 5 and all(v is not None for v in h)
        value = float(np.linalg.norm(np.mean(h, axis=0))) if valid else None
        scores.append(dict(round=int(row['round']), cid=row['cid'], principal_id=pid,
                           attack_active=prior.truth(row['attack_active']), valid=valid, coherence=value))
    return scores


def analyze(root):
    lock = verify_lock(root)
    scope = dict(validator.verify_cell.__globals__, MODES=MODES, RTC=(PARENT, OBSERVER))
    verify = types.FunctionType(validator.verify_cell.__code__, scope)
    runs = [verify(root, c, lock) for c in lock['cells']]
    observations = []
    for name, attack, seed, defenses in batches():
        index = {r['cell']['defense']: r for r in runs if r['cell']['batch'] == name}
        a, b = index[PARENT], index[OBSERVER]
        for run in index.values():
            prior.paired(a, run)
        for x, y in zip(a['rounds'][1:], b['rounds'][1:]):
            for key in ('server_accuracy', 'server_loss', 'fit_aggregate_update_sketch_json', 'fit_defense_random_seed'):
                assert x[key] == y[key], ('Observer changed trajectory', name, x['round'], key)
            for key, value in metadata().items():
                assert str(y['fit_' + key]) == str(value)
        original = {(r['round'], r['cid']): r for r in a['clients']}
        for row in b['clients']:
            other = original[row['round'], row['cid']]
            for key in ('aggregation_weight', 'rtc_r1_q', 'rtc_r2_q', 'rtc_v3_cumulative_q_full'):
                assert row[key] == other[key], (name, row['round'], key)
        observations.append(dict(batch=name, attack=attack, seed=seed, records=coherence(b['clients'])))
    result = dict(stage='R3-temporal-observe', observation_quality_accepted=True, candidate_accepted=False,
        reason='Only inert evidence collected; no threshold or new cap is accepted',
        parent=PARENT, observations=observations, summaries=[prior.metrics(r) if r['cell']['defense'] != OBSERVER
            else prior.metrics({**r, 'cell': {**r['cell'], 'defense': PARENT}}) for r in runs],
        lock_sha256=prior.digest(root / 'r3t_lock.json'), final_goal_achieved=False)
    prior.write(root / 'analysis/observation.json', result)
    return result


def main():
    if not __debug__:
        raise RuntimeError('Do not disable assertions')
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group()
    group.add_argument('--execute', action='store_true')
    group.add_argument('--analyze', action='store_true')
    p.add_argument('--output', type=Path, default=OUTPUT)
    p.add_argument('--data-dir', type=Path)
    args = p.parse_args()
    if args.data_dir is not None and (args.execute or args.analyze):
        p.error('--data-dir only permitted during preparation')
    root = args.output.expanduser().resolve()
    result = execute(root) if args.execute else analyze(root) if args.analyze else prepare(root, args.data_dir)
    if result:
        print(json.dumps({k: result[k] for k in ('stage', 'training_units', 'observation_quality_accepted', 'candidate_accepted') if k in result}))


if __name__ == '__main__':
    main()
