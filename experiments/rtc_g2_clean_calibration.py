"""G2 clean calibration with inert lower-tail observation on accepted M3. All training is manual."""
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
from experiments import rtc_reference_memory_validation as validator
from defenses.rtc import reference_eligibility
import statistics
import math

BASE = 'rtc_i12_eligibility_confirmed'
PARENT = BASE
OBSERVER = 'rtc_i12_confirmed_lower_observe'
MK = 'rtc_i12_multikrum'
RFA = 'rtc_i12_rfa'
OUTPUT = ROOT / 'logs/rtc_g2_clean_calibration'
PROTOCOL = ROOT / 'config/rtc_g2_clean_calibration_protocol.json'
RECEIPT = ROOT / 'config/rtc_reference_memory_accepted.json'
MODES = {BASE: ('cap', 'cap'), PARENT: ('cap', 'cap'), OBSERVER: ('cap', 'cap'), MK: ('observe', 'observe'), RFA: ('observe', 'observe')}
CLIENT_CONTRACT = dict(local_epochs=5, batch_size=96, optimizer='sgd', learning_rate=.01,
    momentum=.9, weight_decay=.0001, lr_scheduler='cosine')


def runtime_config(base, data_dir):
    """Pin the registered training recipe without editing the host's master config."""
    cfg = copy.deepcopy(base)
    cfg['dataset']['data_dir'] = str(data_dir)
    cfg.setdefault('client', {}).update(CLIENT_CONTRACT)
    return cfg


def verify_client_contract(actual):
    differences = {key: {'expected': CLIENT_CONTRACT.get(key), 'actual': actual.get(key),
        'actual_type': type(actual.get(key)).__name__}
        for key in sorted(set(CLIENT_CONTRACT) | set(actual))
        if key not in actual or key not in CLIENT_CONTRACT or actual[key] != CLIENT_CONTRACT[key]}
    if differences:
        raise ValueError('G2 clean training contract mismatch: ' + json.dumps(differences, sort_keys=True))


def batches():
    return [(f'none_seed{seed}', 'none', seed, (BASE, OBSERVER)) for seed in (106,107)]


def command(root, name, attack, seed, defenses, execute=False):
    inherited = prior.command(root, name, attack, seed, defenses, execute=execute)
    # _build_spec_config gives the CLI precedence over runtime_config.yaml.
    # Shared host command edits must not change this registered clean recipe.
    args = []
    index = 0
    while index < len(inherited):
        token = inherited[index]
        if token == '--batch-size':
            if index + 1 >= len(inherited) or inherited[index + 1].startswith('--'):
                raise ValueError('Inherited --batch-size is missing its value')
            index += 2
        elif token.startswith('--batch-size='):
            index += 1
        else:
            args.append(token)
            index += 1
    position = args.index('--dry-run') if '--dry-run' in args else len(args)
    args[position:position] = ['--batch-size', str(CLIENT_CONTRACT['batch_size'])]
    args[args.index('--ray-client-num-gpus') + 1] = '.125'
    return args


def verify_lock(root):
    lock = prior.read_json(root / 'g2c_lock.json')
    assert lock['protocol'] == prior.read_json(PROTOCOL)
    assert lock['accepted_parent'] == prior.read_json(RECEIPT)
    assert lock['sources'] == prior.sources(), 'Source drift; preserve old batch'
    assert lock['environment'] == prior.environment(), 'Use original training host/environment'
    for name, h in lock['artifacts'].items():
        assert prior.digest(root / name) == h, name
    assert {(c['batch'], c['attack'], c['seed'], c['defense']) for c in lock['cells']} == {
        (name, a, s, d) for name, a, s, ds in batches() for d in ds}
    assert len(lock['cells']) == lock['training_units'] == 4
    return lock


def prepare(root, data_dir=None):
    import yaml
    from attacks.spec import attack_source_hash
    from experiments import periodic_attack
    from experiments.run import parse_args
    if (root / 'g2c_lock.json').exists():
        lock = verify_lock(root)
        if data_dir is not None:
            assert str(Path(data_dir).resolve()) == lock['data_dir']
        return lock
    if root.exists() and any(root.iterdir()):
        raise ValueError('Nonempty unfrozen output; inspect before choosing a new directory')
    receipt = prior.read_json(RECEIPT)
    assert receipt['candidate_accepted'] and receipt['candidate'] == BASE
    protocol = prior.read_json(PROTOCOL)
    assert protocol['training_units'] == 4
    root.mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load((ROOT / 'config/config.yaml').read_text(encoding='utf-8'))
    data_dir = str(Path(data_dir or ROOT / 'data').resolve())
    cfg = runtime_config(base, data_dir)
    cfg.setdefault('ray', {})['client_num_gpus'] = .125
    (root / 'runtime_config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    prior.write(root / 'preparation_client_parameters.json', {
        'source': 'config/config.yaml', 'source_client': base.get('client', {}),
        'registered_client': CLIENT_CONTRACT,
        'registered_ray_client_num_gpus': .125,
        'reason': 'Pin batch96 and GPU quota .125, matching the returned H2b execution environment.'})
    from config.config_loader import load_config
    verify_client_contract(asdict(load_config(root / 'runtime_config.yaml').client))
    prior.write(root / 'attack.freeze.json', {'schema_version': 'RTCByzantineAttackFreezeV1',
        'implementation_source_sha256': attack_source_hash(), 'selection_policy': 'Clean-only calibration on accepted M3; no attacks or threshold tuning on holdout',
        'attacks': protocol['attacks']})
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
            verify_client_contract(config['client'])
            assert config['federation']['num_rounds'] == 60 and config['federation']['deterministic_client_training']
            assert config['model']['architecture'] == 'resnet18' and not config['model']['pretrained']
            configs[s['defense']] = config
            prior.write(root / name / 'resolved' / (rid + '.json'), config)
            cells.append(dict(batch=name, attack=attack, seed=seed, defense=s['defense'], run_id=rid,
                **{k: s[k] for k in ('trial_plan_hash', 'attack_implementation_hash', 'attack_contract_hash')}))
        observed = copy.deepcopy(configs[OBSERVER])
        custom = observed['security']['defense']['custom_params']
        assert custom.pop('lower_tail_mode') == 'observe'
        assert custom.pop('lower_tail_calibration') == 'config/rtc_r3_lower_tail_calibration.json'
        assert observed == configs[BASE], 'Only an inert observer may differ'
        assert configs[BASE]['ray']['client_num_gpus'] == .125
    cal = prior.load_calibration(prior.CAL)
    raw = prior.raw_norm.load_calibration(prior.RAW_CAL)
    source_hashes = prior.sources()
    with zipfile.ZipFile(root / 'g2c_sources.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, expected in sorted(source_hashes.items()):
            content = (ROOT / name).read_bytes()
            assert hashlib.sha256(content).hexdigest() == expected, name
            archive.writestr(name, content)
    artifacts = {p.relative_to(root).as_posix(): prior.digest(p) for p in root.rglob('*') if p.is_file()}
    lock = dict(schema='RTCG2CleanCalibrationV1', stage='G2-clean-calibration', training_units=4,
        reused_training_units=0, accepted_parent=receipt, protocol=protocol, sources=source_hashes,
        artifacts=artifacts, environment=prior.environment(), data_dir=data_dir, cells=cells,
        calibration=cal, calibration_hash=prior.calibration_hash(cal), raw_calibration=raw,
        raw_calibration_hash=prior.raw_norm.calibration_hash(raw))
    prior.write(root / 'g2c_lock.json', lock)
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


def replay_filtered(run, threshold):
    from defenses.rtc.filtered_lower_tail import FilteredLowerTailEvidence
    from defenses.rtc.lower_tail import VERSION
    c = dict(version=VERSION,ratio_threshold=threshold,min_peers=9,required_visits=2,
             cap_multiplier=0,provenance={})
    scorer = FilteredLowerTailEvidence(c)
    records = []
    for rnd in range(1,61):
        clients = [c for c in run['clients'] if int(c['round']) == rnd]
        rows, transition = scorer.prepare(
            [float(c['rtc_r3l_residual_norm']) for c in clients],
            [c['principal_id'] for c in clients],
            [float(c['rtc_r1_nominal_mass']) for c in clients], 'observe',
            [prior.truth(c['rtc_r2_flagged']) for c in clients])
        scorer.commit(transition)
        records.extend(dict(round=rnd,cid=c['cid'],**row,
            union=bool(row['rtc_r3l_flagged'] or float(c['rtc_r1_q'])<1 or float(c['rtc_r2_q'])<1))
            for c,row in zip(clients,rows))
    return records


def decide(runs, protocol):
    index = {(r['cell']['seed'],r['cell']['defense']):r for r in runs}
    assert len(runs)==4 and set(index)=={(s,d) for _,_,s,ds in batches() for d in ds}
    for seed in (106,107):
        a,b = index[seed,BASE],index[seed,OBSERVER]
        prior.paired(a,b)
        for x,y in zip(a['rounds'],b['rounds']):
            for key in ('server_accuracy','server_loss','fit_aggregate_update_sketch_json'):
                assert x[key] == y[key], (seed,key,x['round'])
            if int(x['round']) > 0:
                assert x['fit_defense_random_seed'] == y['fit_defense_random_seed']
        ca={(c['round'],c['cid']):c for c in a['clients']}
        for y in b['clients']:
            x=ca[y['round'],y['cid']]
            for key in ('aggregation_weight','rtc_r1_q','rtc_r2_q'):
                assert x[key] == y[key], (seed,key,y['round'])
    calibration = replay_filtered(index[106,OBSERVER], .5)
    ratios = [r['rtc_r3l_ratio'] for r in calibration if r['rtc_r3l_valid']]
    assert len(ratios)==600 and all(math.isfinite(v) and v>0 for v in ratios)
    threshold = min(ratios)
    assert 0 < threshold < 1, 'Invalid clean threshold; stop without retuning'
    holdout = replay_filtered(index[107,OBSERVER], threshold)
    flags = sum(r['rtc_r3l_flagged'] for r in holdout)
    union = sum(r['union'] for r in holdout)
    valid = sum(r['rtc_r3l_valid'] for r in holdout)
    gates = dict(holdout_coverage=valid==600,holdout_lower_flags=flags/600<=.01,
                 holdout_union_flags=union/600<=.01)
    return dict(stage='G2-clean-calibration',quality_accepted=True,candidate_accepted=False,
        calibration_accepted=all(gates.values()),checks=gates,threshold=threshold,
        calibration_records=600,holdout_records=600,holdout_flags=flags,holdout_union_flags=union,
        calibration_seed=106,holdout_seed=107,threshold_uses_attack_data=False,
        parent='M3',training_units=4,final_goal_achieved=False,
        pending=protocol['pending'],retained_candidates=['M2-G1','M2-H2'])


def analyze(root):
    from experiments import rtc_r3_lower_tail_stage as lower_stage
    from defenses.rtc.lower_tail import load_calibration
    lock = verify_lock(root)
    scope = dict(validator.verify_cell.__globals__,MODES={BASE:('cap','cap')},RTC=(BASE,))
    verify = types.FunctionType(validator.verify_cell.__code__,scope)
    runs=[]
    old_cal = load_calibration(ROOT/'config/rtc_r3_lower_tail_calibration.json')
    for cell in lock['cells']:
        # The observer retains exactly the confirmed-memory parent dynamics;
        # its separate lower-tail fields are verified below.
        mapped = dict(cell,defense=BASE)
        run = verify(root,mapped,lock)
        run['cell']=cell
        if cell['defense']==OBSERVER:
            lower_stage.verify_lower(run,old_cal)
        runs.append(run)
    result=decide(runs,lock['protocol'])
    result['source_lock_sha256']=prior.digest(root/'g2c_lock.json')
    prior.write(root/'analysis/decision.json',result)
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
        print(json.dumps({k: result[k] for k in ('stage', 'training_units', 'quality_accepted', 'candidate_accepted') if k in result}))


if __name__ == '__main__':
    main()
