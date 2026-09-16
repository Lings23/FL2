"""H2 reference-principal eligibility on accepted M2. All training is manual."""
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
from experiments import rtc_reference_eligibility_validation as validator
from defenses.rtc import reference_eligibility
import statistics
import math

PARENT = 'rtc_i12_eligibility_observe'
OBSERVER = 'rtc_i12_eligibility_cap'
MK = 'rtc_i12_multikrum'
RFA = 'rtc_i12_rfa'
OUTPUT = ROOT / 'logs/rtc_reference_eligibility'
PROTOCOL = ROOT / 'config/rtc_reference_eligibility_protocol.json'
RECEIPT = ROOT / 'config/rtc_i12_accepted.json'
MODES = {PARENT: ('cap', 'cap'), OBSERVER: ('cap', 'cap'), MK: ('observe', 'observe'), RFA: ('observe', 'observe')}
CLIENT_CONTRACT = dict(local_epochs=5, batch_size=48, optimizer='sgd', learning_rate=.01,
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
        raise ValueError('H2 client training contract mismatch: ' + json.dumps(differences, sort_keys=True))


def batches():
    return [(f'{attack}_seed{seed}', attack, seed, ds) for seed in (201,204)
            for attack,ds in (('none',(PARENT,OBSERVER)),('sign_flip',(PARENT,OBSERVER,MK)),
                ('gaussian_noise',(PARENT,OBSERVER,MK,RFA)),('lie',(PARENT,OBSERVER,MK)))]


def command(root, name, attack, seed, defenses, execute=False):
    return prior.command(root, name, attack, seed, defenses, execute=execute)


def verify_lock(root):
    lock = prior.read_json(root / 'r1e_lock.json')
    assert lock['protocol'] == prior.read_json(PROTOCOL)
    assert lock['accepted_parent'] == prior.read_json(RECEIPT)
    assert lock['sources'] == prior.sources(), 'Source drift; preserve old batch'
    assert lock['environment'] == prior.environment(), 'Use original training host/environment'
    for name, h in lock['artifacts'].items():
        assert prior.digest(root / name) == h, name
    assert {(c['batch'], c['attack'], c['seed'], c['defense']) for c in lock['cells']} == {
        (name, a, s, d) for name, a, s, ds in batches() for d in ds}
    assert len(lock['cells']) == lock['training_units'] == 24
    return lock


def prepare(root, data_dir=None):
    import yaml
    from attacks.spec import attack_source_hash
    from experiments import periodic_attack
    from experiments.run import parse_args
    if (root / 'r1e_lock.json').exists():
        lock = verify_lock(root)
        if data_dir is not None:
            assert str(Path(data_dir).resolve()) == lock['data_dir']
        return lock
    if root.exists() and any(root.iterdir()):
        raise ValueError('Nonempty unfrozen output; inspect before choosing a new directory')
    receipt = prior.read_json(RECEIPT)
    assert receipt['candidate_accepted'] and receipt['candidate'] == 'rtc_i12_combined'
    protocol = prior.read_json(PROTOCOL)
    assert protocol['training_units'] == 24 and protocol['eligibility_version'] == reference_eligibility.VERSION
    root.mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load((ROOT / 'config/config.yaml').read_text(encoding='utf-8'))
    data_dir = str(Path(data_dir or ROOT / 'data').resolve())
    cfg = runtime_config(base, data_dir)
    (root / 'runtime_config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    prior.write(root / 'preparation_client_parameters.json', {
        'source': 'config/config.yaml', 'source_client': base.get('client', {}),
        'registered_client': CLIENT_CONTRACT,
        'reason': 'Use preregistered H2 training parameters independently of host defaults; GPU quotas are unchanged.'})
    from config.config_loader import load_config
    verify_client_contract(asdict(load_config(root / 'runtime_config.yaml').client))
    prior.write(root / 'attack.freeze.json', {'schema_version': 'RTCByzantineAttackFreezeV1',
        'implementation_source_sha256': attack_source_hash(), 'selection_policy': 'Fixed M2 reference repair and old-attack regression; no strength or sampling tuning',
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
        assert observed['security']['defense']['custom_params']['reference_eligibility_mode'] == 'cap'
        observed['security']['defense']['custom_params']['reference_eligibility_mode'] = 'observe'
        assert observed == configs[PARENT], 'Reference eligibility mode must be the only configuration change'
    cal = prior.load_calibration(prior.CAL)
    raw = prior.raw_norm.load_calibration(prior.RAW_CAL)
    source_hashes = prior.sources()
    with zipfile.ZipFile(root / 'r1e_sources.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, expected in sorted(source_hashes.items()):
            content = (ROOT / name).read_bytes()
            assert hashlib.sha256(content).hexdigest() == expected, name
            archive.writestr(name, content)
    artifacts = {p.relative_to(root).as_posix(): prior.digest(p) for p in root.rglob('*') if p.is_file()}
    lock = dict(schema='RTCReferenceEligibilityV1', stage='reference-eligibility', training_units=24,
        reused_training_units=0, accepted_parent=receipt, protocol=protocol, sources=source_hashes,
        artifacts=artifacts, environment=prior.environment(), data_dir=data_dir, cells=cells,
        calibration=cal, calibration_hash=prior.calibration_hash(cal), raw_calibration=raw,
        raw_calibration_hash=prior.raw_norm.calibration_hash(raw))
    prior.write(root / 'r1e_lock.json', lock)
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


def metrics(run):
    result=prior.summarize(run);rtc=run['cell']['defense'] in (PARENT,OBSERVER)
    start=1 if run['cell']['attack']=='none' else 11
    cc=[c for c in run['clients'] if int(c['round'])>=start]
    good=[c for c in cc if not prior.truth(c['attack_active'])]
    bad=[c for c in cc if prior.truth(c['attack_active'])]
    union=lambda c:rtc and (float(c['rtc_r1_q'])<1 or float(c['rtc_r2_q'])<1)
    result.update(benign_union_flags=sum(union(c) for c in good),benign_records=len(good),
        benign_union_flag_rate=sum(union(c) for c in good)/len(good))
    if rtc:
        result.update(eligibility_vetoed_benign=sum(prior.truth(c['rtc_r1e_vetoed']) for c in good),
            eligibility_vetoed_malicious=sum(prior.truth(c['rtc_r1e_vetoed']) for c in bad),
            direction_flagged_malicious=sum(prior.truth(c['rtc_r1_flagged']) for c in bad),
            first_direction_attack_flag_round=min((int(c['round']) for c in bad if prior.truth(c['rtc_r1_flagged'])),default=None))
    return result


def decide(runs,protocol):
    index={(r['cell']['attack'],r['cell']['seed'],r['cell']['defense']):r for r in runs}
    assert set(index)=={(a,s,d) for _,a,s,ds in batches() for d in ds} and len(runs)==24
    summaries=[metrics(r) for r in runs];table={(r['attack'],r['seed'],r['defense']):r for r in summaries}
    gates={};effects=[];divergences=[];g=protocol['gates']
    for name,attack,seed,ds in batches():
        for d in ds:prior.paired(index[attack,seed,PARENT],index[attack,seed,d])
        a,b=index[attack,seed,PARENT],index[attack,seed,OBSERVER]
        for x,y in zip(a['rounds'][1:],b['rounds'][1:]):assert x['fit_defense_random_seed']==y['fit_defense_random_seed']
        parent,candidate=table[attack,seed,PARENT],table[attack,seed,OBSERVER]
        gates[name+'/candidate_benign_union']=candidate['benign_union_flag_rate']<=g['candidate_benign_union_max']
        for key in ('active_accuracy','final_accuracy'):
            delta=candidate[key]-parent[key]
            gates[name+'/'+key]=delta>=g['accuracy_noninferiority_min']
            effects.append(dict(attack=attack,seed=seed,metric=key,candidate_minus_parent=delta))
        if attack!='none':
            gates[name+'/malicious_weight']=candidate['malicious_weight_per_round']<=parent['malicious_weight_per_round']+g['malicious_weight_increase_max']
        if attack=='sign_flip' and seed==201:
            gates[name+'/strict_benign_reduction']=candidate['benign_union_flags']<parent['benign_union_flags']
        old={(c['round'],c['cid']):c for c in a['clients']}
        first=next((dict(round=int(c['round']),cid=c['cid']) for c in sorted(b['clients'],key=lambda c:(int(c['round']),c['cid']))
            if abs(float(c['aggregation_weight'])-float(old[c['round'],c['cid']]['aggregation_weight']))>1e-10),None)
        matched=[(old[c['round'],c['cid']],c) for c in b['clients'] if int(c['round'])>=11 and prior.truth(c['attack_active'])
            and prior.truth(c['rtc_r1_usable']) and prior.truth(old[c['round'],c['cid']]['rtc_r1_usable'])]
        divergences.append(dict(attack=attack,seed=seed,first_weight_difference=first,
            first_sketch_difference_round=next((int(x['round']) for x,y in zip(a['rounds'][1:],b['rounds'][1:]) if x['fit_aggregate_update_sketch_json']!=y['fit_aggregate_update_sketch_json']),None),
            matched_projection_records=len(matched),
            parent_negative_proxy=sum(float(x['rtc_r1_negative_contribution']) for x,y in matched)/50 if matched else None,
            candidate_negative_proxy=sum(float(y['rtc_r1_negative_contribution']) for x,y in matched)/50 if matched else None,
            proxy_limit='Different per-client LOO reference axes; not total harmful direction'))
    return dict(stage='reference-eligibility',quality_accepted=True,candidate_accepted=all(gates.values()),checks=gates,
        summaries=summaries,effects=effects,divergences=divergences,parent=PARENT,candidate=OBSERVER,
        parent_safety='Known seed201 parent safety failure remains; this new protocol evaluates repair, never rewrites old decisions',
        retained_candidate='M2-G1 remains retained_for_repair_and_reintegration',final_goal_achieved=False,
        scope='IID seeds201 development and204 held-out engineering screen only',pending=protocol['pending'])


def analyze(root):
    lock=verify_lock(root)
    scope=dict(validator.verify_cell.__globals__,MODES=MODES,RTC=(PARENT,OBSERVER))
    verify=types.FunctionType(validator.verify_cell.__code__,scope)
    runs=[verify(root,c,lock) for c in lock['cells']]
    result=decide(runs,lock['protocol'])
    result['source_lock_sha256']=prior.digest(root/'r1e_lock.json')
    result['evidence']={p.relative_to(root).as_posix():prior.digest(p) for p in root.glob('*/*/*')
                        if p.is_file() and p.suffix in ('.json','.csv')}
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
