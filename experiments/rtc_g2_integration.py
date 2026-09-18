"""G2 filtered lower-tail integration on accepted M3. All training is manual."""
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
from defenses.rtc.lower_tail import load_calibration as load_lower, calibration_hash as lower_hash
from experiments.rtc_g2_validation import verify_lower
LOWER_CAL = ROOT / 'config/rtc_g2_lower_tail_calibration.json'
import statistics
import math

BASE = 'rtc_i12_g2_observe'
PARENT = 'rtc_i12_g1_recalibrated'
OBSERVER = 'rtc_i12_g2_cap'
MK = 'rtc_i12_multikrum'
RFA = 'rtc_i12_rfa'
OUTPUT = ROOT / 'logs/rtc_g2_integration'
PROTOCOL = ROOT / 'config/rtc_g2_integration_protocol.json'
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
        raise ValueError('G2 integration client training contract mismatch: ' + json.dumps(differences, sort_keys=True))


def batches():
    return [(f'{attack}_seed{seed}', attack, seed, ds) for seed in (201,206)
            for attack,ds in (('none',(BASE,PARENT,OBSERVER)),('sign_flip',(BASE,PARENT,OBSERVER,MK)),
                ('gaussian_noise',(BASE,PARENT,OBSERVER,MK,RFA)),('lie',(BASE,PARENT,OBSERVER,MK)))]


def command(root, name, attack, seed, defenses, execute=False):
    inherited = prior.command(root, name, attack, seed, defenses, execute=execute)
    # _build_spec_config gives the CLI precedence over runtime_config.yaml.
    # Shared host command edits must not change this registered G2 recipe.
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
    lock = prior.read_json(root / 'g2_lock.json')
    assert lock['protocol'] == prior.read_json(PROTOCOL)
    assert lock['accepted_parent'] == prior.read_json(RECEIPT)
    assert lock['sources'] == prior.sources(), 'Source drift; preserve old batch'
    assert lock['environment'] == prior.environment(), 'Use original training host/environment'
    for name, h in lock['artifacts'].items():
        assert prior.digest(root / name) == h, name
    assert {(c['batch'], c['attack'], c['seed'], c['defense']) for c in lock['cells']} == {
        (name, a, s, d) for name, a, s, ds in batches() for d in ds}
    assert len(lock['cells']) == lock['training_units'] == 32
    return lock


def prepare(root, data_dir=None):
    import yaml
    from attacks.spec import attack_source_hash
    from experiments import periodic_attack
    from experiments.run import parse_args
    if (root / 'g2_lock.json').exists():
        lock = verify_lock(root)
        if data_dir is not None:
            assert str(Path(data_dir).resolve()) == lock['data_dir']
        return lock
    if root.exists() and any(root.iterdir()):
        raise ValueError('Nonempty unfrozen output; inspect before choosing a new directory')
    receipt = prior.read_json(RECEIPT)
    assert receipt['candidate_accepted'] and receipt['candidate'] == 'rtc_i12_eligibility_confirmed'
    protocol = prior.read_json(PROTOCOL)
    assert protocol['training_units'] == 32 and protocol['lower_calibration_hash'] == lower_hash(load_lower(LOWER_CAL))
    root.mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load((ROOT / 'config/config.yaml').read_text(encoding='utf-8'))
    data_dir = str(Path(data_dir or ROOT / 'data').resolve())
    cfg = runtime_config(base, data_dir)
    cfg.setdefault('ray', {})['client_num_gpus'] = .125
    (root / 'runtime_config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    prior.write(root / 'preparation_client_parameters.json', {
        'source': 'config/config.yaml', 'source_client': base.get('client', {}),
        'registered_client': CLIENT_CONTRACT,
        'reason': 'Pin registered batch96; CLI and runtime GPU quota .125.'})
    from config.config_loader import load_config
    verify_client_contract(asdict(load_config(root / 'runtime_config.yaml').client))
    prior.write(root / 'attack.freeze.json', {'schema_version': 'RTCByzantineAttackFreezeV1',
        'implementation_source_sha256': attack_source_hash(), 'selection_policy': 'M3/G1/G2 fixed paired regression; no attack tuning',
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
        assert observed['security']['defense']['custom_params']['lower_tail_mode']=='cap'
        observed['security']['defense']['custom_params']['lower_tail_mode']='observe'
        assert observed==configs[BASE], 'Candidate only enables cap on filtered observer'
        unfiltered=copy.deepcopy(configs[OBSERVER])
        assert unfiltered['security']['defense']['custom_params']['lower_tail_reference_policy']=='raw_eligible'
        unfiltered['security']['defense']['custom_params']['lower_tail_reference_policy']='all'
        assert unfiltered==configs[PARENT], 'G1 control only differs in reference filtering'
        assert all(cfg['ray']['client_num_gpus']==.125 for cfg in configs.values())
    cal = prior.load_calibration(prior.CAL)
    raw = prior.raw_norm.load_calibration(prior.RAW_CAL)
    source_hashes = prior.sources()
    with zipfile.ZipFile(root / 'g2_sources.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, expected in sorted(source_hashes.items()):
            content = (ROOT / name).read_bytes()
            assert hashlib.sha256(content).hexdigest() == expected, name
            archive.writestr(name, content)
    artifacts = {p.relative_to(root).as_posix(): prior.digest(p) for p in root.rglob('*') if p.is_file()}
    lock = dict(schema='RTCG2IntegrationV1', stage='G2-integration', training_units=32,
        reused_training_units=0, accepted_parent=receipt, protocol=protocol, sources=source_hashes,
        artifacts=artifacts, environment=prior.environment(), data_dir=data_dir, cells=cells,
        calibration=cal, calibration_hash=prior.calibration_hash(cal), raw_calibration=raw,
        raw_calibration_hash=prior.raw_norm.calibration_hash(raw))
    prior.write(root / 'g2_lock.json', lock)
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
    result=prior.summarize(run);rtc=run['cell']['defense'] in (BASE,PARENT,OBSERVER)
    start=1 if run['cell']['attack']=='none' else 11
    cc=[c for c in run['clients'] if int(c['round'])>=start]
    good=[c for c in cc if not prior.truth(c['attack_active'])]
    bad=[c for c in cc if prior.truth(c['attack_active'])]
    union=lambda c:rtc and any(float(c[k])<1 for k in ('rtc_r1_q','rtc_r2_q','rtc_r3l_q'))
    result.update(benign_union_flags=sum(union(c) for c in good),benign_records=len(good),
        benign_union_flag_rate=sum(union(c) for c in good)/len(good))
    if rtc:
        result.update(lower_benign_flags=sum(prior.truth(c['rtc_r3l_flagged']) for c in good),
            lower_malicious_flags=sum(prior.truth(c['rtc_r3l_flagged']) for c in bad),
            lower_valid_records=sum(prior.truth(c['rtc_r3l_valid']) for c in cc),
            first_lower_attack_flag_round=min((int(c['round']) for c in bad if prior.truth(c['rtc_r3l_flagged'])),default=None))
    return result


def decide(runs, protocol):
    index={(r['cell']['attack'],r['cell']['seed'],r['cell']['defense']):r for r in runs}
    assert len(runs)==32 and set(index)=={(a,s,d) for _,a,s,ds in batches() for d in ds}
    summaries=[metrics(r) for r in runs];table={(r['attack'],r['seed'],r['defense']):r for r in summaries}
    gates={};effects=[];divergences=[];g=protocol['gates']
    for name,attack,seed,ds in batches():
        c=table[attack,seed,OBSERVER];m3=table[attack,seed,BASE];g1=table[attack,seed,PARENT]
        gates[name+'/candidate_benign_union']=c['benign_union_flag_rate']<=g['candidate_benign_union_max']
        gates[name+'/preserve_G1_benign_count']=c['benign_union_flags']<=g1['benign_union_flags']
        for d in ds:prior.paired(index[attack,seed,BASE],index[attack,seed,d])
        for ref,label in ((BASE,'M3'),(PARENT,'G1')):
            parent=table[attack,seed,ref];a,b=index[attack,seed,ref],index[attack,seed,OBSERVER]
            for x,y in zip(a['rounds'][1:],b['rounds'][1:]):assert x['fit_defense_random_seed']==y['fit_defense_random_seed']
            for key in ('active_accuracy','final_accuracy'):
                delta=c[key]-parent[key]
                if label=='G1':minimum=g['G1_active_final_noninferiority_min']
                elif attack=='lie':minimum=g['lie_active_gain_min'] if key=='active_accuracy' else g['lie_final_gain_min']
                else:minimum=g['other_active_final_gain_min']
                gates[name+'/'+label+'/'+key]=delta>=minimum
                effects.append(dict(attack=attack,seed=seed,reference=label,metric=key,candidate_minus_reference=delta))
            if attack!='none':
                limit=(g['G1_malicious_weight_increase_max'] if label=='G1' else g['other_malicious_weight_increase_max'])
                maximum=(parent['malicious_weight_per_round']*g['lie_malicious_weight_ratio_max']
                    if label=='M3' and attack=='lie' else parent['malicious_weight_per_round']+limit)
                gates[name+'/'+label+'/malicious_weight']=c['malicious_weight_per_round']<=maximum
            old={(c['round'],c['cid']):c for c in a['clients']}
            first=next((dict(round=int(c['round']),cid=c['cid']) for c in sorted(b['clients'],key=lambda c:(int(c['round']),c['cid']))
                if abs(float(c['aggregation_weight'])-float(old[c['round'],c['cid']]['aggregation_weight']))>1e-10),None)
            divergences.append(dict(attack=attack,seed=seed,reference=label,first_weight_difference=first,
                first_sketch_difference_round=next((int(x['round']) for x,y in zip(a['rounds'][1:],b['rounds'][1:])
                    if x['fit_aggregate_update_sketch_json']!=y['fit_aggregate_update_sketch_json']),None)))
    return dict(stage='G2-integration',quality_accepted=True,candidate_accepted=all(gates.values()),checks=gates,
        summaries=summaries,effects=effects,divergences=divergences,accepted_parent='M3',parent_runtime=BASE,
        unfiltered_control=PARENT,candidate=OBSERVER,retained_candidates=['M2-G1','M2-H2'],final_goal_achieved=False,
        scope='IID batch96 seeds201 development/206 engineering; retained G1 mechanism uses fresh clean-only calibration',pending=protocol['pending'])


def verify_run(root,cell,lock):
    rtc=cell['defense'] in (BASE,PARENT,OBSERVER)
    alias='rtc_i12_eligibility_confirmed' if rtc else cell['defense']
    scope=dict(validator.verify_cell.__globals__,MODES={alias:MODES[cell['defense']]},RTC=('rtc_i12_eligibility_confirmed',))
    verify=types.FunctionType(validator.verify_cell.__code__,scope)
    run=verify(root,dict(cell,defense=alias),lock);run['cell']=cell
    if rtc:
        mode='observe' if cell['defense']==BASE else 'cap'
        policy='all' if cell['defense']==PARENT else 'raw_eligible'
        verify_lower(run,load_lower(LOWER_CAL),mode,policy)
    return run


def analyze(root):
    lock=verify_lock(root)
    runs=[verify_run(root,c,lock) for c in lock['cells']]
    result=decide(runs,lock['protocol']);result['source_lock_sha256']=prior.digest(root/'g2_lock.json')
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
