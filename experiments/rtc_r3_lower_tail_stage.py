"""R3 persistent lower-tail candidate on accepted M2. All training is manual."""
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
from defenses.rtc import lower_tail
import statistics
import math

PARENT = 'rtc_i12_lower_observe'
OBSERVER = 'rtc_i12_lower_cap'
MK = 'rtc_i12_multikrum'
RFA = 'rtc_i12_rfa'
LOWER_CAL = ROOT / 'config/rtc_r3_lower_tail_calibration.json'
OUTPUT = ROOT / 'logs/rtc_r3_lower_tail'
PROTOCOL = ROOT / 'config/rtc_r3_lower_tail_protocol.json'
RECEIPT = ROOT / 'config/rtc_i12_accepted.json'
MODES = {PARENT: ('cap', 'cap'), OBSERVER: ('cap', 'cap'), MK: ('observe', 'observe'), RFA: ('observe', 'observe')}


def batches():
    return [(f'{attack}_seed{seed}', attack, seed, ds) for seed in (201,202)
            for attack,ds in (('none',(PARENT,OBSERVER)),('sign_flip',(PARENT,OBSERVER,MK)),
                ('gaussian_noise',(PARENT,OBSERVER,MK,RFA)),('lie',(PARENT,OBSERVER,MK)))]


def command(root, name, attack, seed, defenses, execute=False):
    return prior.command(root, name, attack, seed, defenses, execute=execute)


def verify_lock(root):
    lock = prior.read_json(root / 'r3l_lock.json')
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
    if (root / 'r3l_lock.json').exists():
        lock = verify_lock(root)
        if data_dir is not None:
            assert str(Path(data_dir).resolve()) == lock['data_dir']
        return lock
    if root.exists() and any(root.iterdir()):
        raise ValueError('Nonempty unfrozen output; inspect before choosing a new directory')
    receipt = prior.read_json(RECEIPT)
    assert receipt['candidate_accepted'] and receipt['candidate'] == 'rtc_i12_combined'
    protocol = prior.read_json(PROTOCOL)
    assert protocol['training_units'] == 24 and protocol['lower_calibration_hash'] == lower_tail.calibration_hash(lower_tail.load_calibration(LOWER_CAL))
    root.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((ROOT / 'config/config.yaml').read_text(encoding='utf-8'))
    data_dir = str(Path(data_dir or ROOT / 'data').resolve())
    cfg['dataset']['data_dir'] = data_dir
    (root / 'runtime_config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    prior.write(root / 'attack.freeze.json', {'schema_version': 'RTCByzantineAttackFreezeV1',
        'implementation_source_sha256': attack_source_hash(), 'selection_policy': 'Fixed M2 regression and LIE .5 lower-tail test; no attack-strength tuning',
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
            assert config['client'] == dict(local_epochs=5, batch_size=48, optimizer='sgd', learning_rate=.01,
                momentum=.9, weight_decay=.0001, lr_scheduler='cosine')
            assert config['federation']['num_rounds'] == 60 and config['federation']['deterministic_client_training']
            assert config['model']['architecture'] == 'resnet18' and not config['model']['pretrained']
            configs[s['defense']] = config
            prior.write(root / name / 'resolved' / (rid + '.json'), config)
            cells.append(dict(batch=name, attack=attack, seed=seed, defense=s['defense'], run_id=rid,
                **{k: s[k] for k in ('trial_plan_hash', 'attack_implementation_hash', 'attack_contract_hash')}))
        observed = copy.deepcopy(configs[OBSERVER])
        assert observed['security']['defense']['custom_params']['lower_tail_mode'] == 'cap'
        observed['security']['defense']['custom_params']['lower_tail_mode'] = 'observe'
        assert observed == configs[PARENT], 'Lower-tail mode must be the only configuration change'
    cal = prior.load_calibration(prior.CAL)
    raw = prior.raw_norm.load_calibration(prior.RAW_CAL)
    source_hashes = prior.sources()
    with zipfile.ZipFile(root / 'r3l_sources.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, expected in sorted(source_hashes.items()):
            content = (ROOT / name).read_bytes()
            assert hashlib.sha256(content).hexdigest() == expected, name
            archive.writestr(name, content)
    artifacts = {p.relative_to(root).as_posix(): prior.digest(p) for p in root.rglob('*') if p.is_file()}
    lock = dict(schema='RTCR3LowerTailV1', stage='R3-lower-tail', training_units=24,
        reused_training_units=0, accepted_parent=receipt, protocol=protocol, sources=source_hashes,
        artifacts=artifacts, environment=prior.environment(), data_dir=data_dir, cells=cells,
        calibration=cal, calibration_hash=prior.calibration_hash(cal), raw_calibration=raw,
        raw_calibration_hash=prior.raw_norm.calibration_hash(raw))
    prior.write(root / 'r3l_lock.json', lock)
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


def verify_lower(run, calibration):
    states={}
    mode='cap' if run['cell']['defense']==OBSERVER else 'observe'
    for r in run['rounds'][1:]:
        assert r['fit_rtc_r3l_version']==lower_tail.VERSION and r['fit_rtc_r3l_mode']==mode
        assert r['fit_rtc_r3l_calibration_hash']==lower_tail.calibration_hash(calibration)
        clients=[c for c in run['clients'] if c['round']==r['round']]
        assert len(clients)==10 and len({c['principal_id'] for c in clients})==10
        for c in clients:
            norm=float(c['rtc_r3l_residual_norm'])
            assert math.isfinite(norm) and norm>=0
            assert math.isfinite(float(c['rtc_r3l_principal_residual_norm']))
            assert abs(norm-float(c['rtc_r3l_principal_residual_norm']))<=1e-10*max(1,norm)
            others=[float(x['rtc_r3l_residual_norm']) for x in clients if x['principal_id']!=c['principal_id']]
            median=statistics.median(others);valid=median>1e-12
            ratio=norm/median if valid else None
            low=valid and ratio<calibration['ratio_threshold']
            pid=c['principal_id'];states[pid]=min(2,states.get(pid,0)+1) if low else 0
            flag=states[pid]>=2;q=0. if flag and mode=='cap' else 1.
            assert int(c['rtc_r3l_reference_count'])==9
            assert abs(float(c['rtc_r3l_reference_median'])-median)<=1e-10*max(1,median)
            assert prior.truth(c['rtc_r3l_valid'])==valid and prior.truth(c['rtc_r3l_low'])==low
            if valid:assert abs(float(c['rtc_r3l_ratio'])-ratio)<=1e-10*max(1,ratio)
            else:assert c['rtc_r3l_ratio']==''
            assert int(c['rtc_r3l_streak'])==states[pid] and prior.truth(c['rtc_r3l_flagged'])==flag
            assert float(c['rtc_r3l_q'])==q
            existing=min(float(c['rtc_r1_existing_q']),float(c['rtc_r1_q']),float(c['rtc_r2_q']))
            assert float(c['rtc_r3l_existing_q'])==existing
            assert float(c['rtc_r3l_nominal_mass'])==float(c['rtc_r1_nominal_mass'])
            assert prior.truth(c['rtc_r3l_applied'])==(q<existing)
            assert 0<=float(c['aggregation_weight'])<=float(c['rtc_r3l_nominal_mass'])*min(q,existing)+1e-8


def metrics(run):
    result=prior.summarize(run)
    is_rtc=run['cell']['defense'] in (PARENT,OBSERVER)
    start=1 if run['cell']['attack']=='none' else 11
    clients=[c for c in run['clients'] if int(c['round'])>=start]
    good=[c for c in clients if not prior.truth(c['attack_active'])]
    bad=[c for c in clients if prior.truth(c['attack_active'])]
    def union(c):
        return is_rtc and (prior.truth(c['rtc_r1_flagged']) or prior.truth(c['rtc_r2_flagged']) or
            run['cell']['defense']==OBSERVER and prior.truth(c['rtc_r3l_flagged']))
    result['benign_union_flag_rate']=sum(union(c) for c in good)/len(good)
    if is_rtc:
        result['lower_benign_flags']=sum(prior.truth(c['rtc_r3l_flagged']) for c in good)
        result['lower_malicious_flags']=sum(prior.truth(c['rtc_r3l_flagged']) for c in bad)
        result['first_lower_attack_flag_round']=min((int(c['round']) for c in bad if prior.truth(c['rtc_r3l_flagged'])),default=None)
    return result


def decide(runs, protocol):
    index={(r['cell']['attack'],r['cell']['seed'],r['cell']['defense']):r for r in runs}
    assert set(index)=={(a,s,d) for _,a,s,ds in batches() for d in ds} and len(runs)==24
    summaries=[metrics(r) for r in runs]
    table={(r['attack'],r['seed'],r['defense']):r for r in summaries}
    gates={};effects=[];g=protocol['gates']
    for name,attack,seed,ds in batches():
        for d in ds:
            prior.paired(index[attack,seed,PARENT],index[attack,seed,d])
        a,b=index[attack,seed,PARENT],index[attack,seed,OBSERVER]
        for x,y in zip(a['rounds'][1:],b['rounds'][1:]):assert x['fit_defense_random_seed']==y['fit_defense_random_seed']
        parent,candidate=table[attack,seed,PARENT],table[attack,seed,OBSERVER]
        for label,item in [('parent',parent),('candidate',candidate)]:
            gates[name+'/'+label+'/benign_union']=item['benign_union_flag_rate']<=g['benign_union_flag_max']
        for key in ('active_accuracy','final_accuracy'):
            delta=candidate[key]-parent[key]
            minimum=(g['lie_active_gain_min'] if key=='active_accuracy' else g['lie_final_gain_min']) if attack=='lie' else g['other_active_final_gain_min']
            gates[name+'/'+key]=delta>=minimum
            effects.append(dict(attack=attack,seed=seed,metric=key,candidate_minus_parent=delta))
        if attack=='lie':
            gates[name+'/malicious_weight']=candidate['malicious_weight_per_round']<=parent['malicious_weight_per_round']*g['lie_malicious_weight_ratio_max']
        elif attack!='none':
            gates[name+'/malicious_weight']=candidate['malicious_weight_per_round']<=parent['malicious_weight_per_round']+g['other_malicious_weight_increase_max']
    return dict(stage='R3-lower-tail',quality_accepted=True,candidate_accepted=all(gates.values()),checks=gates,
        summaries=summaries,effects=effects,parent='M2 with inert lower-tail observer',candidate=OBSERVER,
        final_goal_achieved=False,scope='IID seeds201/202 clean/Sign-flip/Gaussian/LIE .5 only; engineering screen',
        pending=protocol['pending'])


def analyze(root):
    lock=verify_lock(root)
    scope=dict(validator.verify_cell.__globals__,MODES=MODES,RTC=(PARENT,OBSERVER))
    verify=types.FunctionType(validator.verify_cell.__code__,scope)
    runs=[verify(root,c,lock) for c in lock['cells']]
    calibration=lower_tail.load_calibration(LOWER_CAL)
    for r in runs:
        if r['cell']['defense'] in (PARENT,OBSERVER):verify_lower(r,calibration)
    result=decide(runs,lock['protocol'])
    result['source_lock_sha256']=prior.digest(root/'r3l_lock.json')
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
