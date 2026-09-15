"""Reference reliability observation on accepted I12 parent. All training is manual."""
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
from defenses.rtc.reference_history import VERSION

PARENT = 'rtc_i12_combined'
OBSERVER = 'rtc_i12_reference_observe'
MK = 'rtc_i12_multikrum'
OUTPUT = ROOT / 'logs/rtc_reference_history_observation'
PROTOCOL = ROOT / 'config/rtc_reference_history_observation.json'
RECEIPT = ROOT / 'config/rtc_i12_accepted.json'
MODES = {PARENT: ('cap', 'cap'), OBSERVER: ('cap', 'cap'), MK: ('observe', 'observe')}


def batches():
    return [('none_seed201', 'none', 201, (PARENT, OBSERVER)),
            ('sign_flip_seed201', 'sign_flip', 201, (PARENT, OBSERVER, MK))]


def command(root, name, attack, seed, defenses, execute=False):
    return prior.command(root, name, attack, seed, defenses, execute=execute)


def verify_lock(root):
    lock = prior.read_json(root / 'r1h_lock.json')
    assert lock['protocol'] == prior.read_json(PROTOCOL)
    assert lock['accepted_parent'] == prior.read_json(RECEIPT)
    assert lock['sources'] == prior.sources(), 'Source drift; preserve old batch'
    assert lock['environment'] == prior.environment(), 'Use original training host/environment'
    for name, h in lock['artifacts'].items():
        assert prior.digest(root / name) == h, name
    assert {(c['batch'], c['attack'], c['seed'], c['defense']) for c in lock['cells']} == {
        (name, a, s, d) for name, a, s, ds in batches() for d in ds}
    assert len(lock['cells']) == lock['training_units'] == 5
    return lock


def prepare(root, data_dir=None):
    import yaml
    from attacks.spec import attack_source_hash
    from experiments import periodic_attack
    from experiments.run import parse_args
    if (root / 'r1h_lock.json').exists():
        lock = verify_lock(root)
        if data_dir is not None:
            assert str(Path(data_dir).resolve()) == lock['data_dir']
        return lock
    if root.exists() and any(root.iterdir()):
        raise ValueError('Nonempty unfrozen output; inspect before choosing a new directory')
    receipt = prior.read_json(RECEIPT)
    assert receipt['candidate_accepted'] and receipt['candidate'] == PARENT
    protocol = prior.read_json(PROTOCOL)
    assert protocol['training_units'] == 5 and protocol['observer_version'] == VERSION
    root.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((ROOT / 'config/config.yaml').read_text(encoding='utf-8'))
    data_dir = str(Path(data_dir or ROOT / 'data').resolve())
    cfg['dataset']['data_dir'] = data_dir
    (root / 'runtime_config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    prior.write(root / 'attack.freeze.json', {'schema_version': 'RTCByzantineAttackFreezeV1',
        'implementation_source_sha256': attack_source_hash(), 'selection_policy': 'Fixed seed201 reference reliability diagnosis; no sampling or strength changes',
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
        assert observed['security']['defense']['custom_params'].pop('reference_history_observe_only') is True
        assert observed == configs[PARENT], 'Observer must be the only configuration change'
    cal = prior.load_calibration(prior.CAL)
    raw = prior.raw_norm.load_calibration(prior.RAW_CAL)
    source_hashes = prior.sources()
    with zipfile.ZipFile(root / 'r1h_sources.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, expected in sorted(source_hashes.items()):
            content = (ROOT / name).read_bytes()
            assert hashlib.sha256(content).hexdigest() == expected, name
            archive.writestr(name, content)
    artifacts = {p.relative_to(root).as_posix(): prior.digest(p) for p in root.rglob('*') if p.is_file()}
    lock = dict(schema='RTCReferenceHistoryObservationV1', stage='reference-history-observe', training_units=5,
        reused_training_units=0, accepted_parent=receipt, protocol=protocol, sources=source_hashes,
        artifacts=artifacts, environment=prior.environment(), data_dir=data_dir, cells=cells,
        calibration=cal, calibration_hash=prior.calibration_hash(cal), raw_calibration=raw,
        raw_calibration_hash=prior.raw_norm.calibration_hash(raw))
    prior.write(root / 'r1h_lock.json', lock)
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


def verify_history(run):
    previous_hash='';previous_norm=None;scores=[]
    for r in run['rounds'][1:]:
        assert r['fit_rtc_r1h_version']==VERSION
        assert prior.truth(r['fit_rtc_r1h_observe_only'])
        assert r['fit_rtc_r1h_previous_sha256']==previous_hash
        current_hash=r['fit_rtc_r1h_current_sha256']
        assert len(current_hash)==64 and all(c in '0123456789abcdef' for c in current_hash)
        if previous_norm is None:assert r['fit_rtc_r1h_previous_norm']==''
        else:assert abs(float(r['fit_rtc_r1h_previous_norm'])-previous_norm)<=1e-10*max(1,previous_norm)
        ids=json.loads(r['fit_rtc_r1_client_ids_json'])
        index={c['cid']:c for c in run['clients'] if c['round']==r['round']}
        g=np.asarray(json.loads(r['fit_rtc_r1_gram_json']));norm=np.sqrt(np.maximum(np.diag(g),0))
        dots=np.asarray([float(index[c]['rtc_r1h_previous_dot']) for c in ids]) if previous_norm is not None else None
        if dots is not None:
            assert np.isfinite(dots).all()
            augmented=np.block([[g,dots[:,None]],[dots[None,:],np.array([[previous_norm**2]])]])
            assert np.linalg.eigvalsh(augmented).min()>=-1e-8*max(1.,float(np.diag(augmented).max()))
        for i,cid in enumerate(ids):
            c=index[cid];ref=json.loads(c['rtc_r1_reference_ids_json']);beta=np.zeros(len(ids))
            if ref:
                indexes=[ids.index(j) for j in ref]
                beta[indexes]=np.asarray(json.loads(c['rtc_r1_reference_weights_json']))/norm[indexes]
            rn=float(np.sqrt(max(0,beta@g@beta)))
            valid=bool(previous_norm is not None and previous_norm>1e-12 and prior.truth(c['rtc_r1_valid']) and rn>1e-12)
            assert prior.truth(c['rtc_r1h_valid'])==valid
            value=float(np.clip(beta@dots/rn/previous_norm,-1,1)) if valid else None
            if valid:assert abs(float(c['rtc_r1h_reference_cosine'])-value)<1e-10
            else:assert c['rtc_r1h_reference_cosine']==''
            if previous_norm is None:assert c['rtc_r1h_previous_dot']==''
            scores.append(dict(round=int(r['round']),cid=cid,attack_active=prior.truth(c['attack_active']),
                r1_flagged=prior.truth(c['rtc_r1_flagged']),valid=valid,reference_previous_cosine=value,
                reference_attackers=sum(prior.truth(index[j]['attack_active']) for j in ref),
                active_attackers=sum(prior.truth(x['attack_active']) for x in index.values())))
        previous_hash=current_hash;previous_norm=float(r['fit_rtc_r1_actual_aggregate_norm'])
    return scores


def decide(runs):
    assert len(runs)==5 and {(r['cell']['batch'],r['cell']['defense']) for r in runs}=={(n,d) for n,_,_,ds in batches() for d in ds}
    observations=[]
    for name,attack,seed,defenses in batches():
        index={r['cell']['defense']:r for r in runs if r['cell']['batch']==name}
        a,b=index[PARENT],index[OBSERVER]
        for run in index.values():prior.paired(a,run)
        for x,y in zip(a['rounds'][1:],b['rounds'][1:]):
            for key in ('server_accuracy','server_loss','fit_aggregate_update_sketch_json','fit_defense_random_seed'):
                assert x[key]==y[key],('Observer changed trajectory',name,x['round'],key)
        original={(c['round'],c['cid']):c for c in a['clients']}
        for c in b['clients']:
            for key in ('aggregation_weight','rtc_r1_q','rtc_r2_q','rtc_v3_cumulative_q_full'):
                assert c[key]==original[c['round'],c['cid']][key]
        observations.append(dict(batch=name,attack=attack,seed=seed,records=verify_history(b)))
    return dict(stage='reference-history-observe',observation_quality_accepted=True,candidate_accepted=False,
        reason='Observation only; known parent safety failure is retained and is not waived',
        parent=PARENT,observations=observations,summaries=[prior.summarize(r) for r in runs],final_goal_achieved=False)


def analyze(root):
    lock=verify_lock(root)
    scope=dict(validator.verify_cell.__globals__,MODES=MODES,RTC=(PARENT,OBSERVER))
    verify=types.FunctionType(validator.verify_cell.__code__,scope)
    result=decide([verify(root,c,lock) for c in lock['cells']])
    result['source_lock_sha256']=prior.digest(root/'r1h_lock.json')
    prior.write(root/'analysis/observation.json',result)
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
