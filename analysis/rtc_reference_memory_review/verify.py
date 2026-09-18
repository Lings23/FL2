"""Read-only import audit of Linux H2b batch96 results; never prepares or starts training."""
from pathlib import Path
import csv
import ast
import copy
from decimal import Decimal
import json
import sys
import types
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments import rtc_reference_memory_stage as stage
from experiments import rtc_reference_memory_validation as validation
from analysis.rtc_reference_history_review.verify import read, sha, compare
from analysis.rtc_r1c_review.verify import direct_reference

BATCH = ROOT / 'logs/rtc_reference_memory'
HERE = Path(__file__).resolve().parent


def integral_metadata(value):
    number = Decimal(value)
    assert number.is_finite() and number == number.to_integral_value(), value
    return str(int(number))


def audit():
    lock = read(BATCH / 'r1m_lock.json')
    assert lock['protocol'] == read(stage.PROTOCOL)
    assert lock['accepted_parent'] == read(stage.RECEIPT)
    assert len(lock['cells']) == lock['training_units'] == 32
    assert {(c['batch'], c['attack'], c['seed'], c['defense']) for c in lock['cells']} == {
        (n, a, s, d) for n, a, s, ds in stage.batches() for d in ds}
    for name, expected in lock['artifacts'].items():
        assert sha((BATCH / name).read_bytes()) == expected, name
    with zipfile.ZipFile(BATCH / 'r1m_sources.zip') as archive:
        assert len(archive.namelist()) == len(set(archive.namelist()))
        assert set(archive.namelist()) == set(lock['sources'])
        sources = {n: archive.read(n) for n in archive.namelist()}
    for name, expected in lock['sources'].items():
        assert sha(sources[name]) == expected, name
    relevant = ['defenses/rtc/spectral_direction.py', 'defenses/rtc/corroboration.py',
        'defenses/rtc/raw_norm.py',
        'experiments/rtc_reference_memory_validation.py', 'defenses/rtc/reference_eligibility.py',
        'experiments/rtc_r2_raw_norm.py', 'experiments/rtc_r0b_observation.py',
        'analysis/rtc_r3_diagnostics/replay.py', 'config/rtc_reference_eligibility_protocol.json']
    for name in relevant:
        assert (ROOT / name).read_bytes().replace(b'\r\n', b'\n') == sources[name].replace(b'\r\n', b'\n'), name
    # Only the exact observed resource/preparation differences are permitted.
    for name in ('experiments/rtc_reference_memory_stage.py', 'experiments/rtc_i12_bridge.py'):
        local = (ROOT/name).read_text()
        if name.endswith('memory_stage.py'):
            local = local.replace("+ 1] = '.1'", "+ 1] = '.125'")
        else:
            local = local.replace("'--batch-size', '48'", "'--batch-size', '96'")
            local = local.replace("'--ray-client-num-gpus', '.25'", "'--ray-client-num-gpus', '.125'")
        assert local == sources[name].decode().replace('\r\n','\n'), name
    expected_client = dict(local_epochs=5, batch_size=96, optimizer='sgd', learning_rate=.01,
        momentum=.9, weight_decay=.0001, lr_scheduler='cosine')
    resolved = {}
    for cell in lock['cells']:
        cfg = read(BATCH / cell['batch'] / 'resolved' / (cell['run_id']+'.json'))
        assert cfg['client'] == expected_client, cell
        assert cfg['ray']['client_num_gpus'] == .125, cell
        assert cfg['federation']['num_rounds'] == 60 and cfg['model']['architecture'] == 'resnet18' and not cfg['model']['pretrained']
        resolved[cell['batch'],cell['defense']] = cfg
    for name, attack, seed, ds in stage.batches():
        candidate = copy.deepcopy(resolved[name,stage.OBSERVER])
        assert candidate['security']['defense']['custom_params'].pop('reference_eligibility_memory') == 'confirmed'
        assert candidate == resolved[name,stage.PARENT], name
        candidate['security']['defense']['custom_params']['reference_eligibility_mode'] = 'observe'
        assert candidate == resolved[name,stage.BASE], name
    mappings = set()
    normalized_fields = []

    def imported_read(path):
        p = Path(path)
        s = p.as_posix()
        if p.is_file() and p.resolve().is_relative_to(BATCH):
            return read(p)
        marker = '/logs/rtc_reference_memory/'
        if marker in s:
            rel = s.split(marker, 1)[1]
            target = (BATCH / rel).resolve()
            assert target.is_relative_to(BATCH) and rel in lock['artifacts']
            assert sha(target.read_bytes()) == lock['artifacts'][rel]
            mappings.add((s, str(target)))
            return read(target)
        if '/config/' in s:
            return json.loads(sources['config/' + s.split('/config/', 1)[1]])
        return read(p)

    scope = dict(validation.verify_cell.__globals__, read_json=imported_read, read_csv=validation.read_csv,
        MODES=stage.MODES, RTC=(stage.BASE, stage.PARENT, stage.OBSERVER))
    verify = types.FunctionType(validation.verify_cell.__code__, scope)
    runs = [verify(BATCH, c, lock) for c in lock['cells']]
    result = stage.decide(runs, lock['protocol'])
    records = 0
    max_error = 0.
    flagged = []
    for run in runs:
        lookup = {(c['round'], c['cid']): c for c in run['clients']}
        rtc = run['cell']['defense'] in (stage.BASE, stage.PARENT, stage.OBSERVER)
        for r in run['rounds'][1:]:
            g = np.asarray(json.loads(r['fit_rtc_r1_gram_json']))
            eig, vec = np.linalg.eigh(g)
            assert eig.min() >= -1e-8 * max(1., np.diag(g).max())
            x = vec * np.sqrt(np.maximum(eig, 0))[None, :]
            ids = json.loads(r['fit_rtc_r1_client_ids_json'])
            for i, cid in enumerate(ids):
                c = lookup[r['round'], cid]
                ref = direct_reference(x, i)
                assert (ref is not None) == stage.prior.truth(c['rtc_r1_valid'])
                if ref:
                    error = abs(ref['cosine'] - float(c['rtc_r1_cosine']))
                    assert error < 1e-8
                    max_error = max(max_error, error)
                records += 1
                if rtc and (stage.prior.truth(c['rtc_r1e_original_flagged']) or stage.prior.truth(c['rtc_r1_flagged'])):
                    refs = json.loads(c['rtc_r1_reference_ids_json'])
                    flagged.append(dict(batch=run['cell']['batch'], defense=run['cell']['defense'],
                        round=int(r['round']), cid=cid, attack_active=stage.prior.truth(c['attack_active']),
                        original=stage.prior.truth(c['rtc_r1e_original_flagged']),
                        effective=stage.prior.truth(c['rtc_r1_flagged']), vetoed=stage.prior.truth(c['rtc_r1e_vetoed']),
                        eligible=int(c['rtc_r1e_eligible_count']), required=int(c['rtc_r1e_required']),
                        rejected_principals=json.loads(c['rtc_r1e_rejected_principals_json']),
                        spectral_cosine=float(c['rtc_r1_cosine']), weight=float(c['aggregation_weight']),
                        reference_count=len(refs), reference_attackers=sum(stage.prior.truth(lookup[r['round'], j]['attack_active']) for j in refs)))
    server = read(BATCH / 'analysis/decision.json')
    assert server['source_lock_sha256'] == sha((BATCH/'r1m_lock.json').read_bytes())
    for name, expected in server['evidence'].items():
        assert sha((BATCH/name).read_bytes()) == expected, name
    for k, value in result.items():
        compare(server[k], value, k)
    result.update(source_lock_sha256=sha((BATCH / 'r1m_lock.json').read_bytes()),
        source_archive_sha256=sha((BATCH / 'r1m_sources.zip').read_bytes()),
        sources_verified=len(sources), artifacts_verified=len(lock['artifacts']), verified_runs=len(runs),
        runner_gates=sum(len(list(csv.DictReader((BATCH / n / 'quality_gates.csv').open(encoding='utf-8-sig')))) for n, _, _, _ in stage.batches()),
        independent_reference_records=records, max_cosine_error=max_error, server_decision_matches=True,
        server_comparison_float_tolerance=1e-12, flagged_eligibility_rows=flagged,
        client_contract=expected_client, ray_client_num_gpus=.125, path_mappings=sorted(mappings), training_environment=lock['environment'],
        integral_metadata_normalization=normalized_fields,
        protocol_deviations=[dict(field='ray_client_num_gpus', registered=.1, actual=.125,
            scope='all 32 resolved and raw configs; preparation source difference only; no pooled old runs')],
        evidence={p.relative_to(BATCH).as_posix(): sha(p.read_bytes()) for p in BATCH.rglob('*')
            if p.is_file() and p.suffix in ('.json', '.csv')})
    return result


if __name__ == '__main__':
    if not __debug__:
        raise RuntimeError('Assertions must remain enabled')
    result = audit()
    HERE.mkdir(exist_ok=True)
    (HERE / 'review.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('quality_accepted', 'candidate_accepted', 'verified_runs',
        'sources_verified', 'artifacts_verified', 'runner_gates', 'independent_reference_records',
        'max_cosine_error', 'server_decision_matches')}, indent=2))
    print('Failed gates:', [k for k, v in result['checks'].items() if not v])
