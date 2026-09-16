"""Read-only import audit of Linux H1 results; never prepares or starts training."""
from pathlib import Path
import csv
import json
import sys
import types
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments import rtc_reference_guard_stage as stage
from experiments import rtc_reference_guard_validation as validation
from analysis.rtc_reference_history_review.verify import read, sha, compare
from analysis.rtc_r1c_review.verify import direct_reference

BATCH = ROOT / 'logs/rtc_reference_guard'
HERE = Path(__file__).resolve().parent


def audit():
    lock = read(BATCH / 'r1g_lock.json')
    assert lock['protocol'] == read(stage.PROTOCOL)
    assert lock['accepted_parent'] == read(stage.RECEIPT)
    assert len(lock['cells']) == lock['training_units'] == 24
    assert {(c['batch'], c['attack'], c['seed'], c['defense']) for c in lock['cells']} == {
        (n, a, s, d) for n, a, s, ds in stage.batches() for d in ds}
    for name, expected in lock['artifacts'].items():
        assert sha((BATCH / name).read_bytes()) == expected, name
    with zipfile.ZipFile(BATCH / 'r1g_sources.zip') as archive:
        assert len(archive.namelist()) == len(set(archive.namelist()))
        assert set(archive.namelist()) == set(lock['sources'])
        sources = {n: archive.read(n) for n in archive.namelist()}
    for name, expected in lock['sources'].items():
        assert sha(sources[name]) == expected, name
    relevant = ['defenses/rtc/spectral_direction.py', 'defenses/rtc/corroboration.py',
        'defenses/rtc/raw_norm.py', 'defenses/rtc/reference_guard.py', 'defenses/rtc/reference_history.py',
        'experiments/rtc_reference_guard_stage.py', 'experiments/rtc_reference_guard_validation.py',
        'experiments/rtc_reference_history_observation.py', 'experiments/rtc_i12_bridge.py',
        'experiments/rtc_r2_raw_norm.py', 'experiments/rtc_r0b_observation.py',
        'analysis/rtc_r3_diagnostics/replay.py', 'config/rtc_reference_guard_protocol.json']
    for name in relevant:
        assert (ROOT / name).read_bytes().replace(b'\r\n', b'\n') == sources[name].replace(b'\r\n', b'\n'), name
    mappings = set()

    def imported_read(path):
        p = Path(path)
        s = p.as_posix()
        if p.is_file() and p.resolve().is_relative_to(BATCH):
            return read(p)
        marker = '/logs/rtc_reference_guard/'
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

    scope = dict(validation.verify_cell.__globals__, read_json=imported_read,
        MODES=stage.MODES, RTC=(stage.PARENT, stage.OBSERVER))
    verify = types.FunctionType(validation.verify_cell.__code__, scope)
    runs = [verify(BATCH, c, lock) for c in lock['cells']]
    history = [stage.verify_history(r) for r in runs if r['cell']['defense'] in (stage.PARENT, stage.OBSERVER)]
    result = stage.decide(runs, lock['protocol'])
    records = 0
    max_error = 0.
    flagged = []
    for run in runs:
        lookup = {(c['round'], c['cid']): c for c in run['clients']}
        rtc = run['cell']['defense'] in (stage.PARENT, stage.OBSERVER)
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
                if rtc and (stage.prior.truth(c['rtc_r1g_original_flagged']) or stage.prior.truth(c['rtc_r1_flagged'])):
                    refs = json.loads(c['rtc_r1_reference_ids_json'])
                    flagged.append(dict(batch=run['cell']['batch'], defense=run['cell']['defense'],
                        round=int(r['round']), cid=cid, attack_active=stage.prior.truth(c['attack_active']),
                        original=stage.prior.truth(c['rtc_r1g_original_flagged']),
                        effective=stage.prior.truth(c['rtc_r1_flagged']), vetoed=stage.prior.truth(c['rtc_r1g_vetoed']),
                        history_cosine=float(c['rtc_r1g_reference_cosine']) if c['rtc_r1g_reference_cosine'] else None,
                        spectral_cosine=float(c['rtc_r1_cosine']), weight=float(c['aggregation_weight']),
                        reference_count=len(refs), reference_attackers=sum(stage.prior.truth(lookup[r['round'], j]['attack_active']) for j in refs)))
    server = read(BATCH / 'analysis/decision.json')
    for k, value in result.items():
        compare(server[k], value, k)
    result.update(source_lock_sha256=sha((BATCH / 'r1g_lock.json').read_bytes()),
        source_archive_sha256=sha((BATCH / 'r1g_sources.zip').read_bytes()),
        sources_verified=len(sources), artifacts_verified=len(lock['artifacts']), verified_runs=len(runs),
        runner_gates=sum(len(list(csv.DictReader((BATCH / n / 'quality_gates.csv').open(encoding='utf-8-sig')))) for n, _, _, _ in stage.batches()),
        independent_reference_records=records, max_cosine_error=max_error, server_decision_matches=True,
        server_comparison_float_tolerance=1e-12, flagged_history_rows=flagged,
        history_runs_verified=len(history), path_mappings=sorted(mappings), training_environment=lock['environment'],
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
