"""Read-only audit of a copied I12 batch; never invokes training or rewrites its lock.

The frozen validator is reused with a hash-checked JSON path reader. This is an
import audit, not a waiver of verify_lock's same-host execution requirements.
Missing server sources keep formal acceptance pending even if numeric gates pass.
"""
from pathlib import Path
import collections
import hashlib
import json
import math
import subprocess
import sys
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments import rtc_i12_validation as validation
from experiments import rtc_i12_bridge as bridge
from analysis.rtc_r1c_review.verify import direct_reference

BATCH = ROOT / 'logs/rtc_i12_bridge_v2'
HERE = Path(__file__).resolve().parent


def sha(b):
    return hashlib.sha256(b).hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def recover_sources(lock):
    receipt, recovered = {}, {}
    for name, expected in lock['sources'].items():
        path = ROOT / name
        candidates = []
        if path.is_file():
            raw = path.read_bytes()
            candidates.extend([('local_exact', raw), ('local_LF', raw.replace(b'\r\n', b'\n'))])
        # Optional originals exported by the user from the training server.
        exported = HERE / 'server_sources' / name
        if exported.is_file():
            candidates.insert(0, ('server_export', exported.read_bytes()))
        match = next(((kind, data) for kind, data in candidates if sha(data) == expected), None)
        if match is None:
            result = subprocess.run(['git', '-C', str(ROOT), 'show', 'a0a90b0:' + name], capture_output=True)
            if result.returncode == 0 and sha(result.stdout) == expected:
                match = ('git_a0a90b0', result.stdout)
        receipt[name] = {'expected_sha256': expected, 'recovery': match[0] if match else 'missing'}
        if match:
            recovered[name] = match[1]
    return receipt, recovered


def main():
    lock = read(BATCH / 'i12_lock.json')
    assert lock['protocol'] == read(ROOT / 'config/rtc_i12_protocol.json')
    assert len(lock['cells']) == lock['training_units'] == 40
    receipt, recovered = recover_sources(lock)
    for name in ('experiments/rtc_i12_validation.py', 'experiments/rtc_i12_bridge.py',
                 'defenses/rtc/spectral_direction.py', 'defenses/rtc/corroboration.py',
                 'defenses/rtc/raw_norm.py', 'analysis/rtc_r3_diagnostics/replay.py'):
        assert name in recovered, 'Cannot audit with an unverified implementation: ' + name
        assert (ROOT / name).read_bytes().replace(b'\r\n', b'\n') == recovered[name].replace(b'\r\n', b'\n')
    for name, expected in lock['artifacts'].items():
        assert sha((BATCH / name).read_bytes()) == expected, name

    mappings = set()
    def imported_read(path):
        p = Path(path)
        s = p.as_posix()
        if p.is_file() and p.resolve().is_relative_to(BATCH):
            return read(p)
        marker = '/logs/rtc_i12_bridge_v2/'
        if marker in s:
            rel = s.split(marker, 1)[1]
            target = (BATCH / rel).resolve()
            assert target.is_relative_to(BATCH) and rel in lock['artifacts']
            assert sha(target.read_bytes()) == lock['artifacts'][rel]
            mappings.add((s, str(target)))
            return read(target)
        if '/config/' in s:
            rel = 'config/' + s.split('/config/', 1)[1]
            assert rel in recovered, 'Missing original configuration: ' + rel
            return json.loads(recovered[rel])
        return read(p)

    globals_copy = dict(validation.verify_cell.__globals__)
    globals_copy['read_json'] = imported_read
    verify = types.FunctionType(validation.verify_cell.__code__, globals_copy)
    runs = [verify(BATCH, c, lock) for c in lock['cells']]
    result = bridge.decide(runs, lock['protocol'])

    # Independently reconstruct vectors from each logged Gram matrix, then
    # recompute the leave-one-out spectral reference and corroboration votes.
    max_cosine_error, records = 0., 0
    for run in runs:
        clients = {(r['round'], r['cid']): r for r in run['clients']}
        for row in run['rounds'][1:]:
            gram = np.asarray(json.loads(row['fit_rtc_r1_gram_json']))
            eig, vec = np.linalg.eigh(gram)
            assert eig.min() >= -1e-8 * max(1., np.diag(gram).max())
            x = vec * np.sqrt(np.maximum(eig, 0))[None, :]
            ids = json.loads(row['fit_rtc_r1_client_ids_json'])
            for i, cid in enumerate(ids):
                c = clients[row['round'], cid]
                ref = direct_reference(x, i)
                assert (ref is not None) == (c['rtc_r1_valid'] == 'True')
                if ref:
                    error = abs(ref['cosine'] - float(c['rtc_r1_cosine']))
                    max_cosine_error = max(max_cosine_error, error)
                    assert error < 1e-8
                    votes = sum(float(x[i] @ x[j] / np.linalg.norm(x[i]) / np.linalg.norm(x[j]))
                                < lock['calibration']['corroboration']['pairwise_threshold'] for j in ref['indices'])
                    assert votes == int(c['rtc_r1_corroboration_votes'])
                    assert math.ceil(2 * len(ref['indices']) / 3) == int(c['rtc_r1_corroboration_required'])
                records += 1

    unresolved = [n for n, r in receipt.items() if r['recovery'] == 'missing']
    numeric_pass = result.pop('candidate_accepted')
    result.pop('quality_accepted')
    result.update(numeric_gates_passed=numeric_pass,
                  artifact_and_mechanism_checks_passed=True,
                  source_provenance_complete=not unresolved,
                  candidate_accepted=numeric_pass if not unresolved else None,
                  acceptance_status='pending_server_sources' if unresolved else 'engineering_screen_passed' if numeric_pass else 'rejected',
                  source_receipt=receipt, unresolved_server_sources=unresolved,
                  source_lock_sha256=sha((BATCH / 'i12_lock.json').read_bytes()),
                  training_environment=lock['environment'], path_mappings=sorted(mappings),
                  independent_reference_records=records, max_cosine_error=max_cosine_error,
                  imported_batch_unchanged=True)
    result['evidence'] = {p.relative_to(BATCH).as_posix(): sha(p.read_bytes())
                          for p in sorted(BATCH.rglob('*')) if p.is_file() and p.suffix in ('.json', '.csv')}
    (HERE / 'review.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('acceptance_status', 'numeric_gates_passed',
          'independent_reference_records', 'max_cosine_error', 'unresolved_server_sources')}, indent=2))
    print('source recovery:', dict(collections.Counter(r['recovery'] for r in receipt.values())))
    print('integration gates:', len(result['checks']), 'standalone gates:', len(result['standalone_checks']))


if __name__ == '__main__':
    if not __debug__:
        raise RuntimeError('Assertions must remain enabled')
    main()
