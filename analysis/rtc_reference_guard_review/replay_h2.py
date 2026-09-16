"""Fixed H2 rule on frozen parent trajectories; not a closed-loop experiment."""
from pathlib import Path
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def replay():
    batch = ROOT / 'logs/rtc_reference_guard'
    review = json.loads((HERE / 'review.json').read_text())
    lock_bytes = (batch / 'r1g_lock.json').read_bytes()
    assert hashlib.sha256(lock_bytes).hexdigest() == review['source_lock_sha256']
    lock = json.loads(lock_bytes)
    results = []
    for cell in lock['cells']:
        if cell['defense'] != 'rtc_i12_guard_observe':
            continue
        path = batch / cell['batch'] / 'raw' / (cell['run_id'] + '_clients.csv')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == review['evidence'][path.relative_to(batch).as_posix()]
        with path.open(encoding='utf-8-sig') as stream:
            rows = list(csv.DictReader(stream))
        assert all(r['principal_id'] == r['cid'] for r in rows)
        state, veto = {}, []
        counts = dict(benign_original=0, benign_kept=0, malicious_original=0, malicious_kept=0)
        for rd in range(1, 61):
            clients = [r for r in rows if int(r['round']) == rd]
            for r in clients:
                refs = set(json.loads(r['rtc_r1_reference_ids_json'])) - {r['principal_id']}
                eligible = sum(not state.get(j, False) for j in refs)
                passes = bool(refs) and eligible >= (2 * len(refs) + 2) // 3
                flagged = r['rtc_r1_flagged'].lower() == 'true'
                bad = r['attack_active'].lower() == 'true'
                if rd >= (1 if cell['attack'] == 'none' else 11) and flagged:
                    prefix = 'malicious' if bad else 'benign'
                    counts[prefix + '_original'] += 1
                    counts[prefix + '_kept'] += passes
                    if not passes:
                        veto.append(dict(round=rd, cid=r['cid'], attack_active=bad, eligible=eligible, total=len(refs)))
            state.update({r['principal_id']: r['rtc_r1_flagged'].lower() == 'true' or
                r['rtc_r2_flagged'].lower() == 'true' for r in clients})
        results.append(dict(batch=cell['batch'], counts=counts, veto=veto))
    return dict(rule='h2_design.json, fixed two-thirds, last original R1c/R2 flag; no tuning',
        source_lock_sha256=review['source_lock_sha256'], closed_loop=False, results=results)


if __name__ == '__main__':
    if not __debug__:
        raise RuntimeError('Assertions must remain enabled')
    result = replay()
    (HERE / 'h2_offline.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
