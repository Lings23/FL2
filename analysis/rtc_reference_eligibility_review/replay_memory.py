"""Offline H2b state rule on old trajectories, without training or ACC prediction."""
from pathlib import Path
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def replay():
    batch = ROOT / 'logs/rtc_reference_eligibility'
    review = json.loads((HERE / 'review.json').read_text())
    lock_bytes = (batch / 'r1e_lock.json').read_bytes()
    assert hashlib.sha256(lock_bytes).hexdigest() == review['source_lock_sha256']
    lock = json.loads(lock_bytes)
    results = []
    for cell in lock['cells']:
        if cell['defense'] not in ('rtc_i12_eligibility_observe', 'rtc_i12_eligibility_cap'):
            continue
        path = batch / cell['batch'] / 'raw' / (cell['run_id'] + '_clients.csv')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == review['evidence'][path.relative_to(batch).as_posix()]
        with path.open(encoding='utf-8-sig') as stream:
            rows = list(csv.DictReader(stream))
        assert all(r['principal_id'] == r['cid'] for r in rows)
        state, changed = {}, []
        counts = dict(good_original=0, good_kept=0, bad_original=0, bad_kept=0)
        for rd in range(1, 61):
            clients = [r for r in rows if int(r['round']) == rd]
            next_state = dict(state)
            for row in clients:
                pid = row['principal_id']
                refs = set(json.loads(row['rtc_r1_reference_ids_json'])) - {pid}
                passes = bool(refs) and sum(not state.get(j, False) for j in refs) >= (2 * len(refs) + 2) // 3
                original = row['rtc_r1e_original_flagged'].lower() == 'true'
                raw = row['rtc_r2_flagged'].lower() == 'true'
                veto, effective = original and not passes, original and passes
                if raw:
                    next_state[pid] = True
                elif not veto:
                    next_state[pid] = original
                if rd >= (1 if cell['attack'] == 'none' else 11):
                    bad = row['attack_active'].lower() == 'true'
                    prefix = 'bad' if bad else 'good'
                    counts[prefix + '_original'] += original
                    counts[prefix + '_kept'] += effective
                    if effective != (row['rtc_r1_flagged'].lower() == 'true'):
                        changed.append(dict(round=rd, cid=pid, attack_active=bad, new_flag=effective))
            state = next_state
        results.append(dict(batch=cell['batch'], defense=cell['defense'], counts=counts, changed=changed))
    return dict(rule='Keep previous rejection state when R1c is vetoed, unless current R2 rejects. No threshold changes.',
        source_lock_sha256=review['source_lock_sha256'], closed_loop=False, results=results)


if __name__ == '__main__':
    if not __debug__:
        raise RuntimeError('Assertions must remain enabled')
    result = replay()
    (HERE / 'h2b_offline.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(trajectories=len(result['results']), closed_loop=False)))
