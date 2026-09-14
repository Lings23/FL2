"""Independent R1c metrics/reference audit and immutable source archive; no training."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / 'logs/rtc_r1c_pairwise_direction'


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def rows(p):
    with p.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def canonical(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def direct_reference(x, i):
    # Explicit unit vectors and their eigenvector community, independent of production helper.
    norms = np.linalg.norm(x, axis=1)
    keep = [j for j in range(len(x)) if j != i and norms[j] > 1e-12]
    if len(keep) < 9 or norms[i] <= 1e-12:
        return None
    unit = x[keep] / norms[keep, None]
    cosine = unit @ unit.T
    np.fill_diagonal(cosine, 0.)
    values, vectors = np.linalg.eigh(cosine)
    v = vectors[:, -1]
    p, n = sum(v > 1e-8), sum(v < -1e-8)
    gap = (values[-1] - values[-2]) / max(abs(values[-1]), 1e-12)
    if values[-1] <= 1e-12 or gap <= 1e-8 or p == n:
        return None
    if p < n:
        v = -v
    selected = np.flatnonzero(v > 1e-8)
    if len(selected) < len(keep) - 3:
        return None
    weights = v[selected] / sum(v[selected])
    reference = weights @ unit[selected]
    if np.linalg.norm(reference) <= 1e-6:
        return None
    return {'cosine': float(x[i] @ reference / norms[i] / np.linalg.norm(reference)),
            'gap': float(gap), 'indices': [keep[j] for j in selected]}


def main():
    lock = read(SOURCE / 'r1c_lock.json')
    archive = HERE / 'r1c_sources.zip'
    if not archive.exists():
        for relative, expected in lock['sources'].items():
            assert sha(ROOT / relative) == expected, relative
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as f:
            for relative in lock['sources']:
                f.write(ROOT / relative, relative)
    with zipfile.ZipFile(archive) as f:
        assert set(f.namelist()) == set(lock['sources'])
        for relative, expected in lock['sources'].items():
            assert hashlib.sha256(f.read(relative)).hexdigest() == expected
    for relative, expected in lock['artifacts'].items():
        assert sha(SOURCE / relative) == expected
    for relative, expected in lock['evidence'].items():
        assert sha(ROOT / relative) == expected, relative
    decision = read(SOURCE / 'analysis/decision.json')
    evidence, reports, false_flags, pairs, clients_by_run = {}, [], [], {}, {}
    q_count = 0
    for batch in ('sign_flip_seed43', 'clean_seed43'):
        gates = rows(SOURCE / batch / 'quality_gates.csv')
        assert all(r['passed'] == 'True' for r in gates)
        q_count += len(gates)
    for cell in lock['cells']:
        batch, rid = SOURCE / cell['batch'], cell['run_id']
        status = read(batch / 'status' / (rid + '.json'))
        assert status['state'] == 'completed' and status['exit_code'] == 0 and status['last_round'] == 60
        rr = sorted(rows(batch / 'rounds' / (rid + '.csv')), key=lambda r: int(r['round']))
        cc = rows(batch / 'raw' / (rid + '_clients.csv'))
        assert len(rr) == 61 and [int(r['round']) for r in rr] == list(range(61))
        assert len(cc) == 600 and len({(c['round'], c['cid']) for c in cc}) == 600
        pair = read(batch / 'raw' / (rid + '_pairing_manifest.json'))
        data = read(batch / 'raw' / (rid + '_data_manifest.json'))
        assert canonical({k:v for k,v in data.items() if k != 'sha256'}) == data['sha256'] == pair['data_manifest_sha256']
        trial = read(Path(pair['trial_plan_path']))
        assert canonical({k:v for k,v in trial.items() if k != 'trial_plan_hash'}) == pair['trial_plan_hash'] == cell['trial_plan_hash']
        pairing_content = {k:v for k,v in pair.items() if k != 'trial_plan_path'}
        assert pairs.setdefault(cell['attack'], pairing_content) == pairing_content
        max_error = 0.
        for r in rr[1:]:
            ids = json.loads(r['fit_rtc_r1_client_ids_json'])
            rc = {c['cid']: c for c in cc if c['round'] == r['round']}
            assert len(rc) == len(ids) == len(set(ids)) == 10 and set(rc) == set(ids)
            p = trial['rounds'][int(r['round']) - 1]
            assert json.loads(r['fit_completed_partition_ids_json']) == p['partition_ids'] and r['fit_fit_seed_digest'] == p['fit_seed_digest']
            gram = np.array(json.loads(r['fit_rtc_r1_gram_json']))
            values, vectors = np.linalg.eigh(gram)
            assert min(values) >= -1e-8 * max(1, np.max(np.diag(gram)))
            x = vectors * np.sqrt(np.maximum(values, 0))[None, :]
            for i, cid in enumerate(ids):
                reference = direct_reference(x, i)
                row = rc[cid]
                assert (reference is not None) == (row['rtc_r1_valid'] == 'True')
                if reference:
                    max_error = max(max_error, abs(reference['cosine'] - float(row['rtc_r1_cosine'])))
                    usable = reference['gap'] >= lock['calibration']['minimum_gap']
                    flagged = usable and reference['cosine'] < lock['calibration']['cosine_threshold']
                else:
                    usable = flagged = False
                if reference:
                    refs = reference['indices']
                    unit = x / np.linalg.norm(x, axis=1)[:, None]
                    votes = sum(float(unit[i] @ unit[j]) < lock['calibration']['corroboration']['pairwise_threshold'] for j in refs)
                    required = (2 * len(refs) + 2) // 3
                else:
                    votes = required = 0
                assert votes == int(row['rtc_r1_corroboration_votes'])
                assert required == int(row['rtc_r1_corroboration_required'])
                assert flagged == (row['rtc_r1_base_flagged'] == 'True')
                flagged = flagged and required > 0 and votes >= required
                assert usable == (row['rtc_r1_usable'] == 'True') and flagged == (row['rtc_r1_flagged'] == 'True')
                if cell['defense'] == 'rtc_r1c_pairwise_cap' and flagged and row['attack_active'] == 'False':
                    refs = json.loads(row['rtc_r1_reference_ids_json'])
                    false_flags.append({'attack': cell['attack'], 'round': int(r['round']), 'cid': cid,
                        'cosine': float(row['rtc_r1_cosine']), 'weight': float(row['aggregation_weight']),
                        'reference_attackers': sum(rc[k]['attack_active'] == 'True' for k in refs),
                        'round_attackers': sum(c['attack_active'] == 'True' for c in rc.values())})
        assert max_error < 1e-8
        start = 1 if cell['attack'] == 'none' else 11
        window = [c for c in cc if int(c['round']) >= start]
        good = [c for c in window if c['attack_active'] == 'False']
        bad = [c for c in window if c['attack_active'] == 'True']
        report = {**cell, 'active_accuracy': statistics.mean(float(r['server_accuracy']) for r in rr if int(r['round']) >= start),
                  'final_accuracy': float(rr[-1]['server_accuracy']),
                  'benign_flagged': sum(c['rtc_r1_flagged'] == 'True' for c in good), 'benign_records': len(good),
                  'malicious_flagged': sum(c['rtc_r1_flagged'] == 'True' for c in bad), 'malicious_records': len(bad),
                  'malicious_weight_per_round': sum(float(c['aggregation_weight']) for c in bad)/(61-start) if bad else None,
                  'coordinate_cosine_max_error': max_error}
        original = next(s for s in decision['summaries'] if s['run_id'] == rid)
        for key in ('active_accuracy', 'final_accuracy', 'malicious_weight_per_round'):
            assert report[key] is None and original[key] is None or report[key] is not None and abs(report[key]-original[key]) < 1e-12
        assert abs(report['benign_flagged']/len(good) - original['benign_flag_rate']) < 1e-12
        reports.append(report)
        clients_by_run[(cell['attack'], cell['defense'])] = {(c['round'], c['cid']): c for c in cc}
        for path in batch.rglob('*'):
            if path.is_file() and path.suffix in ('.json', '.csv'):
                evidence[str(path.relative_to(ROOT)).replace('\\', '/')] = sha(path)
    base = clients_by_run['sign_flip', 'rtc_r1c_pairwise_baseline']
    cap = clients_by_run['sign_flip', 'rtc_r1c_pairwise_cap']
    matched = [k for k,c in cap.items() if int(k[0]) >= 11 and c['attack_active'] == 'True'
               and c['rtc_r1_usable'] == base[k]['rtc_r1_usable'] == 'True']
    projection = {}
    for label, clients in [('baseline',base), ('candidate',cap)]:
        total = sum(float(clients[k]['aggregation_weight']) * max(0, -float(clients[k]['rtc_r1_projection'])) for k in matched)/50
        assert abs(total-decision['negative_proxy'][label]) < 1e-12
        projection[label] = total
    projection['matched_rows'] = len(matched)
    by = {(r['attack'], r['defense']):r for r in reports}
    base_r, cap_r = [by['sign_flip', 'rtc_r1c_pairwise_'+s] for s in ('baseline','cap')]
    clean_b, clean_c = [by['none', 'rtc_r1c_pairwise_'+s] for s in ('baseline','cap')]
    checks = {
        'sign_active_gain': cap_r['active_accuracy']-base_r['active_accuracy'] >= .01,
        'sign_final_gain': cap_r['final_accuracy'] >= base_r['final_accuracy'],
        'clean_active': clean_c['active_accuracy']-clean_b['active_accuracy'] >= -.002,
        'clean_final': clean_c['final_accuracy']-clean_b['final_accuracy'] >= -.002,
        'sign_benign_flags': cap_r['benign_flagged']/cap_r['benign_records'] <= .01,
        'clean_benign_flags': clean_c['benign_flagged']/clean_c['benign_records'] <= .01,
        'malicious_weight_decreased': cap_r['malicious_weight_per_round'] < base_r['malicious_weight_per_round'],
        'paired_projection_coverage': len(matched)/cap_r['malicious_records'] >= .5,
        'negative_proxy_decreased': projection['candidate'] < projection['baseline'],
    }
    assert all(checks.values()) and decision['candidate_accepted'] is True
    for key, value in checks.items():
        assert decision['checks'][key] == value
    clean_base = clients_by_run['none', 'rtc_r1c_pairwise_baseline']
    clean_cap = clients_by_run['none', 'rtc_r1c_pairwise_cap']
    assert clean_base.keys() == clean_cap.keys()
    assert all(clean_base[k]['aggregation_weight'] == clean_cap[k]['aggregation_weight'] for k in clean_base)
    evidence['logs/rtc_r1c_pairwise_direction/analysis/decision.json'] = sha(SOURCE / 'analysis/decision.json')
    result = {'stage': 'R1c', 'quality_accepted': True, 'candidate_accepted': True, 'independent_checks': checks, 'runner_quality_gates': q_count,
              'reports': reports, 'false_flags': false_flags, 'negative_proxy': projection,
              'source_archive_sha256': sha(archive), 'evidence': evidence}
    (HERE / 'review.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='evidence'}, indent=2))


if __name__ == '__main__':
    main()
