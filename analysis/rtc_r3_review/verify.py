"""Independent R3 observation audit; reads training artifacts, never starts training."""
import hashlib
import json
import math
from pathlib import Path
import statistics as st
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.rtc_r1c_review.verify import read, rows, sha, canonical, direct_reference
from analysis.rtc_r3_diagnostics.replay import replay

HERE = Path(__file__).resolve().parent
SOURCE = ROOT / 'logs/rtc_r3_lie_observation'


def main():
    lock = read(SOURCE / 'r3_observation_lock.json')
    archive = HERE / 'r3_sources.zip'
    if not archive.exists():
        for p, h in lock['sources'].items():
            assert sha(ROOT / p) == h, p
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
            for p in lock['sources']:
                z.write(ROOT / p, p)
    with zipfile.ZipFile(archive) as z:
        assert set(z.namelist()) == set(lock['sources'])
        for p, h in lock['sources'].items():
            assert hashlib.sha256(z.read(p)).hexdigest() == h, p
    for key, base in [('artifacts', SOURCE), ('evidence', ROOT)]:
        for p, h in lock[key].items():
            assert sha(base / p) == h, p
    decision = read(SOURCE / 'analysis/decision.json')
    assert decision['quality_accepted'] and decision['observation_accepted']
    assert decision['candidate_accepted'] is False
    reports, pairs, evidence = [], {}, {}
    count = 0
    for batch in sorted({c['batch'] for c in lock['cells']}):
        gates = rows(SOURCE / batch / 'quality_gates.csv')
        assert gates and all(r['passed'] == 'True' for r in gates)
        count += len(gates)
        evidence[f'logs/rtc_r3_lie_observation/{batch}/quality_gates.csv'] = sha(SOURCE / batch / 'quality_gates.csv')
    for c in lock['cells']:
        batch, rid = SOURCE / c['batch'], c['run_id']
        status = read(batch / 'status' / (rid + '.json'))
        assert status['state'] in ('completed', 'completed_cached') and status['exit_code'] == 0 and status['last_round'] == 60
        rr = sorted(rows(batch / 'rounds' / (rid + '.csv')), key=lambda r: int(r['round']))
        cc = rows(batch / 'raw' / (rid + '_clients.csv'))
        assert [int(r['round']) for r in rr] == list(range(61))
        assert len(cc) == len({(x['round'], x['cid']) for x in cc}) == 600
        cfg = read(batch / 'raw' / (rid + '_config.json'))
        assert cfg == read(batch / 'resolved' / (rid + '.json'))
        custom = cfg['security']['defense']['custom_params']
        assert custom['spectral_direction_mode'] == custom['raw_norm_mode'] == 'observe'
        pair = read(batch / 'raw' / (rid + '_pairing_manifest.json'))
        data = read(batch / 'raw' / (rid + '_data_manifest.json'))
        trial = read(Path(pair['trial_plan_path']))
        assert canonical({k: v for k, v in data.items() if k != 'sha256'}) == data['sha256'] == pair['data_manifest_sha256']
        assert canonical({k: v for k, v in trial.items() if k != 'trial_plan_hash'}) == pair['trial_plan_hash'] == c['trial_plan_hash']
        common = {k: v for k, v in pair.items() if k != 'trial_plan_path'}
        assert pairs.setdefault(c['attack'], common) == common
        is_rtc = c['defense'] == 'rtc_r2_raw_baseline'
        max_error = 0.
        for r in rr[1:]:
            assert all(math.isfinite(float(r[k])) for k in ('server_accuracy', 'server_loss'))
            clients = {x['cid']: x for x in cc if x['round'] == r['round']}
            ids = json.loads(r['fit_rtc_r1_client_ids_json'])
            plan = trial['rounds'][int(r['round']) - 1]
            assert len(ids) == len(set(ids)) == len(clients) == 10
            assert set(ids) == set(clients) == set(plan['partition_ids'])
            assert json.loads(r['fit_completed_partition_ids_json']) == plan['partition_ids']
            assert r['fit_fit_seed_digest'] == plan['fit_seed_digest']
            for k in ('trial_plan_hash', 'attack_implementation_hash', 'attack_contract_hash'):
                assert r[k] == c[k]
            gram = np.array(json.loads(r['fit_rtc_r1_gram_json']))
            values, vectors = np.linalg.eigh(gram)
            assert min(values) >= -1e-8 * max(1., max(np.diag(gram)))
            x = vectors * np.sqrt(np.maximum(values, 0))[None, :]
            norms = [float(clients[i]['rtc_r2_raw_norm']) for i in ids]
            for i, cid in enumerate(ids):
                row = clients[cid]
                active = int(cid) in pair['malicious_partition_ids'] and c['attack'] != 'none' and int(r['round']) >= 11
                assert (row['attack_active'] == 'True') == active
                assert float(row['rtc_r1_q']) == float(row['rtc_r2_q']) == 1.
                others = [v for j, v in enumerate(norms) if i != j and v > 1e-12]
                valid = len(others) >= 9
                ratio = norms[i] / st.median(others) if valid else None
                assert (row['rtc_r2_valid'] == 'True') == valid
                if valid:
                    assert abs(float(row['rtc_r2_ratio']) - ratio) <= 1e-9 * max(1., ratio)
                assert (row['rtc_r2_flagged'] == 'True') == (valid and ratio > lock['raw_calibration']['ratio_threshold'])
                ref = direct_reference(x, i)
                assert (ref is not None) == (row['rtc_r1_valid'] == 'True')
                flagged = False
                if ref:
                    max_error = max(max_error, abs(ref['cosine'] - float(row['rtc_r1_cosine'])))
                    votes = sum(x[i] @ x[j] / np.linalg.norm(x[i]) / np.linalg.norm(x[j]) < lock['calibration']['corroboration']['pairwise_threshold'] for j in ref['indices'])
                    required = math.ceil(2 * len(ref['indices']) / 3)
                    assert int(row['rtc_r1_corroboration_votes']) == votes
                    assert int(row['rtc_r1_corroboration_required']) == required
                    flagged = ref['gap'] >= lock['calibration']['minimum_gap'] and ref['cosine'] < lock['calibration']['cosine_threshold'] and votes >= required
                assert (row['rtc_r1_flagged'] == 'True') == flagged
                if is_rtc:
                    assert float(row['aggregation_weight']) <= float(row['rtc_r2_nominal_mass']) * float(row['rtc_r2_existing_q']) + 1e-8
            if is_rtc:
                w = sum(float(v['aggregation_weight']) for v in clients.values())
                anchor, zero = float(r['fit_rtc_v3_anchor_recycle_mass']), float(r['fit_rtc_v3_zero_update_mass'])
                assert abs(w + anchor + zero - 1) < 1e-8
                assert abs(anchor - .51 * (1 - w)) < 1e-8
                assert float(r['fit_rtc_v3_max_constraint_violation']) <= 1e-8
        assert max_error < 1e-8
        start = 1 if c['attack'] == 'none' else 11
        good = [x for x in cc if int(x['round']) >= start and x['attack_active'] == 'False']
        bad = [x for x in cc if int(x['round']) >= start and x['attack_active'] == 'True']
        summary = {'attack': c['attack'], 'defense': c['defense'], 'seed': c['seed'],
                   'active_accuracy': st.mean(float(r['server_accuracy']) for r in rr if int(r['round']) >= start),
                   'final_accuracy': float(rr[-1]['server_accuracy']), 'good_n': len(good), 'bad_n': len(bad),
                   'malicious_weight_per_round': sum(float(x['aggregation_weight']) for x in bad) / (61 - start) if bad else None,
                   'spectral_good_flags': sum(x['rtc_r1_flagged'] == 'True' for x in good),
                   'spectral_bad_flags': sum(x['rtc_r1_flagged'] == 'True' for x in bad),
                   'raw_good_flags': sum(x['rtc_r2_flagged'] == 'True' for x in good),
                   'raw_bad_flags': sum(x['rtc_r2_flagged'] == 'True' for x in bad), 'independent_reference_error': max_error}
        if is_rtc:
            _, error = replay(cc, read(ROOT / custom['calibration_path'])['cumulative']['resolutions']['full'])
            summary.update(cumulative_q_error=error,
                first_bad_q_reduction=min((int(x['round']) for x in bad if float(x['rtc_v3_cumulative_q_full']) < 1 - 1e-12), default=None),
                bad_q_below1=sum(float(x['rtc_v3_cumulative_q_full']) < 1 - 1e-12 for x in bad),
                bad_weight_while_q1=sum(float(x['aggregation_weight']) for x in bad if float(x['rtc_v3_cumulative_q_full']) >= 1 - 1e-12) / sum(float(x['aggregation_weight']) for x in bad) if bad else None)
        expected = next(s for s in decision['summaries'] if s['run_id'] == rid)
        for k in ('active_accuracy', 'final_accuracy', 'malicious_weight_per_round'):
            assert summary[k] == expected[k] if summary[k] is None else abs(summary[k] - expected[k]) < 1e-12
        reports.append(summary)
    evidence.update({f'logs/rtc_r3_lie_observation/{p}': h for p, h in decision['evidence'].items()})
    evidence['logs/rtc_r3_lie_observation/analysis/decision.json'] = sha(SOURCE / 'analysis/decision.json')
    result = {'quality_accepted': True, 'observation_accepted': True, 'candidate_accepted': False,
              'runner_quality_gates': count, 'summaries': reports, 'evidence': evidence,
              'source_archive_sha256': sha(archive), 'lock_sha256': sha(SOURCE / 'r3_observation_lock.json')}
    (HERE / 'review.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'evidence'}, indent=2))


if __name__ == '__main__':
    main()
