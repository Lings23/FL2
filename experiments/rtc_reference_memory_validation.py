"""H2b strict artifact validation: original R1c evidence and principal eligibility.
Historical R2/R3 analyzers remain unchanged. No training entry point.
"""
import json
import math
from pathlib import Path
import numpy as np
from experiments.rtc_r2_raw_norm import RUNNER_GATES, GATES, canonical
from experiments.rtc_r0b_observation import read_json, read_csv, truth
from defenses.rtc.spectral_direction import score_gram, VERSION
from defenses.rtc import raw_norm
BASE, DIRECTION, NORM, COMBINED, MK, RFA = ('rtc_i12_' + x for x in ('b0', 'direction', 'norm', 'combined', 'multikrum', 'rfa'))
RTC = (BASE, DIRECTION, NORM, COMBINED)
MODES = {BASE: ('observe', 'observe'), DIRECTION: ('cap', 'observe'),
         NORM: ('observe', 'cap'), COMBINED: ('cap', 'cap'),
         MK: ('observe', 'observe'), RFA: ('observe', 'observe')}

def verify_cell(root, c, lock):
    batch, rid = root / c['batch'], c['run_id']
    s = read_json(batch / 'status' / (rid + '.json'))
    assert s['state'] in ('completed', 'completed_cached') and s['exit_code'] == 0 and s['last_round'] == 60
    assert read_json(batch / 'resolved' / (rid + '.json')) == read_json(batch / 'raw' / (rid + '_config.json'))
    rr, cc = read_csv(batch / 'rounds' / (rid + '.csv')), read_csv(batch / 'raw' / (rid + '_clients.csv'))
    assert len(rr) == 61 and {int(r['round']) for r in rr} == set(range(61))
    rr.sort(key=lambda r: int(r['round']))
    assert len(cc) == 600 and len({(r['round'], r['cid']) for r in cc}) == 600
    if c['defense'] in RTC:
        assert all(r['principal_id'] == r['cid'] for r in cc), 'Protocol requires one principal per client'
    q = read_csv(batch / 'quality_gates.csv')
    expected_gates = RUNNER_GATES | ({'attack_execution:' + x['run_id'] for x in lock['cells'] if x['batch'] == c['batch']} if c['attack'] != 'none' else set())
    assert expected_gates <= {r['gate'] for r in q} and all(truth(r['passed']) for r in q)
    pair = read_json(batch / 'raw' / (rid + '_pairing_manifest.json'))
    data = read_json(batch / 'raw' / (rid + '_data_manifest.json'))
    trial = read_json(Path(pair['trial_plan_path']))
    assert canonical({k: v for k, v in data.items() if k != 'sha256'}) == data['sha256'] == pair['data_manifest_sha256']
    assert canonical({k: v for k, v in trial.items() if k != 'trial_plan_hash'}) == pair['trial_plan_hash'] == c['trial_plan_hash']
    assert pair['pairing_mode'] == 'strict' and pair['sampling_protocol'] == 'principal_uniform' and pair['deterministic_client_training'] is True
    eligibility_state = {}
    for r in rr:
        for key in ('server_accuracy', 'server_loss'):
            assert math.isfinite(float(r[key]))
        if int(r['round']) == 0:
            continue
        for key in ('trial_plan_hash', 'attack_implementation_hash', 'attack_contract_hash'):
            assert r[key] == c[key]
        for key in ('fit_received_updates_finite', 'fit_coordinated_updates_finite', 'fit_aggregate_parameters_finite'):
            assert truth(r[key])
        assert r['fit_rtc_r1_version'] == VERSION and r['fit_rtc_r1_calibration_hash'] == lock['calibration_hash']
        direction_mode, raw_mode = MODES[c['defense']]
        assert r['fit_rtc_r1_mode'] == direction_mode
        for key in ('fit_rtc_r1_reconstruction_error', 'fit_rtc_r1_actual_aggregate_norm',
                    'fit_rtc_r1_anchor_contribution_norm', 'fit_rtc_r1_client_sum_norm', 'fit_aggregation_time_seconds'):
            assert math.isfinite(float(r[key])) and float(r[key]) >= 0, key
        assert float(r['fit_rtc_r1_reconstruction_error']) / max(float(r['fit_rtc_r1_actual_aggregate_norm']), 1e-8) <= GATES['reconstruction_relative_error_max']
        ids = json.loads(r['fit_rtc_r1_client_ids_json'])
        plan = trial['rounds'][int(r['round']) - 1]
        assert json.loads(r['fit_completed_partition_ids_json']) == plan['partition_ids']
        assert r['fit_fit_seed_digest'] == plan['fit_seed_digest']
        clients = {row['cid']: row for row in cc if row['round'] == r['round']}
        assert len(ids) == len(set(ids)) == len(clients) == 10 and set(ids) == set(clients) == set(plan['partition_ids'])
        gram = np.asarray(json.loads(r['fit_rtc_r1_gram_json']))
        scores = score_gram(gram, lock['calibration']['f'])
        assert r['fit_rtc_r2_version'] == raw_norm.VERSION
        assert r['fit_rtc_r2_calibration_hash'] == lock['raw_calibration_hash']
        raw_mode = MODES[c['defense']][1]
        assert r['fit_rtc_r2_mode'] == raw_mode
        assert math.isfinite(float(r['fit_rtc_r2_scoring_seconds'])) and float(r['fit_rtc_r2_scoring_seconds']) >= 0
        norms = [float(clients[cid]['rtc_r2_raw_norm']) for cid in ids]
        assert all(math.isfinite(v) and v >= 0 for v in norms)
        # Recompute leave-one-out medians without production score().
        import statistics
        for i, cid in enumerate(ids):
            row = clients[cid]
            clip_factor = float(row['rtc_r2_clip_factor'])
            assert math.isfinite(clip_factor) and 0 <= clip_factor <= 1
            assert abs(norms[i]*clip_factor - float(row['rtc_r1_trainable_clipped_norm'])) <= 1e-8*max(1,norms[i]*clip_factor)
            if c['defense'] in (MK, RFA):
                assert clip_factor == 1.
            others = [v for j,v in enumerate(norms) if j != i and v > 1e-12]
            valid = len(others) >= 9
            median = statistics.median(others) if valid else None
            ratio = norms[i]/median if valid else None
            flag = valid and ratio > lock['raw_calibration']['ratio_threshold']
            assert int(row['rtc_r2_reference_count']) == len(others)
            assert truth(row['rtc_r2_valid']) == valid and truth(row['rtc_r2_flagged']) == flag
            for key, value in [('rtc_r2_reference_median',median), ('rtc_r2_ratio',ratio)]:
                if value is None:
                    assert row[key] == ''
                else:
                    assert math.isfinite(float(row[key])) and abs(float(row[key])-value) <= 1e-10*max(1,abs(value))
            q = 0. if flag and raw_mode == 'cap' else 1.
            assert float(row['rtc_r2_q']) == q
            if c['defense'] in RTC:
                previous = min(float(row['rtc_r1_existing_q']), float(row['rtc_r1_q']))
                assert float(row['rtc_r2_existing_q']) == previous
                assert truth(row['rtc_r2_applied']) == (q < previous)
                assert float(row['rtc_r2_nominal_mass']) == float(row['rtc_r1_nominal_mass'])
                assert float(row['aggregation_weight']) <= float(row['rtc_r2_nominal_mass'])*min(q,previous)+1e-8

        for i, cid in enumerate(ids):
            row, score = clients[cid], scores[i]
            identity = int(cid) in pair['malicious_partition_ids']
            assert truth(row['is_malicious']) == identity
            assert truth(row['attack_active']) == (identity and c['attack'] != 'none' and int(r['round']) >= 11)
            assert math.isfinite(float(row['aggregation_weight'])) and 0 <= float(row['aggregation_weight']) <= 1
            assert abs(float(row['rtc_r1_trainable_clipped_norm']) ** 2 - gram[i, i]) <= 1e-8 * max(1, gram[i, i])
            valid = score['valid'] and score['gap'] >= lock['calibration']['minimum_gap']
            flag = valid and score['cosine'] < lock['calibration']['cosine_threshold']
            from defenses.rtc.corroboration import corroboration
            guard = corroboration(gram, i, score['reference_indices'], lock['calibration']['corroboration']['pairwise_threshold'])
            assert truth(row['rtc_r1_base_flagged']) == flag
            assert int(row['rtc_r1_corroboration_votes']) == guard['votes']
            assert int(row['rtc_r1_corroboration_required']) == guard['required']
            assert truth(row['rtc_r1_corroboration_passes']) == guard['passes']
            flag = flag and guard['passes']
            if c['defense'] in RTC:
                assert truth(row['rtc_r1e_original_flagged']) == flag
                assert float(row['rtc_r1e_original_q']) == (0. if flag and direction_mode == 'cap' else 1.)
                # This protocol freezes one authenticated principal per client.
                refs = sorted({ids[j] for j in score['reference_indices']})
                rejected = [p for p in refs if eligibility_state.get(p, False)]
                eligible = len(refs) - len(rejected)
                required = (2 * len(refs) + 2) // 3
                passes = bool(refs) and eligible >= required
                assert json.loads(row['rtc_r1e_reference_principals_json']) == refs
                assert json.loads(row['rtc_r1e_rejected_principals_json']) == rejected
                assert int(row['rtc_r1e_known_count']) == sum(p in eligibility_state for p in refs)
                assert int(row['rtc_r1e_eligible_count']) == eligible
                assert int(row['rtc_r1e_required']) == required
                assert truth(row['rtc_r1e_previous_known']) == (cid in eligibility_state)
                assert truth(row['rtc_r1e_previous_rejected']) == eligibility_state.get(cid, False)
                assert truth(row['rtc_r1e_passes']) == passes
                mode = 'observe' if c['defense'] == 'rtc_i12_eligibility_observe' else 'cap'
                assert r['fit_rtc_r1e_mode'] == mode and r['fit_rtc_r1e_version'] == 'rtc_reference_eligibility.v1'
                assert float(r['fit_rtc_r1e_quorum_numerator']) == 2 and float(r['fit_rtc_r1e_quorum_denominator']) == 3
                expected_memory = 'confirmed' if c['defense'] == 'rtc_i12_eligibility_confirmed' else 'original'
                assert r['fit_rtc_r1e_memory'] == expected_memory
                assert truth(row['rtc_r1e_vetoed']) == (mode == 'cap' and flag and not passes)
                if mode == 'cap':
                    flag = flag and passes
            assert truth(row['rtc_r1_valid']) == score['valid'] and truth(row['rtc_r1_usable']) == valid
            assert truth(row['rtc_r1_flagged']) == flag
            assert json.loads(row['rtc_r1_reference_ids_json']) == [ids[j] for j in score['reference_indices']]
            assert np.allclose(json.loads(row['rtc_r1_reference_weights_json']), score['reference_weights'], atol=1e-10, rtol=0)
            assert float(row['rtc_r1_q']) == (0. if flag and direction_mode == 'cap' else 1.)
            if score['valid']:
                assert abs(float(row['rtc_r1_cosine']) - score['cosine']) < 1e-8
            if valid:
                assert all(math.isfinite(float(row[k])) for k in ('rtc_r1_anchor_projection', 'rtc_r1_actual_projection', 'rtc_r1_weighted_projection'))
                projection = math.sqrt(max(0, gram[i, i])) * score['cosine']
                assert abs(float(row['rtc_r1_projection']) - projection) <= 1e-8 * max(1., abs(projection))
                negative = float(row['aggregation_weight']) * max(0, -projection)
                assert abs(float(row['rtc_r1_negative_contribution']) - negative) <= 1e-8 * max(1., negative)
            else:
                assert row['rtc_r1_projection'] == row['rtc_r1_negative_contribution'] == ''
            if c['defense'] in RTC:
                assert float(row['aggregation_weight']) <= float(row['rtc_r1_nominal_mass']) * min(float(row['rtc_r1_existing_q']), float(row['rtc_r1_q'])) + 1e-8
                assert truth(row['rtc_r1_applied']) == (float(row['rtc_r1_q']) < float(row['rtc_r1_existing_q']))
        if c['defense'] in RTC:
            for cid, row in clients.items():
                raw = truth(row['rtc_r2_flagged'])
                if raw or c['defense'] != 'rtc_i12_eligibility_confirmed' or not truth(row['rtc_r1e_vetoed']):
                    eligibility_state[cid] = raw or truth(row['rtc_r1e_original_flagged'])
            for key, value in {'semantic_intervention_risk_floor': .5, 'cumulative_q_cap_power': 1,
                               'anchor_recycle_fraction': .51, 'norm_clip_mad_k': 2.5}.items():
                assert float(r['fit_rtc_v3_' + key]) == value
            assert r['fit_rtc_v3_anchor_recycle_weighting'] == 'accepted'
            assert float(r['fit_rtc_v3_max_constraint_violation']) <= GATES['budget_violation_max']
    if c['defense'] in RTC:
        from analysis.rtc_r3_diagnostics.replay import replay
        config = read_json(batch / 'resolved' / (rid + '.json'))
        path = Path(config['security']['defense']['custom_params']['calibration_path'])
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[1] / path
        replay(cc, read_json(path)['cumulative']['resolutions']['full'])
        for r in rr[1:]:
            w = sum(float(x['aggregation_weight']) for x in cc if x['round'] == r['round'])
            a, z = float(r['fit_rtc_v3_anchor_recycle_mass']), float(r['fit_rtc_v3_zero_update_mass'])
            assert abs(w + a + z - 1) <= 1e-8
            assert abs(a - .51 * (1 - w)) <= 1e-8
            assert abs(w + a - float(r['fit_rtc_v3_effective_update_mass'])) <= 1e-8
    return {'cell': c, 'rounds': rr, 'clients': cc, 'pair': pair}
