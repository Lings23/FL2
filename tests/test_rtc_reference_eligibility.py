"""H2 synthetic state and aggregation tests, without any client training."""
import copy
import json
import numpy as np
import pytest
from defenses.rtc.reference_eligibility import ReferenceEligibility
from defenses.rtc.spectral_direction import measure, load_calibration


def measured(signs):
    x = [[np.array([float(s), .01 * float(s)])] for s in signs]
    return measure(x, [0], load_calibration('config/rtc_r1c_pairwise_calibration.json'), 'cap')


def test_unknown_does_not_disable_detection_and_state_is_transactional():
    h = ReferenceEligibility()
    ids = list(map(str, range(10)))
    m = measured([1] * 7 + [-1] * 3)
    rows, original = h.prepare(m, ids, 'cap')
    assert all(m['rows'][i]['flagged'] for i in (7, 8, 9))
    assert not any(r['rtc_r1e_previous_known'] for r in rows)
    nxt = h.transition(ids, original, [dict(rtc_r2_flagged=False)] * 10)
    assert h.state_dict() == {}
    h.commit(nxt)
    nxt['0'] = True
    assert not h.state_dict()['0']
    # Absence preserves last participation, a clean upload clears it.
    nxt = h.transition(['7'], [False], [dict(rtc_r2_flagged=False)])
    assert nxt['8'] and not nxt['7']
    assert h.state_dict()['7']
    with pytest.raises(ValueError):
        h.transition(ids, [], [])
    with pytest.raises(ValueError):
        h.prepare(m, ids, 'wrong')


def test_contaminated_reference_veto_and_unique_principal_votes():
    h = ReferenceEligibility()
    ids = list(map(str, range(10)))
    h.commit({str(i): True for i in range(4, 10)})
    m = measured([1] * 4 + [-1] * 6)
    rows, original = h.prepare(m, ids, 'cap')
    assert all(original[:4]) and all(rows[i]['rtc_r1e_vetoed'] for i in range(4))
    assert all(m['rows'][i]['q'] == 1 for i in range(4))
    # Observe neither mutates decisions nor commits state.
    m = measured([1] * 4 + [-1] * 6)
    before = copy.deepcopy(m['rows'])
    h.prepare(m, ids, 'observe')
    assert m['rows'] == before
    duplicate_ids = ['target', 'a', 'a', 'a', 'b', 'c', 'd', 'e', 'f', 'g']
    m = measured([1] * 4 + [-1] * 6)
    m['rows'][0].update(reference_indices=[1, 2, 3, 4, 5], flagged=True, q=0.)
    h.commit({'a': True})
    rows, _ = h.prepare(m, duplicate_ids, 'cap')
    assert json.loads(rows[0]['rtc_r1e_reference_principals_json']) == ['a', 'b', 'c']
    assert rows[0]['rtc_r1e_eligible_count'] == rows[0]['rtc_r1e_required'] == 2
    assert rows[0]['rtc_r1e_passes']
    nxt = h.transition(['a', 'a'], [False, True], [dict(rtc_r2_flagged=False)] * 2)
    assert nxt['a']


def test_phase6_observer_inert_and_cap_preserves_other_budgets():
    from tests.test_rtc_v3_semantic_temporal_exposure import _manifest, _params, ROLES, NAMES
    from config.config_loader import DefenseConfig
    from defenses.rtc.v3 import RTCv3Defense
    from defenses.rtc.calibration import content_hash
    ids = list(map(str, range(10)))
    manifest = _manifest(frozen_cones=True)
    manifest['cumulative'] = {'enabled': True, 'resolutions': {'full': dict(
        scale_center=1., scale_lower=1., scale_upper=1., kappa=.5, threshold=.5, eta=2., q_min=.5)}}
    manifest['content_hash'] = content_hash(manifest)
    params = dict(calibration_manifest=manifest, implementation_phase=6, parameter_roles=ROLES,
        principal_map={i: i for i in ids}, principal_first_sampling_verified=True, semantic_intervention_risk_floor=.5,
        cumulative_q_cap_power=1, anchor_recycle_fraction=.51, anchor_recycle_weighting='accepted',
        spectral_direction_mode='cap', spectral_direction_calibration='config/rtc_r1c_pairwise_calibration.json',
        raw_norm_mode='cap', raw_norm_calibration='config/rtc_r2_raw_norm_calibration.json')
    defenses = [RTCv3Defense(DefenseConfig(enabled=True, type='rtc_v3_candidate', custom_params={
        **params, 'reference_eligibility_mode': mode}), num_clients=10) for mode in ('off', 'observe', 'cap')]
    base = _params()
    for rd, signs in [(1, [1] * 7 + [-1] * 3), (2, [1] * 4 + [-1] * 6)]:
        if rd == 2:
            for d in defenses[1:]:
                d._reference_eligibility.commit({str(i): True for i in range(4, 10)})
        outputs = []
        for d in defenses:
            d.set_context(rd, ids, base, principal_ids=ids, parameter_roles=ROLES,
                parameter_names=NAMES, trainable_parameter_indices=[0, 1, 2])
            updates = [([base[0] + np.ones(4, np.float32) * s * .02, base[1].copy(), base[2].copy()], 10) for s in signs]
            outputs.append(d.aggregate(updates))
            assert d.last_round_metrics['rtc_v3_max_constraint_violation'] <= 1e-8
            for cid, row, raw in zip(ids, d._last_spectral_rows, d._last_raw_norm_rows):
                assert d.last_client_aggregation_weights[cid] <= raw['rtc_r2_nominal_mass'] * min(
                    raw['rtc_r2_q'], raw['rtc_r2_existing_q']) + 1e-8
                assert raw['rtc_r2_q'] in (0., 1.)
        assert all(np.array_equal(a, b) for a, b in zip(outputs[0], outputs[1]))
        assert defenses[0].last_client_aggregation_weights == defenses[1].last_client_aggregation_weights
        assert defenses[0]._cumulative.state_dict() == defenses[1]._cumulative.state_dict()
        base = outputs[0]
    rows = defenses[2]._last_spectral_rows
    assert all(rows[i]['rtc_r1e_vetoed'] and not rows[i]['rtc_r1_flagged'] for i in range(4))
    assert all(defenses[2]._reference_eligibility.state_dict()[str(i)] for i in range(4))
    with pytest.raises(ValueError, match='requires'):
        RTCv3Defense(DefenseConfig(enabled=True, type='rtc_v3_candidate', custom_params={
            **params, 'reference_eligibility_mode': 'cap', 'reference_guard_mode': 'cap'}), num_clients=10)
