"""H2b memory regression tests with synthetic updates only."""
import numpy as np
from defenses.rtc.reference_eligibility import ReferenceEligibility


def test_abstention_preserves_old_memory_and_raw_rejection_takes_precedence():
    h = ReferenceEligibility()
    h.commit({'good': False, 'bad': True})
    ids = ['good', 'bad', 'unknown', 'raw', 'shared', 'shared']
    original = [True] * len(ids)
    raw = [dict(rtc_r2_flagged=(p == 'raw')) for p in ids]
    veto = [dict(rtc_r1e_vetoed=True) for _ in ids]
    veto[-1]['rtc_r1e_vetoed'] = False
    nxt = h.transition_confirmed(ids, original, raw, veto)
    assert nxt == {'good': False, 'bad': True, 'raw': True, 'shared': True}
    assert h.state_dict() == {'good': False, 'bad': True}
    normal = h.transition_confirmed(['bad'], [False], [dict(rtc_r2_flagged=False)], [dict(rtc_r1e_vetoed=False)])
    assert not normal['bad']


def test_phase6_vetoed_good_clients_do_not_become_rejected_references():
    from tests.test_rtc_v3_semantic_temporal_exposure import _manifest, _params, ROLES, NAMES
    from config.config_loader import DefenseConfig
    from defenses.rtc.v3 import RTCv3Defense
    from defenses.rtc.calibration import content_hash
    ids = list(map(str, range(10)))
    manifest = _manifest(frozen_cones=True)
    manifest['cumulative'] = {'enabled': True, 'resolutions': {'full': dict(scale_center=1.,
        scale_lower=1., scale_upper=1., kappa=.5, threshold=.5, eta=2., q_min=.5)}}
    manifest['content_hash'] = content_hash(manifest)
    params = dict(calibration_manifest=manifest, implementation_phase=6, parameter_roles=ROLES,
        principal_map={i: i for i in ids}, principal_first_sampling_verified=True, semantic_intervention_risk_floor=.5,
        cumulative_q_cap_power=1, anchor_recycle_fraction=.51, anchor_recycle_weighting='accepted',
        spectral_direction_mode='cap', spectral_direction_calibration='config/rtc_r1c_pairwise_calibration.json',
        raw_norm_mode='cap', raw_norm_calibration='config/rtc_r2_raw_norm_calibration.json',
        reference_eligibility_mode='cap', reference_eligibility_memory='confirmed')
    d = RTCv3Defense(DefenseConfig(enabled=True, type='rtc_v3_candidate', custom_params=params), num_clients=10)
    d._reference_eligibility.commit({i: int(i) >= 4 for i in ids})
    base = _params()
    d.set_context(1, ids, base, principal_ids=ids, parameter_roles=ROLES,
        parameter_names=NAMES, trainable_parameter_indices=[0, 1, 2])
    updates = [([base[0] + np.ones(4, np.float32) * s * .02, base[1].copy(), base[2].copy()], 10)
        for s in [1] * 4 + [-1] * 6]
    d.aggregate(updates)
    assert d.last_round_metrics['rtc_v3_max_constraint_violation'] <= 1e-8
    assert d.last_round_metrics['rtc_r1e_memory'] == 'confirmed'
    assert all(d._last_spectral_rows[i]['rtc_r1e_vetoed'] for i in range(4))
    assert all(not d._reference_eligibility.state_dict()[str(i)] for i in range(4))
