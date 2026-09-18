"""Independent G1/G2 per-principal replay from CSV; no training entry point."""
from decimal import Decimal
import math
import statistics
from defenses.rtc.lower_tail import VERSION, calibration_hash
from experiments.rtc_r0b_observation import truth


def integer(value):
    d=Decimal(str(value))
    assert d.is_finite() and d==d.to_integral_value(),value
    return int(d)


def number(actual,expected):
    if expected is None:assert actual in ('',None)
    else:
        v=float(actual)
        assert math.isfinite(v) and abs(v-expected)<=1e-10*max(1.,abs(expected))


def verify_lower(run,calibration,mode,policy):
    assert mode in ('observe','cap') and policy in ('all','raw_eligible')
    state={}
    for r in run['rounds'][1:]:
        assert r['fit_rtc_r3l_version']==VERSION
        assert r['fit_rtc_r3l_mode']==mode and r['fit_rtc_r3l_reference_policy']==policy
        assert r['fit_rtc_r3l_calibration_hash']==calibration_hash(calibration)
        assert math.isfinite(float(r['fit_rtc_r3l_seconds'])) and float(r['fit_rtc_r3l_seconds'])>=0
        clients=[c for c in run['clients'] if c['round']==r['round']]
        assert len(clients)==10 and len({c['principal_id'] for c in clients})==10
        norms={c['principal_id']:float(c['rtc_r3l_residual_norm']) for c in clients}
        assert all(math.isfinite(v) and v>=0 for v in norms.values())
        rejected={c['principal_id'] for c in clients if truth(c['rtc_r2_flagged'])} if policy=='raw_eligible' else set()
        groups={c['principal_id']:norms[c['principal_id']] for c in clients
            if c['principal_id'] not in rejected and float(c['rtc_r1_nominal_mass'])>0}
        for c in clients:
            pid=c['principal_id'];peers=[v for p,v in groups.items() if p!=pid]
            median=statistics.median(peers) if len(peers)>=calibration['min_peers'] else None
            valid=pid in groups and median is not None and median>1e-12
            ratio=groups[pid]/median if valid else None
            low=bool(valid and ratio<calibration['ratio_threshold'])
            streak=min(calibration['required_visits'],state.get(pid,0)+1) if low else 0
            state[pid]=streak;flag=streak>=calibration['required_visits']
            assert integer(c['rtc_r3l_reference_count'])==len(peers)
            number(c['rtc_r3l_principal_residual_norm'],groups.get(pid))
            number(c['rtc_r3l_reference_median'],median);number(c['rtc_r3l_ratio'],ratio)
            if policy=='raw_eligible':assert truth(c['rtc_g2_reference_eligible'])==(pid not in rejected)
            assert truth(c['rtc_r3l_valid'])==valid and truth(c['rtc_r3l_low'])==low
            assert integer(c['rtc_r3l_streak'])==streak and truth(c['rtc_r3l_flagged'])==flag
            q=0. if flag and mode=='cap' else 1.
            assert float(c['rtc_r3l_q'])==q
            existing=min(float(c['rtc_r1_existing_q']),float(c['rtc_r1_q']),float(c['rtc_r2_q']))
            assert float(c['rtc_r3l_existing_q'])==existing
            assert truth(c['rtc_r3l_applied'])==(q<existing)
            nominal=float(c['rtc_r1_nominal_mass'])
            assert float(c['rtc_r3l_nominal_mass'])==nominal
            assert 0<=float(c['aggregation_weight'])<=nominal*min(q,existing)+1e-8
