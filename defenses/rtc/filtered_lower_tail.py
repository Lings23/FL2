"""G2 reference repair. No identity labels or attack types are inputs.

This scorer is not enabled by the clean calibration experiment. It replays
the observed residuals; a later, separately registered trial must enable caps.
"""
import numpy as np
from defenses.rtc.lower_tail import LowerTailEvidence


class FilteredLowerTailEvidence(LowerTailEvidence):
    def prepare(self, norms, principals, masses, mode, raw_rejected):
        if len(raw_rejected) != len(principals) or any(type(x) is not bool for x in raw_rejected):
            raise ValueError('One boolean raw rejection per client is required')
        # Validate unmasked inputs, including negative masses that masking
        # must not accidentally hide. Parent prepare never commits state.
        super().prepare(norms, principals, masses, 'observe')
        ids = list(map(str, principals))
        rejected = {p for p, flag in zip(ids, raw_rejected) if flag}
        eligible_mass = np.asarray(masses, dtype=float).copy()
        for i, p in enumerate(ids):
            if p in rejected:
                eligible_mass[i] = 0.
        rows, transition = super().prepare(norms, principals, eligible_mass, mode)
        for p, row in zip(ids, rows):
            row['rtc_g2_reference_eligible'] = p not in rejected
        return rows, transition
