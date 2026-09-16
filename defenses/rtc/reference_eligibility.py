"""H2 reference-principal eligibility from the last original R1c/R2 flags."""
import copy
import json

VERSION = 'rtc_reference_eligibility.v1'


class ReferenceEligibility:
    def __init__(self):
        self._state = {}

    def prepare(self, measured, principals, mode):
        if mode not in ('observe', 'cap') or len(principals) != len(measured['rows']):
            raise ValueError('Invalid reference eligibility inputs')
        ids = list(map(str, principals))
        original = [bool(r['flagged']) for r in measured['rows']]
        records = []
        for i, row in enumerate(measured['rows']):
            refs = sorted({ids[j] for j in row['reference_indices']} - {ids[i]})
            known = [p for p in refs if p in self._state]
            rejected = [p for p in refs if self._state.get(p, False)]
            eligible = len(refs) - len(rejected)
            required = (2 * len(refs) + 2) // 3
            passes = bool(refs) and eligible >= required
            q = float(row['q'])
            veto = mode == 'cap' and original[i] and not passes
            if veto:
                row['flagged'] = False
                row['q'] = 1.
            records.append(dict(rtc_r1e_original_flagged=original[i], rtc_r1e_original_q=q,
                rtc_r1e_reference_principals_json=json.dumps(refs),
                rtc_r1e_rejected_principals_json=json.dumps(rejected),
                rtc_r1e_known_count=len(known), rtc_r1e_eligible_count=eligible,
                rtc_r1e_required=required, rtc_r1e_passes=passes, rtc_r1e_vetoed=veto,
                rtc_r1e_previous_rejected=self._state.get(ids[i], False),
                rtc_r1e_previous_known=ids[i] in self._state))
        return records, original

    def transition(self, principals, original, raw_rows):
        if not (len(principals) == len(original) == len(raw_rows)):
            raise ValueError('Mismatched eligibility transition')
        observed = {}
        for pid, direction, raw in zip(map(str, principals), original, raw_rows):
            observed[pid] = observed.get(pid, False) or bool(direction) or bool(raw['rtc_r2_flagged'])
        return dict(self._state, **observed)

    def commit(self, state):
        if any(type(v) is not bool for v in state.values()):
            raise ValueError('Invalid eligibility state')
        self._state = copy.deepcopy(state)

    def state_dict(self):
        return dict(self._state)
