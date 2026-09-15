"""Principal-level persistent lower trainable residual cap; labels are not inputs."""
import hashlib
import json
from pathlib import Path
import numpy as np

VERSION = 'rtc_r3_lower_tail_persistence.v1'


def load_calibration(path):
    c=json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if (set(c) != {'version','ratio_threshold','min_peers','required_visits','cap_multiplier','provenance'}
        or c['version']!=VERSION or type(c['ratio_threshold']) not in (int,float)
        or not np.isfinite(c['ratio_threshold']) or not 0<c['ratio_threshold']<1
        or type(c['min_peers']) is not int or c['min_peers']!=9
        or type(c['required_visits']) is not int or c['required_visits']!=2
        or type(c['cap_multiplier']) not in (int,float) or c['cap_multiplier']!=0):
        raise ValueError('Invalid lower-tail calibration')
    return c


def calibration_hash(c):
    return hashlib.sha256(json.dumps(c,sort_keys=True,separators=(',',':')).encode()).hexdigest()


class LowerTailEvidence:
    def __init__(self, calibration):
        self.calibration=calibration
        self._state={}

    def prepare(self, norms, principals, masses, mode):
        if mode not in ('observe','cap'):
            raise ValueError('Invalid lower-tail mode')
        n=np.asarray(norms,dtype=float);m=np.asarray(masses,dtype=float)
        if (n.ndim!=1 or n.shape!=m.shape or len(principals)!=len(n) or not len(n)
            or not np.isfinite(n).all() or not np.isfinite(m).all() or np.any(n<0) or np.any(m<0)):
            raise ValueError('Invalid lower-tail inputs')
        ids=list(map(str,principals));groups={}
        for pid in sorted(set(ids)):
            indexes=[i for i,p in enumerate(ids) if p==pid];mass=float(sum(m[i] for i in indexes))
            if mass>0:
                groups[pid]=float(sum(m[i]*n[i] for i in indexes)/mass)
        next_state=dict(self._state);by_principal={}
        for pid in sorted(set(ids)):
            peers=[v for p,v in groups.items() if p!=pid]
            median=float(np.median(peers)) if len(peers)>=self.calibration['min_peers'] else None
            valid=pid in groups and median is not None and median>1e-12
            ratio=groups[pid]/median if valid else None
            if ratio is not None and not np.isfinite(ratio):raise ValueError('Nonfinite lower-tail ratio')
            low=bool(valid and ratio<self.calibration['ratio_threshold'])
            streak=min(self.calibration['required_visits'],self._state.get(pid,0)+1) if low else 0
            next_state[pid]=streak
            flagged=streak>=self.calibration['required_visits']
            by_principal[pid]=dict(rtc_r3l_principal_residual_norm=groups.get(pid),
                rtc_r3l_reference_count=len(peers),rtc_r3l_reference_median=median,
                rtc_r3l_valid=valid,rtc_r3l_ratio=ratio,rtc_r3l_low=low,rtc_r3l_streak=streak,
                rtc_r3l_flagged=flagged,rtc_r3l_q=0. if flagged and mode=='cap' else 1.)
        rows=[dict(by_principal[pid],rtc_r3l_residual_norm=float(n[i])) for i,pid in enumerate(ids)]
        return rows,next_state

    def commit(self, state):
        self._state=dict(state)

    def state_dict(self):
        return dict(self._state)
