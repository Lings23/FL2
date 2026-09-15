"""Read-only full-trainable spectral reference versus previous aggregate evidence.

No thresholds, labels, RNG, caps, or optimizer decisions. The previous vector
is the actual model displacement (including accepted-anchor contribution).
"""
import hashlib
import json
import numpy as np

VERSION = 'rtc_reference_history_observe.v1'


def fingerprint(arrays):
    h=hashlib.sha256()
    for a in arrays:
        x=np.ascontiguousarray(a,dtype='<f8')
        h.update(json.dumps(list(x.shape),separators=(',',':')).encode()+b'\n')
        h.update(x.tobytes())
    return h.hexdigest()


class ReferenceHistory:
    def __init__(self):
        self.previous=None

    def prepare(self, clipped, actual, measured):
        positions=measured['positions'];rows=measured['rows'];g=np.asarray(measured['gram'],dtype=float)
        current=[np.asarray(actual[p],dtype=np.float64).copy() for p in positions]
        if not all(np.isfinite(a).all() for a in current):raise ValueError('Nonfinite current aggregate')
        norm=np.sqrt(np.maximum(np.diag(g),0));dots=np.zeros(len(rows))
        previous_norm=None
        if self.previous is not None:
            if len(self.previous)!=len(current) or any(a.shape!=b.shape for a,b in zip(self.previous,current)):
                raise ValueError('History layout changed')
            previous_norm=float(np.sqrt(sum(float(np.dot(a.ravel(),a.ravel())) for a in self.previous)))
            for p,old in zip(positions,self.previous):
                for i,delta in enumerate(clipped):
                    dots[i]+=float(np.dot(np.asarray(delta[p],dtype=float).ravel(),old.ravel()))
        records=[]
        for i,row in enumerate(rows):
            beta=np.zeros(len(rows))
            if row['valid']:
                indexes=row['reference_indices']
                beta[indexes]=np.asarray(row['reference_weights'])/norm[indexes]
            reference_norm=float(np.sqrt(max(0,float(beta@g@beta))))
            valid=bool(self.previous is not None and previous_norm>1e-12 and row['valid'] and reference_norm>1e-12)
            cosine=float(np.clip(beta@dots/reference_norm/previous_norm,-1,1)) if valid else None
            records.append(dict(rtc_r1h_previous_dot=float(dots[i]) if self.previous is not None else None,
                rtc_r1h_valid=valid,rtc_r1h_reference_cosine=cosine))
        summary=dict(rtc_r1h_version=VERSION,rtc_r1h_observe_only=True,
            rtc_r1h_previous_norm=previous_norm,
            rtc_r1h_previous_sha256=fingerprint(self.previous) if self.previous is not None else '',
            rtc_r1h_current_sha256=fingerprint(current))
        return records,summary,current

    def commit(self,current):
        self.previous=[a.copy() for a in current]
