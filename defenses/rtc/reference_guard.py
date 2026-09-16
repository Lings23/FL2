"""H1: require historical half-space corroboration before applying an R1c cap."""
import numpy as np

VERSION='rtc_reference_history_guard.v1'


def apply(measured,clipped,previous,mode):
    if mode not in ('observe','cap'):raise ValueError('Invalid reference guard mode')
    g=np.asarray(measured['gram'],dtype=float);rows=measured['rows'];positions=measured['positions']
    norms=np.sqrt(np.maximum(np.diag(g),0));dots=np.zeros(len(rows))
    pn=None
    if previous is not None:
        if len(previous)!=len(positions) or any(not np.isfinite(a).all() for a in previous):
            raise ValueError('Invalid historical vector')
        pn=float(np.sqrt(sum(float(np.dot(a.ravel(),a.ravel())) for a in previous)))
        for p,a in zip(positions,previous):
            for i,delta in enumerate(clipped):
                if np.asarray(delta[p]).shape!=a.shape:raise ValueError('History layout mismatch')
                dots[i]+=float(np.dot(np.asarray(delta[p],dtype=float).ravel(),a.ravel()))
    diagnostics=[]
    for i,row in enumerate(rows):
        beta=np.zeros(len(rows))
        if row['valid']:
            ids=row['reference_indices'];beta[ids]=np.asarray(row['reference_weights'])/norms[ids]
        rn=float(np.sqrt(max(0,beta@g@beta)))
        valid=bool(pn is not None and pn>1e-12 and row['valid'] and rn>1e-12)
        cosine=float(np.clip(beta@dots/rn/pn,-1,1)) if valid else None
        passes=bool(valid and cosine>0.)
        original=bool(row['flagged']);q=float(row['q'])
        if mode=='cap' and not passes:
            row['flagged']=False;row['q']=1.
        diagnostics.append(dict(rtc_r1g_original_flagged=original,rtc_r1g_original_q=q,
            rtc_r1g_valid=valid,rtc_r1g_reference_cosine=cosine,rtc_r1g_passes=passes,
            rtc_r1g_vetoed=bool(mode=='cap' and original and not passes)))
    return diagnostics
