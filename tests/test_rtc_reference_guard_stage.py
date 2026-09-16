"""Synthetic pre-registration and manual-execution checks only."""
import copy
import json
import pytest
import numpy as np
import types
from experiments import rtc_reference_guard_stage as stage
from tests.test_rtc_i12_bridge import combined_artifacts
from tests.test_rtc_r2_stage import write_csv


def test_matrix_defaults_and_failed_run_guard(tmp_path,monkeypatch):
    assert sum(len(ds) for _,_,_,ds in stage.batches())==24
    assert {s for _,_,s,_ in stage.batches()}=={201,203}
    for n,a,s,ds in stage.batches():
        cmd=stage.command(tmp_path,n,a,s,ds)
        assert cmd[-1]=='--dry-run' and '--smoke' not in cmd
    monkeypatch.setattr(stage,'verify_lock',lambda root:{'cells':[dict(batch='b',run_id='r')]})
    monkeypatch.setattr(stage.prior,'ensure_no_training_process',lambda:None)
    monkeypatch.setattr(stage.subprocess,'run',lambda *a,**kw:pytest.fail('No training permitted'))
    p=tmp_path/'b/status/r.json';p.parent.mkdir(parents=True)
    p.write_text(json.dumps(dict(state='failed',exit_code=1,last_round=2)))
    with pytest.raises(ValueError):stage.execute(tmp_path)


def test_repair_requires_safety_reduction_and_preserves_utility_and_weight_gates(monkeypatch):
    runs=[]
    for name,attack,seed,ds in stage.batches():
        for d in ds:
            candidate=d==stage.OBSERVER
            m=dict(attack=attack,seed=seed,defense=d,active_accuracy=.83,final_accuracy=.88,
                benign_union_flags=1 if candidate else 4,benign_union_flag_rate=.003 if candidate else .012,
                malicious_weight_per_round=.04)
            runs.append(dict(cell=dict(batch=name,attack=attack,seed=seed,defense=d),computed=m,rounds=[{}],clients=[]))
    monkeypatch.setattr(stage,'metrics',lambda r:r['computed'])
    monkeypatch.setattr(stage.prior,'paired',lambda *a:None)
    p=stage.prior.read_json(stage.PROTOCOL)
    assert stage.decide(runs,p)['candidate_accepted']
    for key,value in [('active_accuracy',.82),('final_accuracy',.87),('benign_union_flag_rate',.02),('malicious_weight_per_round',.05),('benign_union_flags',4)]:
        bad=copy.deepcopy(runs)
        row=next(r for r in bad if r['cell']['attack']=='sign_flip' and r['cell']['seed']==201 and r['cell']['defense']==stage.OBSERVER)
        row['computed'][key]=value
        assert not stage.decide(bad,p)['candidate_accepted']
    with pytest.raises(AssertionError):stage.decide(runs[:-1],p)


def test_full_validator_accepts_history_evidence_and_rejects_forged_guard(combined_artifacts):
    from defenses.rtc.reference_history import ReferenceHistory
    from defenses.rtc.spectral_direction import measure
    from defenses.rtc.reference_guard import apply,VERSION
    root,c,lock=combined_artifacts;c['defense']=stage.OBSERVER
    batch=root/c['batch'];rid=c['run_id'];v=stage.validator
    rr=v.read_csv(batch/'rounds'/(rid+'.csv'));cc=v.read_csv(batch/'raw'/(rid+'_clients.csv'))
    h=ReferenceHistory()
    for r in rr[1:]:
        rd=int(r['round']);active=rd>=11
        x=np.column_stack([np.array([1]*7+([-1,1,-1] if active else [1]*3)),np.eye(10)*.2])
        clipped=[[a] for a in x];m=measure(clipped,[0],lock['calibration'],'cap')
        clients=[a for a in cc if a['round']==r['round']]
        guard=apply(m,clipped,h.previous,'cap')
        weights=np.array([float(a['aggregation_weight']) for a in clients])
        actual=[weights@x+float(r['fit_rtc_v3_anchor_recycle_mass'])*np.ones(x.shape[1])*.1]
        history,summary,next_state=h.prepare(clipped,actual,m);h.commit(next_state)
        r.update({'fit_'+k:val for k,val in summary.items()})
        r.update(fit_rtc_r1g_version=VERSION,fit_rtc_r1g_mode='cap',fit_rtc_r1g_threshold=0.)
        for row,extra,hrow in zip(clients,guard,history):row.update(extra);row.update(hrow)
    rp=batch/'rounds'/(rid+'.csv');cp=batch/'raw'/(rid+'_clients.csv')
    write_csv(rp,rr);write_csv(cp,cc)
    scope=dict(v.verify_cell.__globals__,MODES=stage.MODES,RTC=(stage.PARENT,stage.OBSERVER))
    verify=types.FunctionType(v.verify_cell.__code__,scope)
    run=verify(root,c,lock);assert len(stage.verify_history(run))==600
    for field,value in [('rtc_r1g_vetoed',True),('rtc_r1g_original_q',1.),('rtc_r1g_reference_cosine',-.5),('rtc_r1g_passes',False)]:
        bad=copy.deepcopy(cc)
        next(a for a in bad if a['round']=='11' and a['cid']=='7')[field]=value
        write_csv(cp,bad)
        with pytest.raises(AssertionError):verify(root,c,lock)
