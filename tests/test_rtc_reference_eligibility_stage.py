"""Synthetic pre-registration and manual-execution checks only."""
import copy
import json
import pytest
import numpy as np
import types
from experiments import rtc_reference_eligibility_stage as stage
from tests.test_rtc_i12_bridge import combined_artifacts
from tests.test_rtc_r2_stage import write_csv


def test_host_client_defaults_cannot_change_registered_training(tmp_path):
    import yaml
    from dataclasses import asdict
    from config.config_loader import load_config
    base = dict(dataset=dict(data_dir='old'), ray=dict(client_num_gpus=.5),
        client=dict(local_epochs=2, batch_size=128, optimizer='adam', learning_rate=.1,
            momentum=0., weight_decay='5e-3', lr_scheduler='none'))
    saved = copy.deepcopy(base)
    cfg = stage.runtime_config(base, '/server/data')
    path = tmp_path / 'runtime.yaml'
    path.write_text(yaml.safe_dump(cfg))
    stage.verify_client_contract(asdict(load_config(path).client))
    assert base == saved and cfg['ray'] == base['ray']
    assert cfg['dataset']['data_dir'] == '/server/data'
    assert cfg['client'] == stage.CLIENT_CONTRACT


def test_contract_error_names_actual_and_expected_fields():
    wrong = dict(stage.CLIENT_CONTRACT, local_epochs=2, weight_decay='0.0001')
    with pytest.raises(ValueError) as error:
        stage.verify_client_contract(wrong)
    text = str(error.value)
    assert 'local_epochs' in text and 'weight_decay' in text and 'actual_type' in text
    assert 'expected' in text and 'actual' in text


def test_interrupted_preparation_preserved(tmp_path, monkeypatch):
    evidence = tmp_path / 'runtime_config.yaml'
    evidence.write_text('old preparation')
    monkeypatch.setattr(stage.subprocess, 'run', lambda *a, **kw: pytest.fail('Must not start subprocess'))
    with pytest.raises(ValueError, match='Nonempty unfrozen'):
        stage.prepare(tmp_path)
    assert evidence.read_text() == 'old preparation'


def test_matrix_defaults_and_failed_run_guard(tmp_path,monkeypatch):
    assert sum(len(ds) for _,_,_,ds in stage.batches())==24
    assert {s for _,_,s,_ in stage.batches()}=={201,204}
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


def test_full_validator_replays_principal_state_and_rejects_forgery(combined_artifacts):
    from defenses.rtc.reference_eligibility import ReferenceEligibility, VERSION
    from defenses.rtc.spectral_direction import measure
    root,c,lock=combined_artifacts;c['defense']=stage.OBSERVER
    batch=root/c['batch'];rid=c['run_id'];v=stage.validator
    rr=v.read_csv(batch/'rounds'/(rid+'.csv'));cc=v.read_csv(batch/'raw'/(rid+'_clients.csv'))
    for row in cc:row['principal_id']=row['cid']
    h=ReferenceEligibility()
    for r in rr[1:]:
        active=int(r['round'])>=11
        x=np.column_stack([np.array([1]*7+([-1,1,-1] if active else [1]*3)),np.eye(10)*.2])
        m=measure([[a] for a in x],[0],lock['calibration'],'cap')
        clients=[a for a in cc if a['round']==r['round']]
        ids=[a['cid'] for a in clients]
        extra,original=h.prepare(m,ids,'cap')
        h.commit(h.transition(ids,original,[dict(rtc_r2_flagged=v.truth(a['rtc_r2_flagged'])) for a in clients]))
        r.update(fit_rtc_r1e_version=VERSION,fit_rtc_r1e_mode='cap',fit_rtc_r1e_quorum_numerator=2,fit_rtc_r1e_quorum_denominator=3)
        for row,values in zip(clients,extra):row.update(values)
    rp=batch/'rounds'/(rid+'.csv');cp=batch/'raw'/(rid+'_clients.csv')
    write_csv(rp,rr);write_csv(cp,cc)
    scope=dict(v.verify_cell.__globals__,MODES=stage.MODES,RTC=(stage.PARENT,stage.OBSERVER))
    verify=types.FunctionType(v.verify_cell.__code__,scope)
    assert len(verify(root,c,lock)['clients'])==600
    for field,value in [('rtc_r1e_vetoed',True),('rtc_r1e_original_q',1.),('rtc_r1e_eligible_count',0),('rtc_r1e_previous_known',False),('rtc_r1e_rejected_principals_json','["0"]')]:
        bad=copy.deepcopy(cc)
        next(a for a in bad if a['round']=='12' and a['cid']=='7')[field]=value
        write_csv(cp,bad)
        with pytest.raises(AssertionError):verify(root,c,lock)
