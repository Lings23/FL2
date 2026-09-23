import copy
from unittest.mock import patch
import pytest
from experiments import rtc_m3_multikrum as stage
from experiments import rtc_m3_all_attacks as parent


def test_manual_matrix_uses_parent_recipe_only_mk(tmp_path):
    for condition in parent.conditions():
        args = stage.command(parent, tmp_path, condition)
        assert args[-1] == '--dry-run'
        assert args[args.index('--defenses')+1] == stage.MK
        assert args[args.index('--batch-size')+1] == '96'
        assert args[args.index('--seeds')+1] == '42'
        assert '--skip-clean' in args
        assert ('--byzantine-evaluation-only' in args) == (condition['attack'] == 'random_noise')


def test_host_drift_rejected_before_training():
    expected = dict(os='Linux', python='3.11.16', devices=['5090','5090'])
    stage.require_training_environment(expected, expected)
    with pytest.raises(ValueError, match='original M3'):
        stage.require_training_environment(dict(expected, python='3.11.17'), expected)


def test_refuse_tampered_parent_before_extraction(tmp_path):
    (tmp_path/'m3_lock.json').write_text('{}')
    with pytest.raises(ValueError, match='parent lock'):
        stage.source_members(tmp_path, dict(parent_lock_sha256='wrong'))


def test_nonempty_output_not_overwritten(tmp_path):
    (tmp_path/'partial').write_text('keep')
    with pytest.raises(ValueError, match='Nonempty'):
        stage.prepare(tmp_path, tmp_path, None)
    assert (tmp_path/'partial').read_text() == 'keep'


def test_config_only_defense_and_paths_may_differ():
    a = dict(client=dict(batch_size=96), project=dict(log_dir='/old'), dataset=dict(data_dir='/old'),
        federation=dict(trial_plan_path='/old', trial_plan_hash='same'), security=dict(defense=dict(enabled=True,
            type='rtc_full', krum_num_malicious=3, krum_num_to_select=1, custom_params={})))
    b = copy.deepcopy(a)
    b['security']['defense'].update(type='krum', krum_num_to_select=5, custom_params=dict(
        spectral_direction_mode='observe', raw_norm_mode='observe', spectral_direction_calibration='a', raw_norm_calibration='b'))
    b['project']['log_dir']='/new'; b['dataset']['data_dir']='/new'; b['federation']['trial_plan_path']='/new'
    stage.check_config(a,b)
    b['client']['batch_size']=48
    with pytest.raises(ValueError, match='configuration differs'):
        stage.check_config(a,b)
    b['client']['batch_size']=96; b['security']['defense']['krum_num_to_select']=1
    with pytest.raises(ValueError, match='select5'):
        stage.check_config(a,b)


def test_asr_and_benign_failures_not_erased_by_acc_gain():
    a = dict(condition='test', active_accuracy=.85, final_accuracy=.85, tail_accuracy=.85,
        asr_mean=.2, asr_peak=.8, asr_tail10=.1, asr_final=.1, benign_union_flag_rate=.03)
    b = dict(a, active_accuracy=.8, asr_mean=.1, asr_peak=.5, benign_union_flag_rate=None)
    r = stage.comparison(a,b,dict(acc_noninferiority_pp=.2,m3_benign_cap_limit=.01))
    assert r['descriptive_engineering_checks']['acc_mean_noninferior']
    assert not r['descriptive_engineering_checks']['asr_peak_no_increase']
    assert not r['descriptive_engineering_checks']['m3_benign_cap_limit']
    assert not r['statistical_superiority_established']


def test_confusion_reconstruction_rejects_misreported_acc():
    import json
    row = dict(server_confusion_matrix_json=json.dumps([[1000 if i==j else 0 for j in range(10)] for i in range(10)]),
        server_accuracy='1', server_model_state_valid=True, server_logits_valid=True, server_loss_valid=True)
    run=dict(cell=dict(attack='none'),rounds=[row])
    stage.validate_metrics(run,bool)
    row['server_accuracy']='.99'
    with pytest.raises(ValueError, match='ACC mismatch'):
        stage.validate_metrics(run,bool)


def test_launch_worker_does_not_use_current_workspace_for_training(tmp_path):
    with patch.object(stage.subprocess,'run') as call:
        stage.launch_worker(tmp_path,'prepare')
    args=call.call_args
    assert args.kwargs['cwd']==tmp_path/'frozen_source'
    assert args.args[0][args.args[0].index('--worker')+1]=='prepare'


def test_protocol_never_promotes_or_tunes_m3():
    p=stage.read(stage.PROTOCOL)
    assert p['training_units']==p['reused_m3_units']==12
    assert p['manual_execution_required'] and p['defense']==stage.MK
    assert 'not retrospective' in p['checks']


def test_own_launcher_not_mistaken_for_another_training():
    import os
    from types import SimpleNamespace
    own = SimpleNamespace(info=dict(pid=os.getppid(), name='python', cmdline=['python',str(stage.SCRIPT),'--execute']))
    stage.ensure_no_other_training([own])
    other = SimpleNamespace(info=dict(pid=-100, name='python', cmdline=['python','main.py']))
    with pytest.raises(ValueError, match='Related training process'):
        stage.ensure_no_other_training([own,other])
