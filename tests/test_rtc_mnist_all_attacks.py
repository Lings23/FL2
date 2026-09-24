"""Configuration, synthetic metric and restart checks; never train or download data."""
import copy
import json
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import torch
import pytest
from experiments import rtc_mnist_all_attacks as stage
from experiments import periodic_attack as runner
from experiments.trial_plan import TrialPlanV1, _condition_payload, generate_trial_plan


@pytest.fixture
def prepared(tmp_path):
    root = tmp_path / 'mnist'
    with patch.object(runner, '_run_specs', side_effect=AssertionError('Training forbidden')):
        lock = stage.prepare(root, 'calibration', data_dir=tmp_path/'data')
    assert len(lock['cells']) == 2
    assert not list(root.rglob('*_clients.csv'))
    return root


def test_all_108_resolved_configs_and_attack_contracts(prepared):
    # A model-bound bootstrap is sufficient to test spec construction, never to
    # authorize evaluation. No fake acceptance receipt is created.
    stage.write(prepared/'calibrated/manifest.json', stage.read(prepared/'bootstrap.json'))
    protocol = stage.read(stage.PROTOCOL)
    cells = []
    for condition in protocol['conditions']:
        specs, args = stage.make_specs(prepared, 'evaluation', condition, 42, protocol['defenses'])
        assert len(specs) == 9
        cells += stage.save_batch(prepared, 'evaluation', condition['name'], specs, args)
        for spec in specs:
            cfg = asdict(runner._build_spec_config(spec, args, prepared/'evaluation'/condition['name']))
            assert cfg['client']['batch_size'] == 96
            assert cfg['ray']['client_num_gpus'] == .125
            assert cfg['security']['defense']['root_dataset_size'] == 100
            assert cfg['security']['defense']['reserve_root_for_all']
            if spec['defense'] == 'multi_krum':
                assert cfg['security']['defense']['krum_num_to_select'] == 5
                assert cfg['security']['defense']['krum_num_malicious'] == 3
            if spec['defense'] == stage.M3:
                custom = cfg['security']['defense']['custom_params']
                assert custom['reference_eligibility_memory'] == 'confirmed'
                assert custom['reference_eligibility_mode'] == custom['spectral_direction_mode'] == custom['raw_norm_mode'] == 'cap'
                assert 'lower_tail_mode' not in custom
            if spec['attack'] in ('dba','scaling_backdoor'):
                assert cfg['security']['attack']['trigger_value'] == pytest.approx((1-.1307)/.3081)
        assert len({s['trial_plan_hash'] for s in specs}) == 1
    assert len(cells) == 108


def test_bootstrap_model_contract_and_calibration_observers(prepared):
    from models.model_factory import get_model, get_parameters, get_parameter_roles
    from defenses.rtc.calibration import model_metadata_hash, CalibrationManifest
    model = get_model(architecture='mnist_cnn', num_classes=10, pretrained=False, dataset_name='mnist')
    manifest = CalibrationManifest.load(prepared/'bootstrap.json')
    assert manifest.payload['model_metadata_hash'] == model_metadata_hash(get_parameters(model), get_parameter_roles(model))
    assert not manifest.payload['metadata']['formal_evaluation_ready']
    for cell in stage.read(prepared/'calibration/lock.json')['cells']:
        custom = cell['spec']['custom_params']
        assert custom['raw_norm_mode'] == custom['spectral_direction_mode'] == 'observe'
        assert custom['export_sketches']
        assert cell['spec']['malicious_fraction'] == 0
        plan = TrialPlanV1.load(cell['spec']['trial_plan_path']).payload
        assert plan['data_config']['dataset'] == 'mnist'


def test_no_implicit_evaluation_without_real_calibration(prepared):
    with pytest.raises(FileNotFoundError):
        stage.prepare(prepared, 'validation')
    with pytest.raises(FileNotFoundError):
        stage.execute(prepared, 'evaluation')


def test_source_drift_blocks_execution(prepared):
    with patch.object(stage, 'sources', return_value={}):
        with pytest.raises(ValueError, match='Source drift'):
            stage.verify_lock(prepared, 'calibration', training=True)


def test_explicit_dataset_plan_and_legacy_defaults(prepared):
    cell = stage.read(prepared/'calibration/lock.json')['cells'][0]
    spec = cell['spec']; args = stage.runner_args(prepared, prepared/'calibration')
    args.sampling_protocol = 'principal_uniform'
    legacy = {k:v for k,v in spec.items() if k not in ('dataset','architecture')}
    old_condition = _condition_payload(legacy, args)
    assert 'dataset' not in old_condition and 'architecture' not in old_condition
    principal_map = {str(i):str(i) for i in range(20)}
    old = generate_trial_plan(old_condition, principal_map)
    new = generate_trial_plan(_condition_payload(spec, args), principal_map)
    assert old['data_config']['dataset'] == 'cifar10'
    assert old['initial_model_config']['architecture'] == 'resnet18'
    assert new['data_config']['dataset'] == 'mnist'
    assert old['trial_plan_hash'] != new['trial_plan_hash']


@pytest.mark.parametrize('attack', ['label_flip_targeted','label_flip_all_reverse','scaling_backdoor','dba'])
def test_mnist_asr_uses_unequal_class_counts(attack):
    counts = [980,1135,1032,1010,982,892,958,1028,974,1009]
    matrix = [[0]*10 for _ in counts]
    for i,n in enumerate(counts):
        matrix[i][i] = n
    matrix[5][5] -= 89; matrix[5][3] = 89
    if attack == 'label_flip_targeted':
        total, success = 892, 89
    elif attack == 'label_flip_all_reverse':
        total, success = 10000, 0
    else:
        total, success = 9020, 90
    rr = [dict(round=f'{i}.0', server_confusion_matrix_json=json.dumps(matrix), server_accuracy=str(9911/10000),
        server_asr=str(success/total), server_asr_total=str(total), server_asr_valid='True',
        server_model_state_valid='True', server_logits_valid='True', server_loss_valid='True') for i in range(61)]
    assert stage.validate_metric_rows(rr, attack) == counts
    rr[-1]['server_asr_total'] = '1000' if attack.startswith('label') else '9000'
    with pytest.raises(ValueError, match='ASR denominator'):
        stage.validate_metric_rows(rr, attack)


@pytest.mark.parametrize('bad', ['2.5','NaN','Infinity'])
def test_nonintegral_metadata_rejected(bad):
    with pytest.raises(ValueError):
        stage.integer(bad)
    assert stage.integer('2.0') == 2


def test_mnist_dba_geometry_and_white_value():
    from attacks.attack_client import get_dba_trigger_coords, stamp_dba_coords_
    x = torch.zeros(1,28,28)
    coords = [xy for i in range(4) for xy in get_dba_trigger_coords(i, x.shape)]
    assert len(coords) == len(set(coords)) == 24
    stamp_dba_coords_(x, coords, trigger_value=(1-.1307)/.3081, value_mode='scalar')
    assert x.count_nonzero().item() == 24
    assert x[0,0,0].item() == pytest.approx((1-.1307)/.3081)


def test_interrupted_cell_requires_explicit_restart_and_preserves_evidence(prepared):
    cell = stage.read(prepared/'calibration/lock.json')['cells'][0]
    batch, status, _ = stage.cell_paths(prepared, 'calibration', cell)
    stage.write(status, dict(state='failed', exit_code=1, last_round=17))
    original = status.read_bytes()
    with patch('experiments.rtc_m3_multikrum.ensure_no_other_training'), patch.object(runner,'_run_specs') as training:
        with pytest.raises(ValueError, match='--restart-incomplete'):
            stage.execute(prepared, 'calibration')
        training.assert_not_called()
        assert status.read_bytes() == original
        training.side_effect = RuntimeError('Mocked training boundary')
        with pytest.raises(RuntimeError, match='Mocked training'):
            stage.execute(prepared, 'calibration', restart=True)
        assert not status.exists()
        assert next((batch/'interrupted').rglob(status.name)).read_bytes() == original


def test_clean_filter_calibration_handles_positive_pair_cosines(tmp_path):
    import numpy as np
    from defenses.rtc.spectral_direction import load_calibration
    from defenses.rtc.raw_norm import load_calibration as load_raw
    gram = .5*np.eye(10) + .5*np.ones((10,10))
    synthetic = dict(rounds=[dict(round=str(i), fit_rtc_r1_gram_json=json.dumps(gram.tolist())) for i in range(61)],
        clients=[dict(round=str(i), cid=str(j), rtc_r2_raw_norm='1') for i in range(1,61) for j in range(10)])
    direction, norm = stage.calibrate_filters([synthetic, copy.deepcopy(synthetic)])
    assert direction['cosine_threshold'] == 0
    assert direction['corroboration']['pairwise_threshold'] == -1e-12
    assert norm['ratio_threshold'] == 3
    stage.write(tmp_path/'direction.json', direction); stage.write(tmp_path/'raw.json', norm)
    load_calibration(tmp_path/'direction.json'); load_raw(tmp_path/'raw.json')


def test_cannot_disable_existing_promoted_parent_floor(prepared):
    from experiments import rtc_v3_formal_calibration as calibration
    args = calibration.parse_args(['--config',str(prepared/'runtime_config.yaml'),
        '--bootstrap-manifest',str(stage.ROOT/'config/rtc_v3_manifest_formal_iid_semantic.json')])
    with patch.object(calibration,'_load_observations', side_effect=AssertionError('Must reject before observations')):
        with pytest.raises(ValueError, match='new-dataset'):
            calibration.calibrate_manifest(args=args, specs=[], inherit_parent_budget_floor=False)
