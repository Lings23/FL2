"""Random noise trainable scope, real fit hooks, and evidence rejection."""
import copy
import json

import numpy as np
import pandas as pd
import pytest
import torch
from flwr.common import FitIns, ndarrays_to_parameters, parameters_to_ndarrays
from torch.utils.data import DataLoader, TensorDataset

from attacks.attack_client import RandomNoiseClient, validate_attack_config
from client.fl_client import FedSecClient
from config.config_loader import AttackConfig, ClientConfig
from experiments.periodic_attack import run_id
from experiments.rtc_v3.random_noise_validation import VERSION, validate_client_evidence, validate_freeze_entry
from models.model_factory import get_parameters, get_trainable_parameter_indices
from utils.numerical_failure import NumericalFailure


def _client(scale=10):
    model = torch.nn.Sequential(torch.nn.BatchNorm1d(2), torch.nn.Linear(2, 2))
    model[1].bias.requires_grad_(False)
    client = object.__new__(RandomNoiseClient)
    client.model = model
    client.attack_cfg = AttackConfig(random_noise_scale=scale)
    client._attack_active = True
    initial = get_parameters(model)
    client.on_before_fit(initial, {})
    local = [x + np.array(0.25 if x.dtype.kind == 'f' else 1, dtype=x.dtype) for x in initial]
    return client, initial, local


@pytest.mark.parametrize('scale', [1, 5, 10])
def test_norm_scope_preserves_bn_and_frozen_parameters(scale):
    client, initial, local = _client(scale)
    indices = get_trainable_parameter_indices(client.model)
    metrics = {}
    attacked = client.on_after_fit(local, metrics)
    ref = np.linalg.norm(np.concatenate([(local[i]-initial[i]).ravel() for i in indices]))
    actual = np.linalg.norm(np.concatenate([(attacked[i]-initial[i]).ravel() for i in indices]))
    assert actual == pytest.approx(scale * ref, rel=1e-6)
    for i in set(range(len(initial))) - set(indices):
        np.testing.assert_array_equal(attacked[i], local[i])
    assert metrics['random_noise_version'] == VERSION
    assert metrics['random_noise_buffers_preserved']
    assert metrics['random_noise_dimension'] == sum(initial[i].size for i in indices)


def test_zero_reference_and_inactive_hook():
    client, initial, _ = _client()
    metrics = {}
    attacked = client.on_after_fit(initial, metrics)
    assert metrics['random_noise_zero_reference']
    assert metrics['random_noise_uploaded_norm'] == 0
    for a, b in zip(attacked, initial):
        np.testing.assert_array_equal(a, b)
    client._attack_active = False
    assert client.on_after_fit(initial, {}) is initial


@pytest.mark.parametrize('scale', [float('nan'), float('inf'), -1])
def test_invalid_scale_is_configuration_error(scale):
    with pytest.raises(ValueError, match='random_noise_scale'):
        validate_attack_config(AttackConfig(type='random_noise', enabled=True, random_noise_scale=scale), num_classes=2)


def test_layout_errors_and_dtype_overflow():
    with pytest.raises(ValueError, match='rademacher'):
        validate_attack_config(AttackConfig(type='random_noise', random_noise_distribution='gaussian'), num_classes=2)
    client, initial, local = _client()
    with pytest.raises(ValueError, match='count'):
        client.on_after_fit(local[:-1], {})
    client.model = None
    with pytest.raises(ValueError, match='layout'):
        client.on_after_fit(local, {})
    client, initial, local = _client(1e40)
    with pytest.raises(NumericalFailure, match='dtype conversion overflow'):
        client.on_after_fit(local, {})


@pytest.mark.parametrize('server_round', [10, 11])
def test_real_client_fit_window_rng_and_buffers(server_round):
    torch.manual_seed(3)
    model = torch.nn.Sequential(torch.nn.BatchNorm1d(2), torch.nn.Linear(2, 2))
    initial = get_parameters(model)
    loader = DataLoader(TensorDataset(torch.randn(8, 2), torch.tensor([0, 1]*4)), batch_size=4)
    cfg = ClientConfig(local_epochs=1, learning_rate=0.01)
    attack = AttackConfig(enabled=True, type='random_noise', random_noise_scale=10, attack_start_round=11)
    common = dict(client_id=1, train_loader=loader, val_loader=loader, client_cfg=cfg,
                  device=torch.device('cpu'), num_classes=2)
    clean = FedSecClient(model=copy.deepcopy(model), **common)
    adversary = RandomNoiseClient(model=copy.deepcopy(model), attack_cfg=attack, **common)
    adversary.is_malicious = True
    ins = FitIns(ndarrays_to_parameters(initial), {'server_round':server_round, 'fit_seed':123, 'attack_seed':456})
    local = parameters_to_ndarrays(clean.fit(ins).parameters)
    result = adversary.fit(ins)
    output = parameters_to_ndarrays(result.parameters)
    again = parameters_to_ndarrays(adversary.fit(ins).parameters)
    indices = get_trainable_parameter_indices(model)
    for i, value in enumerate(output):
        np.testing.assert_array_equal(value, again[i])
        if i not in indices or server_round == 10:
            np.testing.assert_array_equal(value, local[i])
    assert result.metrics['attack_active'] == (server_round == 11)
    if server_round == 11:
        ref = np.linalg.norm(np.concatenate([(local[i]-initial[i]).ravel() for i in indices]))
        assert result.metrics['random_noise_uploaded_norm'] == pytest.approx(ref * 10, rel=1e-5)


def _evidence(tmp_path):
    client, _, local = _client()
    metrics = {}
    client.on_after_fit(local, metrics)
    spec = {'attack':'random_noise', 'attack_version':VERSION, 'random_noise_scale':10,
            'defense':'fedavg', 'seed':42, 'malicious_fraction':0.3, 'period':'continuous', 'boost_factor':10}
    records = pd.DataFrame([dict(metrics, round=11, cid=1, is_malicious=True, attack_active=True)])
    path = tmp_path / f'{run_id(spec)}_clients.csv'
    records.to_csv(path, index=False)
    rounds = pd.DataFrame([{'round':11, 'fit_completed_partition_ids_json':'[1]',
                           'fit_selected_active_attackers':1, 'planned_attack_active':1}])
    return spec, records, path, rounds


@pytest.mark.parametrize('tamper', ['none', 'zero', 'dimension', 'buffer', 'missing'])
def test_execution_gate_checks_actual_noise(tmp_path, tamper):
    spec, records, path, rounds = _evidence(tmp_path)
    if tamper == 'zero': records['random_noise_uploaded_norm'] = 0
    if tamper == 'dimension': records['random_noise_dimension'] *= 2
    if tamper == 'buffer': records['random_noise_buffers_preserved'] = False
    if tamper == 'missing': records = records.drop(columns=['random_noise_reference_norm'])
    records.to_csv(path, index=False)
    if tamper == 'none':
        validate_client_evidence(spec, rounds, tmp_path)
    else:
        with pytest.raises(ValueError): validate_client_evidence(spec, rounds, tmp_path)


def test_pending_freeze_cannot_authorize_formal_run(tmp_path):
    with pytest.raises(ValueError, match='verified FedAvg'):
        validate_freeze_entry({'parameters':{'random_noise_scale':10}}, tmp_path)


def _freeze_evidence(tmp_path):
    from attacks.spec import attack_source_hash, attack_contract_payload
    from experiments.rtc_v3.byzantine import _runtime_attack_config
    from experiments.rtc_v3.random_noise_validation import _artifact, SCOPE
    parameters = {'random_noise_scale':10, 'random_noise_distribution':'rademacher'}
    runtime = _runtime_attack_config('random_noise', parameters, malicious_fraction=0.3, attack_start_round=11, attack_end_round=-1)
    implementation = attack_contract_payload('random_noise', vars(runtime))['implementation_hash']
    entry = {'parameters':parameters, 'selection_metric':'accuracy_drop', 'selection_threshold':0.03,
             'selection_value':0.7, 'validation':{'schema_version':1, 'source_hash':attack_source_hash(),
             'attack_version':VERSION, 'scope':SCOPE, 'runs':[]}}
    for seed in (42, 46):
        root = tmp_path/str(seed)
        root.mkdir()
        spec, records, clients_path, _ = _evidence(root)
        spec.update(seed=seed, trial_plan_hash=f'plan{seed}', attack_implementation_hash=implementation,
                    participation_rate=0.5, partition='iid')
        rows = pd.concat([records.assign(round=r, attack_active=(r>=11)) for r in range(1,61)])
        rows.to_csv(clients_path, index=False)
        manifest = {'trial_plan_hash':f'plan{seed}', 'initial_model_sha256':'model', 'data_manifest_sha256':'data'}
        pairing = root/'pairing.json'
        pairing.write_text(json.dumps(manifest))
        frame = pd.DataFrame({'round':range(61), 'server_loss':1e20, 'server_accuracy':0.1,
                              'attack':'random_noise', 'defense':'fedavg', 'seed':seed,
                              'server_model_state_valid':True, 'server_logits_valid':True, 'server_loss_valid':True,
                              'attack_version':VERSION, 'random_noise_scale':10, 'attack_implementation_hash':implementation,
                              'planned_attack_active':[int(r>=11) for r in range(61)],
                              'fit_selected_active_attackers':[int(r>=11) for r in range(61)],
                              'fit_completed_partition_ids_json':'[1]', 'fit_selected_partition_ids':'1',
                              'fit_fit_seed_digest':'shared'})
        attack_path = root/'attack.csv'
        frame.to_csv(attack_path, index=False)
        clean_path = root/'clean.csv'
        frame.assign(attack='none', server_accuracy=0.8, planned_attack_active=0).to_csv(clean_path, index=False)
        entry['validation']['runs'].append({'seed':seed, 'accuracy_drop':0.7, 'attack_spec':spec,
            'attack_clients':_artifact(clients_path), 'attack_rounds':_artifact(attack_path),
            'clean_rounds':_artifact(clean_path), 'attack_pairing':_artifact(pairing), 'clean_pairing':_artifact(pairing)})
    return entry


@pytest.mark.parametrize('tamper', ['none', 'one_seed', 'weak_effect', 'artifact', 'unpaired', 'version'])
def test_freeze_recomputes_paired_effect_and_checks_provenance(tmp_path, tamper):
    from pathlib import Path
    from experiments.rtc_v3.random_noise_validation import _artifact
    entry = _freeze_evidence(tmp_path)
    if tamper == 'one_seed': entry['validation']['runs'].pop()
    if tamper == 'weak_effect': entry['validation']['runs'][0]['accuracy_drop'] = 0.01
    if tamper == 'version': entry['validation']['attack_version'] = 'v1'
    if tamper == 'artifact':
        Path(entry['validation']['runs'][0]['attack_rounds']['path']).write_text('tampered')
    if tamper == 'unpaired':
        path = tmp_path/'wrong_pair.json'
        path.write_text(json.dumps({'trial_plan_hash':'wrong'}))
        entry['validation']['runs'][0]['clean_pairing'] = _artifact(path)
    if tamper == 'none': validate_freeze_entry(entry, tmp_path)
    else:
        with pytest.raises(ValueError): validate_freeze_entry(entry, tmp_path)


def test_screening_plan_has_two_seeds_and_shared_clean(tmp_path):
    from experiments.run import parse_args, PROFILES
    args = parse_args(['--profile','rtc-byzantine-screen', '--attacks','random_noise', '--seeds','42,46',
                       '--malicious-fractions','0.3', '--num-clients','20', '--rounds','60',
                       '--output',str(tmp_path), '--dry-run'])
    PROFILES[args.profile].runner(args)
    specs = json.loads((tmp_path/'experiment_manifest.json').read_text())['specs']
    assert len(specs) == 8
    for seed in (42,46):
        subset = [s for s in specs if s['seed'] == seed]
        assert len({s['trial_plan_hash'] for s in subset}) == 1
        assert len([s for s in subset if s['attack']=='none']) == 1
        attacked = [s for s in subset if s['attack']=='random_noise']
        assert {s['random_noise_scale'] for s in attacked} == {1,5,10}
        assert all(s['attack_start_round']==11 and s['attack_version']==VERSION for s in attacked)
