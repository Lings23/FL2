"""Large finite loss is measurable; structured nonfinite failure is terminal."""
import json
import math
import pickle

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from experiments import periodic_attack as runner
from models.model_factory import get_parameters
from server.fl_server import ServerEvaluator
from strategies.fed_strategy import _weighted_avg_metrics
from tests.test_experiment_resilience import _args, _spec, _round_frame
from utils.numerical_failure import NumericalFailure, numerical_evidence


def test_structured_failure_survives_pickle_and_flower_wrapper():
    error = NumericalFailure('nonfinite_update', 'aggregation', 'test', 11)
    assert numerical_evidence(pickle.loads(pickle.dumps(error))) == error.evidence
    assert numerical_evidence(RuntimeError('Flower failure: ' + str(error))) == error.evidence
    assert numerical_evidence(RuntimeError('loss too large')) is None


@pytest.mark.parametrize('loss', [1e20, float('nan'), float('inf')])
def test_cache_loss_finiteness_without_upper_cutoff(tmp_path, loss):
    frame = _round_frame(2)
    frame['server_loss'] = loss
    frame['server_accuracy'] = 0.1
    path = tmp_path / 'rounds.csv'
    frame.to_csv(path, index=False)
    assert runner.validate_round_cache(path, 2)[0] == math.isfinite(loss)
    if math.isfinite(loss):
        summary = runner.summarize_run(frame, _spec())
        assert not summary['collapsed_invalid']
        assert summary['utility_collapse']


def test_client_metric_has_no_finite_loss_ceiling():
    assert _weighted_avg_metrics([(1, {'train_loss':1e20})])['train_loss'] == 1e20
    with pytest.raises(NumericalFailure):
        _weighted_avg_metrics([(1, {'train_loss':float('inf')})])


@pytest.mark.parametrize('amplitude', [1e20, 2e38])
def test_evaluator_float64_retry_preserves_large_finite_loss(tmp_path, amplitude):
    model = torch.nn.Linear(1, 2, bias=False)
    with torch.no_grad(): model.weight.copy_(torch.tensor([[amplitude], [-amplitude]]))
    loader = DataLoader(TensorDataset(torch.ones(4, 1), torch.ones(4, dtype=torch.long)), batch_size=4)
    evaluator = ServerEvaluator(model, loader, torch.device('cpu'), tmp_path, num_classes=2, save_best_model=False)
    loss, metrics = evaluator(11, get_parameters(model), {})
    assert math.isfinite(loss)
    assert loss == pytest.approx(amplitude*2, rel=1e-6)
    assert metrics['loss_valid'] and metrics['logits_valid']
    assert metrics['accuracy'] == 0
    assert metrics['loss_float64_recomputations'] == int(amplitude > 1e38)


def test_numerical_failure_is_not_retried_or_substituted_for_final(tmp_path):
    spec = dict(_spec(), trial_plan_hash='plan', attack_implementation_hash='implementation')
    identifier = runner.run_id(spec)
    status_path = tmp_path / 'status' / f'{identifier}.json'
    attempts = []
    def launch(_spec, _payload, _output, path, status, attempt, _offset):
        attempts.append(attempt)
        runner._write_status(status, {'numerical_evidence':NumericalFailure('nonfinite_update', 'aggregation', 'test', 2).evidence})
        return 1, 111
    assert runner._run_spec_with_retries(spec, _args(), tmp_path, tmp_path/'rounds'/f'{identifier}.csv', status_path, launch_fn=launch)
    status = runner._read_status(status_path)
    assert attempts == [1]
    assert runner.valid_numerical_terminal(status, spec)
    assert not runner.valid_numerical_terminal(status, dict(spec, attack_implementation_hash='changed'))
    raw = tmp_path / 'raw'
    raw.mkdir()
    (raw/f'{identifier}.json').write_text(json.dumps([{'round':1, 'split':'server', 'loss':1e20, 'accuracy':0.1}]))
    row = runner.numerical_terminal_row(spec, status, tmp_path)
    assert math.isnan(row['final_accuracy']) and row['last_finite_accuracy'] == 0.1


def test_matrix_continues_after_numerical_terminal(monkeypatch, tmp_path):
    specs = [_spec(), dict(_spec(), seed=46)]
    args = _args()
    args.rerun = True
    launched = []
    monkeypatch.setattr(runner, '_write_execution_environment', lambda *_:None)
    monkeypatch.setattr(runner, '_write_current_summary', lambda *_:pd.DataFrame())
    original = runner._run_spec_with_retries
    def launch(spec, payload, output, path, status, attempt, offset):
        launched.append(spec['seed'])
        if spec['seed'] == 42:
            runner._write_status(status, {'numerical_evidence':NumericalFailure('nonfinite_update', 'aggregation', 'test', 1).evidence})
            return 1, 111
        runner._atomic_write_csv(_round_frame(2), path)
        return 0, 222
    monkeypatch.setattr(runner, '_run_spec_with_retries', lambda *a:original(*a, launch_fn=launch))
    runner._run_specs(specs, args, tmp_path, tmp_path/'rounds')
    assert launched == [42, 46]


def test_worker_persists_wrapped_numerical_evidence(monkeypatch, tmp_path):
    spec = _spec()
    status = tmp_path/'status.json'
    monkeypatch.setattr(runner, '_build_spec_config', lambda *_:None)
    monkeypatch.setattr(runner, 'shutdown_ray_runtime', lambda:None)
    monkeypatch.setattr(runner, '_find_ray_session', lambda _:None)
    def fail(*_, **__):
        raise RuntimeError('RayTaskError: ' + str(NumericalFailure('nonfinite_logits', 'server_evaluation', 'test', 11)))
    monkeypatch.setattr(runner, 'run_simulation', fail)
    with pytest.raises(RuntimeError):
        runner._run_spec_worker(spec, vars(_args()), str(tmp_path), str(tmp_path/'round.csv'), str(status), 1, 0)
    assert runner._read_status(status)['numerical_evidence']['round'] == 11


def test_explicit_rerun_does_not_reuse_stale_numerical_error(tmp_path):
    spec = _spec()
    path = tmp_path/'status.json'
    runner._write_status(path, {'numerical_evidence':NumericalFailure('nonfinite_update','aggregation','old',1).evidence})
    with pytest.raises(RuntimeError, match='failed after 1 attempts'):
        runner._run_spec_with_retries(spec, _args(retries=0), tmp_path, tmp_path/'round.csv', path,
                                     launch_fn=lambda *_:(1, 123))
    assert runner._read_status(path)['state'] == 'failed'


@pytest.mark.parametrize('kind', ['finite', 'update_inf', 'negative_bn', 'client_failure'])
def test_strategy_numeric_boundary_and_evidence(kind):
    from types import SimpleNamespace
    from config.config_loader import StrategyConfig, DefenseConfig
    from flwr.common import FitRes, Code, Status, ndarrays_to_parameters
    from strategies.fed_strategy import FedSecStrategy
    strategy = FedSecStrategy(StrategyConfig(), DefenseConfig(), [np.ones(2, np.float32)],
                              parameter_names={'0':'bn.running_var'})
    if kind == 'client_failure':
        with pytest.raises(NumericalFailure) as error:
            strategy.aggregate_fit(11, [], [RuntimeError(str(NumericalFailure('nonfinite_client_metric','client_training','test')))])
        assert error.value.evidence['round'] == 11
        return
    value = {'finite':1, 'update_inf':float('inf'), 'negative_bn':-1}[kind]
    result = FitRes(Status(Code.OK, ''), ndarrays_to_parameters([np.full(2, value, np.float32)]), 8,
                    {'client_id':1, 'train_loss':1e20, 'random_noise_reference_norm':0.2})
    if kind != 'finite':
        with pytest.raises(NumericalFailure): strategy.aggregate_fit(11, [(SimpleNamespace(cid='1'),result)], [])
    else:
        _, metrics = strategy.aggregate_fit(11, [(SimpleNamespace(cid='1'),result)], [])
        assert metrics['train_loss'] == 1e20
        assert strategy.last_client_records[0]['random_noise_reference_norm'] == 0.2
