import copy
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from experiments import rtc_m3_all_attacks as stage
from experiments.rtc_v3 import byzantine
from attacks.spec import attack_source_hash


def test_matrix_only_m3_and_no_implicit_fedavg(tmp_path):
    conditions = stage.conditions()
    assert len(conditions) == 12 and len({c['name'] for c in conditions}) == 12
    assert {c['attack'] for c in conditions} == {'none', *byzantine.CANONICAL_ATTACKS}
    for c in conditions:
        args = stage.command(tmp_path, c)
        assert args[-1] == '--dry-run' and '--execute' not in args
        assert args[args.index('--defenses') + 1] == stage.M3 and '--skip-clean' in args
        assert args[args.index('--seeds') + 1] == '42'
        assert args[args.index('--batch-size') + 1] == '96'
        assert float(args[args.index('--ray-client-num-gpus') + 1]) == .125
        assert ('--byzantine-evaluation-only' in args) == (c['attack'] == 'random_noise')


def test_exploratory_random_does_not_weaken_formal_gate(tmp_path):
    entry = copy.deepcopy(stage.prior.read_json(stage.PROTOCOL)['attacks']['random_noise'])
    payload = dict(schema_version='RTCByzantineAttackFreezeV1', implementation_source_sha256=attack_source_hash(),
        purpose='fixed_strength_evaluation_only', attacks={'random_noise': entry})
    path = tmp_path / 'freeze.json'
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='verified FedAvg'):
        byzantine._load_attack_freeze(path, ('random_noise',))
    assert byzantine._load_attack_freeze(path, ('random_noise',), evaluation_only=True)['random_noise'] == entry
    entry['strength_level'] = 'strong'
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='exploratory'):
        byzantine._load_attack_freeze(path, ('random_noise',), evaluation_only=True)


@pytest.mark.parametrize('value', ['2.5', 'NaN', 'Infinity'])
def test_integral_csv_rejects_bad_values(tmp_path, value):
    path = tmp_path / 'rows.csv'; path.write_text('rtc_r1e_required\n' + value + '\n')
    with pytest.raises(ValueError, match='Nonintegral'):
        stage.numeric_csv(path)


def test_integral_csv_accepts_decimal_serialization(tmp_path):
    path = tmp_path / 'rows.csv'
    path.write_text('round,rtc_r1e_required,fit_rtc_r1e_quorum_numerator\n1.0,2.0,2.0\n')
    assert stage.numeric_csv(path) == [dict(round='1', rtc_r1e_required='2', fit_rtc_r1e_quorum_numerator='2')]


def test_preflight_rejects_partial_and_orphan_before_training(tmp_path):
    cell = dict(batch='attack', run_id='unit')
    p = tmp_path / 'attack/status/unit.json'; p.parent.mkdir(parents=True)
    p.write_text(json.dumps(dict(state='failed', exit_code=1, last_round=27)))
    with pytest.raises(ValueError, match='Incomplete'):
        stage.preflight(tmp_path, dict(cells=[cell]))
    p.unlink(); p = tmp_path / 'attack/rounds/unit.csv'; p.parent.mkdir()
    p.write_text('round\n0\n')
    with pytest.raises(ValueError, match='Orphaned'):
        stage.preflight(tmp_path, dict(cells=[cell]))


def test_asr_window_not_round_zero_or_warmup_and_missing_rejected():
    run = dict(cell=dict(attack='dba', batch='dba'),
        rounds=[dict(round=str(i),server_accuracy='.8',server_asr='1' if i<11 else str((i-10)/100)) for i in range(61)],
        clients=[dict(round=str(i),attack_active='False',rtc_r1_q='1',rtc_r2_q='1') for i in range(1,61)])
    with patch.object(stage.prior, 'summarize', side_effect=lambda _: dict(asr=None,asr_reason='old')):
        result = stage.metrics(run)
        assert result['asr_mean'] == pytest.approx(.255) and result['asr_peak'] == .5
        assert result['asr_tail10'] == pytest.approx(.455) and result['asr_final'] == .5
        run['rounds'][20]['server_asr'] = ''
        with pytest.raises(ValueError):
            stage.metrics(run)


def test_accepted_config_rejects_silent_g2_enable():
    protocol = stage.prior.read_json(stage.PROTOCOL)
    cfg = dict(client=protocol['training'],ray=dict(client_num_gpus=.125),
        dataset=dict(name='cifar10',partition='iid'),model=dict(architecture='resnet18',pretrained=False),
        federation=dict(num_rounds=60,num_clients=20,deterministic_client_training=True),
        security=dict(defense=dict(custom_params=copy.deepcopy(protocol['accepted_custom_params']))))
    stage.verify_config(cfg, protocol)
    cfg['security']['defense']['custom_params']['lower_tail_mode'] = 'cap'
    with pytest.raises(AssertionError, match='M3 mechanism'):
        stage.verify_config(cfg, protocol)
