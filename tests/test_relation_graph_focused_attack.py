"""Contract tests for the focused Clean+single-attack relation-graph protocol."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from experiments.relation_graph_mnist_paired import (
    FOCUSED_ATTACK_PURPOSE, focused_groups, present_attacks, validate,
)

CONFIG = Path(__file__).resolve().parents[1] / "config" / (
    "relation_graph_mnist_neurotoxin_clean_120.json")


def focused_config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8-sig"))


def test_focused_config_is_valid():
    cfg = focused_config()
    assert cfg["purpose"] == FOCUSED_ATTACK_PURPOSE
    assert cfg["evaluation_channel"] == "separate"
    assert cfg["defense_variant"] == "group_zero_weight"
    assert present_attacks(cfg) == ("neurotoxin",)
    assert cfg["groups"] == focused_groups("neurotoxin")
    validate(cfg)


def test_focused_groups_require_a_known_attack():
    with pytest.raises(ValueError):
        focused_groups("dba")


def test_focused_protocol_rejects_shared_evaluation_channel():
    cfg = focused_config()
    cfg["evaluation_channel"] = "shared"
    with pytest.raises(ValueError, match="evaluation_channel"):
        validate(cfg)


def test_focused_protocol_rejects_extra_attack_pair():
    cfg = focused_config()
    cfg["groups"] = focused_groups("neurotoxin") + focused_groups("badnet")[2:]
    with pytest.raises(ValueError, match="exactly one attack"):
        validate(cfg)


def test_focused_protocol_rejects_non_matching_group_order():
    cfg = focused_config()
    cfg["groups"] = list(reversed(focused_groups("neurotoxin")))
    with pytest.raises(ValueError, match="clean pair plus the attack pair"):
        validate(cfg)


def test_focused_protocol_rejects_short_attack_window():
    cfg = focused_config()
    cfg["attack"]["attack_end_round"] = cfg["total_rounds"] - 1
    with pytest.raises(ValueError, match="Focused attack mechanism"):
        validate(cfg)


def test_focused_protocol_rejects_other_defense_variant():
    cfg = focused_config()
    cfg["defense_variant"] = "soft_downweight"
    with pytest.raises(ValueError, match="defense_variant"):
        validate(cfg)


def test_focused_protocol_rejects_neurotoxin_public_delta_mask():
    cfg = focused_config()
    cfg["attack_profiles"]["neurotoxin"]["neurotoxin_mask_source"] = "public_delta"
    with pytest.raises(ValueError, match="clean-gradient GradMask"):
        validate(cfg)


def test_focused_protocol_copy_is_independent():
    cfg = focused_config()
    changed = copy.deepcopy(cfg)
    changed["groups"][0]["name"] = "tampered"
    assert cfg["groups"][0]["name"] == "clean_fedavg"