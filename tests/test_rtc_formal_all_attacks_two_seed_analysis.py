import json
from pathlib import Path

import pandas as pd

from attacks.spec import attack_source_hash
from analysis.rtc_formal_all_attacks_two_seed import analyze_results as formal


ROOT = Path(__file__).resolve().parents[1]


def test_formal_inventory_and_freeze_are_exact() -> None:
    assert formal.EXPECTED_CELLS == 182
    assert len(formal.ATTACKS) == 10
    assert len(formal.DEFENSES) == 10
    assert formal.SEEDS == (42, 46)
    assert formal.CLEAN_SEEDS == (42, 46)
    assert formal.CLEAN_DEFENSES == ("fedavg",)
    assert "multi_krum" in formal.DEFENSES
    assert "rtc_full" not in formal.DEFENSES
    assert "label_flip_targeted" not in formal.ATTACKS

    freeze = json.loads(
        (ROOT / "config" / "rtc_v3_formal_all_attacks_two_seed.freeze.json").read_text(
            encoding="utf-8"
        )
    )
    assert freeze["implementation_source_sha256"] == attack_source_hash()
    assert set(freeze["attacks"]) == set(formal.ATTACKS) - {"none"}
    assert freeze["attacks"]["lie"]["parameters"]["lie_z"] == 0.5
    assert freeze["attacks"]["sign_flip"]["parameters"]["sign_flip_scale"] == 1.0


def test_rankings_use_accuracy_for_untargeted_and_asr_for_targeted() -> None:
    rows = []
    for attack in formal.ATTACKS:
        for index, defense in enumerate(formal.DEFENSES):
            rows.append(
                {
                    "attack": attack,
                    "defense": defense,
                    "mean_accuracy": 0.9 - index * 0.01,
                    "mean_asr": 0.01 + index * 0.01 if attack in formal.TARGETED else float("nan"),
                    "mean_accuracy_drop_vs_same_seed_fedavg_clean": index * 0.01,
                }
            )
    ranked = formal._rankings(pd.DataFrame(rows))
    assert ranked.loc[ranked["attack"].eq("lie")].iloc[0]["defense"] == "fedavg"
    assert ranked.loc[ranked["attack"].eq("dba")].iloc[0]["defense"] == "fedavg"
    assert set(ranked.groupby("attack").size()) == {10}


def test_rtc_asr_improvement_sign_is_comparator_minus_rtc() -> None:
    rows = []
    for attack in formal.ATTACKS:
        for seed in formal._seeds_for_attack(attack):
            for defense in formal.DEFENSES:
                rtc = defense == formal.RTC
                rows.append(
                    {
                        "attack": attack,
                        "seed": seed,
                        "defense": defense,
                        "mean_accuracy": 0.8 + (0.01 if rtc else 0.0),
                        "final_accuracy": 0.85 + (0.01 if rtc else 0.0),
                        "mean_asr": 0.02 if rtc else 0.03,
                        "peak_asr": 0.04 if rtc else 0.05,
                        "malicious_weight_share": 0.1,
                        "malicious_impact_share": 0.1,
                        "zero_update_mass": 0.0,
                        "benign_clipping_rate": 0.0,
                    }
                )
    compared = formal._rtc_comparisons(pd.DataFrame(rows))
    dba = compared.loc[compared["attack"].eq("dba")]
    assert (dba["mean_asr_improvement_comparator_minus_rtc"] > 0).all()
    lie = compared.loc[compared["attack"].eq("lie")]
    assert lie["mean_asr_improvement_comparator_minus_rtc"].isna().all()
