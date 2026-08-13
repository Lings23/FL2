"""Fail-closed audit for the RTC-V3 semantic candidate.

The auditor only writes evidence and a Markdown report.  It intentionally has
no code path that creates a promotion attestation or changes promotion_ready.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from defenses.rtc.calibration import CalibrationManifest, SCHEMA_VERSION_V3

CLEAN_BENIGN_CLIPPING_MAX = 0.03
REQUIRED_EXTRA_HELDOUT_SEEDS = 4


def _gate(
    name: str,
    passed: bool,
    observed: Any,
    expected: str,
    *,
    promotion_only: bool = False,
) -> dict[str, Any]:
    return {
        "gate": name,
        "passed": bool(passed),
        "observed": observed,
        "expected": expected,
        "promotion_only": bool(promotion_only),
    }


def _execution_gates(root: Path) -> tuple[bool, list[dict[str, Any]]]:
    path = root / "execution_validation.csv"
    if not path.is_file():
        return False, [_gate("execution_validation_present", False, str(path), "file")]
    frame = pd.read_csv(path)
    if not {"gate", "passed"}.issubset(frame):
        return False, [_gate("execution_validation_schema", False, list(frame), "gate,passed")]
    rows = []
    for _, row in frame.iterrows():
        value = str(row["passed"]).strip().lower() in {"true", "1", "1.0"}
        rows.append(_gate(f"strict:{row['gate']}", value, row["passed"], "true"))
    return bool(rows and all(row["passed"] for row in rows)), rows


def _round_frames(root: Path) -> pd.DataFrame:
    paths = sorted((root / "rounds").glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"no round files in {root}")
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["source_file"] = path.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _as_paths(value: str | Path | Sequence[str | Path]) -> list[Path]:
    values = [value] if isinstance(value, (str, Path)) else list(value)
    paths = [Path(item).resolve() for item in values]
    if not paths or len(paths) != len(set(paths)):
        raise ValueError("artifact roots must be non-empty and unique")
    return paths


def _attack_summary(
    roots: Path | Sequence[Path], defense: str
) -> pd.DataFrame:
    root_list = [roots] if isinstance(roots, Path) else list(roots)
    frames = []
    for root in root_list:
        path = root / "comparison_runs.csv"
        if not path.is_file():
            raise FileNotFoundError(f"missing comparison summary: {path}")
        frame = pd.read_csv(path)
        selected = frame[
            (frame["defense"].astype(str) == str(defense))
            & (frame["attack"].astype(str) != "none")
        ].copy()
        selected["source_root"] = str(root)
        frames.append(selected)
    result = pd.concat(frames, ignore_index=True)
    keys = ["trial_plan_hash", "seed", "attack"]
    if result.duplicated(keys).any():
        raise ValueError(f"duplicate {defense} held-out statistical keys")
    return result


def _clean_summary(root: Path, defense: str) -> pd.DataFrame:
    path = root / "comparison_runs.csv"
    if not path.is_file():
        raise FileNotFoundError(f"missing clean comparison summary: {path}")
    frame = pd.read_csv(path)
    selected = frame[frame["defense"].astype(str) == str(defense)].copy()
    if len(selected) != 1:
        raise ValueError(
            f"clean root must contain exactly one {defense} trajectory: {root}"
        )
    return selected


def _clean_rounds_are_inactive(root: Path) -> bool:
    rounds = _round_frames(root)
    active = pd.to_numeric(
        rounds.get("planned_attack_active"), errors="coerce"
    ).fillna(0.0)
    return bool((active <= 0.0).all())


def audit_candidate(
    *,
    candidate_manifest: str | Path,
    calibration_validation: str | Path,
    candidate_validation: str | Path,
    performance_gates: str | Path,
    clean_root: str | Path,
    baseline_clean_root: str | Path,
    attack_root: str | Path | Sequence[str | Path],
    recovery_root: str | Path,
    baseline_attack_root: str | Path | Sequence[str | Path],
    audit_stage: str = "promotion",
) -> dict[str, Any]:
    if audit_stage not in {"seed42", "promotion"}:
        raise ValueError("audit_stage must be seed42 or promotion")
    manifest = CalibrationManifest.load(candidate_manifest)
    metadata = manifest.payload.get("metadata", {})
    calibration = json.loads(Path(calibration_validation).read_text(encoding="utf-8"))
    frozen = json.loads(Path(candidate_validation).read_text(encoding="utf-8"))
    performance = json.loads(Path(performance_gates).read_text(encoding="utf-8"))
    predevelopment_hash = str(
        metadata.get("development_provenance", {}).get(
            "predevelopment_candidate_hash", ""
        )
    )
    clean_path = Path(clean_root)
    baseline_clean_path = Path(baseline_clean_root)
    attack_paths = _as_paths(attack_root)
    recovery_path = Path(recovery_root)
    baseline_paths = _as_paths(baseline_attack_root)
    gates: list[dict[str, Any]] = [
        _gate(
            "schema_v3",
            manifest.payload["schema_version"] == SCHEMA_VERSION_V3,
            manifest.payload["schema_version"],
            SCHEMA_VERSION_V3,
        ),
        _gate(
            "candidate_not_promotion_ready",
            not bool(metadata.get("promotion_ready", False)),
            metadata.get("promotion_ready", False),
            "false",
        ),
        _gate(
            "formal_calibration_passed",
            bool(calibration.get("passed", False))
            and str(calibration.get("content_hash", "")) == predevelopment_hash,
            {
                "passed": calibration.get("passed", False),
                "content_hash": calibration.get("content_hash"),
                "expected_predevelopment_hash": predevelopment_hash,
            },
            "passed=true and content_hash equals predevelopment hash",
        ),
        _gate(
            "frozen_candidate_validation",
            bool(frozen.get("passed", False))
            and str(frozen.get("content_hash", "")) == manifest.hash
            and not bool(frozen.get("promotion_ready", True)),
            {
                "passed": frozen.get("passed", False),
                "content_hash": frozen.get("content_hash"),
                "candidate_hash": manifest.hash,
                "promotion_ready": frozen.get("promotion_ready"),
            },
            "passed=true, candidate hash bound, promotion_ready=false",
        ),
        _gate(
            "performance_passed",
            bool(performance.get("passed", False)),
            performance.get("passed", False),
            "true",
        ),
    ]
    strict_roots = [
        ("clean", clean_path),
        ("baseline_clean", baseline_clean_path),
        ("recovery", recovery_path),
    ]
    strict_roots.extend(
        (f"attacks:{index}", root)
        for index, root in enumerate(attack_paths, start=1)
    )
    strict_roots.extend(
        (f"baseline_attacks:{index}", root)
        for index, root in enumerate(baseline_paths, start=1)
    )
    for label, root in strict_roots:
        passed, strict = _execution_gates(root)
        gates.append(_gate(f"{label}_all_strict_gates", passed, passed, "true"))
        gates.extend(strict)

    clean = _round_frames(clean_path)
    fit_clean = clean[
        (pd.to_numeric(clean["round"], errors="coerce") > 0)
        & clean.get("fit_rtc_v3_calibration_hash").notna()
    ]
    overflow = pd.to_numeric(
        fit_clean.get("fit_rtc_v3_cone_overflow_assignments"), errors="coerce"
    ).fillna(0.0)
    principals = pd.to_numeric(
        fit_clean.get("fit_rtc_v3_principal_count"), errors="coerce"
    ).replace(0.0, np.nan)
    overflow_rate = float((overflow / principals).mean())
    updates = float(
        pd.to_numeric(
            fit_clean.get("fit_rtc_v3_cone_update_applied_count"), errors="coerce"
        ).fillna(0.0).max()
    )
    violation = float(
        pd.to_numeric(
            fit_clean.get("fit_rtc_v3_max_constraint_violation"), errors="coerce"
        ).fillna(np.inf).max()
    )
    clipping_rate = float(
        pd.to_numeric(
            fit_clean.get("fit_rtc_v3_clipping_rate"), errors="coerce"
        ).mean()
    )
    clean_semantic_active_columns = [
        column
        for column in fit_clean
        if column.startswith("fit_rtc_v3_semantic_active_w")
    ]
    clean_semantic_activation = float(
        fit_clean[clean_semantic_active_columns].fillna(0.0).to_numpy(dtype=float).mean()
    ) if clean_semantic_active_columns else float("inf")
    gates.extend(
        [
            _gate("clean_overflow", overflow_rate <= 0.05, overflow_rate, "<=0.05"),
            _gate("online_prototypes_frozen", updates == 0.0, updates, "0"),
            _gate("clean_budget_violation", violation <= 1e-8, violation, "<=1e-8"),
            _gate(
                "clean_benign_clipping",
                clipping_rate <= CLEAN_BENIGN_CLIPPING_MAX,
                clipping_rate,
                f"<={CLEAN_BENIGN_CLIPPING_MAX}",
            ),
            _gate(
                "clean_semantic_budget_activation",
                clean_semantic_activation <= 0.01,
                clean_semantic_activation,
                "<=0.01",
            ),
        ]
    )
    candidate_clean = _clean_summary(clean_path, "rtc_v3")
    baseline_clean = _clean_summary(baseline_clean_path, "rtc_v3")
    # Reuse the shared FedAvg clean counterfactual already paired in the V2
    # root; rerunning the identical trajectory would waste a full experiment.
    fedavg_clean = _clean_summary(baseline_clean_path, "fedavg")
    clean_keys = ["seed", "trial_plan_hash"]
    clean_comparison = candidate_clean.merge(
        baseline_clean,
        on=clean_keys,
        suffixes=("_candidate", "_baseline"),
        validate="one_to_one",
    )
    clean_comparison = clean_comparison.merge(
        fedavg_clean[[*clean_keys, "final_accuracy"]],
        on=clean_keys,
        how="left",
        validate="one_to_one",
    )
    clean_complete = bool(len(clean_comparison) == 1)
    clean_inactive = bool(
        _clean_rounds_are_inactive(clean_path)
        and _clean_rounds_are_inactive(baseline_clean_path)
    )
    clean_accuracy_delta = (
        float(
            clean_comparison.iloc[0]["final_accuracy_candidate"]
            - clean_comparison.iloc[0]["final_accuracy_baseline"]
        )
        if clean_complete
        else float("nan")
    )
    gates.extend(
        [
            _gate(
                "clean_strict_plan_join",
                clean_complete,
                int(len(clean_comparison)),
                "exactly one candidate/V2/FedAvg row joined by seed+trial_plan_hash",
            ),
            _gate(
                "clean_no_attack_rounds",
                clean_inactive,
                clean_inactive,
                "planned_attack_active=0 for every candidate/V2/FedAvg round",
            ),
            _gate(
                "clean_final_accuracy_vs_v2",
                np.isfinite(clean_accuracy_delta) and clean_accuracy_delta >= -0.01,
                clean_accuracy_delta,
                ">=-0.01",
            ),
        ]
    )

    candidate = _attack_summary(attack_paths, "rtc_v3")
    baseline = _attack_summary(baseline_paths, "rtc_v3")
    clip_only = _attack_summary(attack_paths, "clip_only")
    keys = ["trial_plan_hash", "attack", "seed"]
    comparison = candidate.merge(
        baseline,
        on=keys,
        suffixes=("_candidate", "_baseline"),
        validate="one_to_one",
    )
    comparison = comparison.merge(
        clip_only[
            [
                *keys,
                "active_asr",
                "peak_asr",
                "final_accuracy",
            ]
        ],
        on=keys,
        how="left",
        validate="one_to_one",
        suffixes=("", "_clip"),
    )
    expected_attacks = {"label_flip_targeted", "label_flip_all_reverse"}
    observed_attacks = set(candidate["attack"].astype(str))
    observed_seeds = set(int(value) for value in candidate["seed"])
    gates.extend(
        [
            _gate(
                "heldout_attacks_complete",
                expected_attacks.issubset(observed_attacks),
                sorted(observed_attacks),
                str(sorted(expected_attacks)),
            ),
            _gate("seed42_present", 42 in observed_seeds, sorted(observed_seeds), "contains 42"),
            _gate(
                "four_extra_heldout_seeds_present",
                len(observed_seeds - {42}) >= REQUIRED_EXTRA_HELDOUT_SEEDS,
                sorted(observed_seeds),
                (
                    "seed 42 plus at least "
                    f"{REQUIRED_EXTRA_HELDOUT_SEEDS} other held-out seeds"
                ),
                promotion_only=True,
            ),
        ]
    )
    metric_rows: list[dict[str, Any]] = []
    for _, row in comparison.iterrows():
        asr_delta = float(row["active_asr_candidate"] - row["active_asr_baseline"])
        peak_delta = float(row["peak_asr_candidate"] - row["peak_asr_baseline"])
        accuracy_delta = float(
            row["final_accuracy_candidate"] - row["final_accuracy_baseline"]
        )
        asr_delta_clip = float(row["active_asr_candidate"] - row["active_asr"])
        peak_delta_clip = float(row["peak_asr_candidate"] - row["peak_asr"])
        plans_match = bool(str(row["trial_plan_hash"]).strip())
        item = {
            "attack": str(row["attack"]),
            "seed": int(row["seed"]),
            "active_asr_delta": asr_delta,
            "peak_asr_delta": peak_delta,
            "final_accuracy_delta": accuracy_delta,
            "active_asr_delta_vs_clip": asr_delta_clip,
            "peak_asr_delta_vs_clip": peak_delta_clip,
            "trial_plan_hash_match": plans_match,
        }
        metric_rows.append(item)
        prefix = f"{item['attack']}:seed{item['seed']}"
        gates.append(_gate(f"{prefix}:plan_hash", plans_match, plans_match, "true"))
        if str(row["attack"]) == "label_flip_targeted":
            gates.extend(
                [
                    _gate(f"{prefix}:active_asr_vs_v2", asr_delta <= -0.10, asr_delta, "<=-0.10"),
                    _gate(f"{prefix}:peak_asr_vs_v2", peak_delta <= -0.15, peak_delta, "<=-0.15"),
                    _gate(f"{prefix}:active_asr_vs_clip", asr_delta_clip <= -0.10, asr_delta_clip, "<=-0.10"),
                    _gate(f"{prefix}:peak_asr_vs_clip", peak_delta_clip <= -0.15, peak_delta_clip, "<=-0.15"),
                    _gate(f"{prefix}:accuracy", accuracy_delta >= -0.01, accuracy_delta, ">=-0.01"),
                ]
            )
        else:
            gates.extend(
                [
                    _gate(f"{prefix}:active_asr", asr_delta <= 0.02, asr_delta, "<=+0.02"),
                    _gate(f"{prefix}:accuracy", accuracy_delta >= -0.01, accuracy_delta, ">=-0.01"),
                ]
            )

    attack_rounds = pd.concat(
        [_round_frames(root) for root in attack_paths], ignore_index=True
    )
    candidate_attack_rounds = attack_rounds[
        attack_rounds.get("fit_rtc_v3_calibration_hash").notna()
        & (pd.to_numeric(attack_rounds.get("planned_attack_active"), errors="coerce") > 0)
    ]
    attack_active_columns = [
        column
        for column in candidate_attack_rounds
        if column.startswith("fit_rtc_v3_semantic_active_w")
    ]
    attack_activation = float(
        candidate_attack_rounds[attack_active_columns]
        .fillna(0.0)
        .to_numpy(dtype=float)
        .mean()
    ) if attack_active_columns else 0.0
    attack_updates = float(
        pd.to_numeric(
            candidate_attack_rounds.get("fit_rtc_v3_cone_update_applied_count"),
            errors="coerce",
        ).fillna(0.0).max()
    )
    attack_violation = float(
        pd.to_numeric(
            candidate_attack_rounds.get("fit_rtc_v3_max_constraint_violation"),
            errors="coerce",
        ).fillna(np.inf).max()
    )
    gates.extend(
        [
            _gate(
                "attack_semantic_budget_activation",
                attack_activation > clean_semantic_activation,
                attack_activation,
                f">clean {clean_semantic_activation}",
            ),
            _gate("attack_prototype_contamination", attack_updates == 0.0, attack_updates, "0"),
            _gate("attack_budget_violation", attack_violation <= 1e-8, attack_violation, "<=1e-8"),
        ]
    )

    recovery = _round_frames(recovery_path)
    recovery_segments: list[pd.Series] = []
    for _, run in recovery.groupby("source_file", sort=True):
        ordered = run.sort_values("round")
        active = pd.to_numeric(
            ordered.get("planned_attack_active"), errors="coerce"
        ).fillna(0.0)
        active_rounds = pd.to_numeric(
            ordered.loc[active > 0.0, "round"], errors="coerce"
        ).dropna()
        if active_rounds.empty:
            continue
        post = ordered[
            pd.to_numeric(ordered["round"], errors="coerce")
            > int(active_rounds.max())
        ]
        series = pd.to_numeric(
            post.get("fit_rtc_v3_semantic_risk_mean"), errors="coerce"
        ).dropna()
        if len(series) >= 10:
            recovery_segments.append(series.reset_index(drop=True))
    recovery_declines = bool(
        recovery_segments
        and all(
            float(series.iloc[-5:].mean()) <= float(series.iloc[:5].mean())
            for series in recovery_segments
        )
    )
    first5 = (
        float(np.mean([series.iloc[:5].mean() for series in recovery_segments]))
        if recovery_segments
        else None
    )
    last5 = (
        float(np.mean([series.iloc[-5:].mean() for series in recovery_segments]))
        if recovery_segments
        else None
    )
    gates.append(
        _gate(
            "controlled_recovery_risk_declines",
            recovery_declines,
            {
                "segments": len(recovery_segments),
                "first5": first5,
                "last5": last5,
            },
            "last5<=first5 with >=10 observed rounds",
        )
    )
    seed42_passed = bool(
        gates
        and all(item["passed"] for item in gates if not item["promotion_only"])
    )
    promotion_evidence_passed = bool(
        gates and all(item["passed"] for item in gates)
    )
    passed = seed42_passed if audit_stage == "seed42" else promotion_evidence_passed
    return {
        "artifact_kind": "rtc_v3_semantic_temporal_exposure_gate_audit",
        "candidate_manifest_hash": manifest.hash,
        "audit_stage": audit_stage,
        "passed": passed,
        "seed42_passed": seed42_passed,
        "promotion_evidence_passed": promotion_evidence_passed,
        "promotion_attestation_generated": False,
        "gates": gates,
        "attack_comparison": metric_rows,
    }


def write_report(evidence: Mapping[str, Any], destination: Path) -> None:
    lines = [
        "# RTC-V3 语义—时间—暴露闭环实验报告",
        "",
        f"- Candidate hash: `{evidence['candidate_manifest_hash']}`",
        f"- 全部门禁通过: `{str(bool(evidence['passed'])).lower()}`",
        "- Promotion attestation generated: `false`",
        "",
        "## Gates",
        "",
        "| Gate | Passed | Observed | Expected |",
        "|---|---:|---|---|",
    ]
    for gate in evidence["gates"]:
        observed = json.dumps(gate["observed"], ensure_ascii=False).replace("|", "\\|")
        lines.append(
            f"| {gate['gate']} | {str(bool(gate['passed'])).lower()} | "
            f"`{observed}` | `{gate['expected']}` |"
        )
    lines.extend(
        [
            "",
            "## 结论",
            "",
            (
                "所有门禁通过；该结果仍只表示候选可进入独立 promotion 流程，本工具未生成 attestation。"
                if evidence["passed"]
                else "至少一个门禁失败；保留 candidate，禁止生成 promotion attestation。"
            ),
        ]
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--calibration-validation", required=True)
    parser.add_argument("--performance-gates", required=True)
    parser.add_argument("--clean-root", required=True)
    parser.add_argument("--baseline-clean-root", required=True)
    parser.add_argument("--attack-root", action="append", required=True)
    parser.add_argument("--recovery-root", required=True)
    parser.add_argument("--baseline-attack-root", action="append", required=True)
    parser.add_argument(
        "--evidence-output",
        default="logs/rtc_v3_semantic_gate_audit.json",
    )
    parser.add_argument(
        "--report-output",
        default="docs/rtc_v3_semantic_temporal_exposure_experiment_report.md",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    evidence = audit_candidate(
        candidate_manifest=args.candidate_manifest,
        calibration_validation=args.calibration_validation,
        performance_gates=args.performance_gates,
        clean_root=args.clean_root,
        baseline_clean_root=args.baseline_clean_root,
        attack_root=args.attack_root,
        recovery_root=args.recovery_root,
        baseline_attack_root=args.baseline_attack_root,
    )
    evidence_path = Path(args.evidence_output).resolve()
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_report(evidence, Path(args.report_output).resolve())
    if not evidence["passed"]:
        raise RuntimeError("semantic candidate gates failed; attestation is forbidden")
    return evidence_path


if __name__ == "__main__":
    main()
