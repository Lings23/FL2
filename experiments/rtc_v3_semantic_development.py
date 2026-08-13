"""Freeze an unchanged semantic candidate after development-only review."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from defenses.rtc.calibration import CalibrationManifest, content_hash
from experiments.rtc_v3_formal_calibration import DEVELOPMENT_SEEDS, EVALUATION_SEEDS


REQUIRED_DEVELOPMENT_ATTACKS = (
    "label_flip_targeted",
    "label_flip_all_reverse",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _strict_gates(root: Path) -> tuple[bool, list[dict[str, Any]]]:
    source = root / "execution_validation.csv"
    if not source.is_file():
        raise FileNotFoundError(f"development execution validation missing: {source}")
    frame = pd.read_csv(source)
    if not {"gate", "passed"}.issubset(frame):
        raise ValueError("development execution validation has invalid schema")
    rows = []
    for _, row in frame.iterrows():
        passed = str(row["passed"]).strip().lower() in {"true", "1", "1.0"}
        rows.append(
            {"root": str(root), "gate": str(row["gate"]), "passed": passed}
        )
    if not rows:
        raise ValueError("development execution validation is empty")
    return all(row["passed"] for row in rows), rows


def build_development_evidence(
    *,
    source_manifest: str | Path,
    development_root: str | Path | Sequence[str | Path],
    destination: str | Path,
) -> Path:
    """Derive immutable freeze evidence from development-only run artifacts."""

    manifest = CalibrationManifest.load(source_manifest)
    root_values = (
        [development_root]
        if isinstance(development_root, (str, Path))
        else list(development_root)
    )
    roots = tuple(Path(value).resolve() for value in root_values)
    if not roots:
        raise ValueError("at least one development root is required")
    if len(set(roots)) != len(roots):
        raise ValueError("development roots must be unique")

    specs: list[Mapping[str, Any]] = []
    manifest_sources: list[dict[str, Any]] = []
    summary_sources: list[dict[str, Any]] = []
    strict_rows: list[dict[str, Any]] = []
    strict_passed = True
    for root in roots:
        experiment_manifest_path = root / "experiment_manifest.json"
        summary_path = root / "comparison_runs.csv"
        if not experiment_manifest_path.is_file() or not summary_path.is_file():
            raise FileNotFoundError(
                f"development run manifest or comparison summary missing: {root}"
            )
        experiment_manifest = json.loads(
            experiment_manifest_path.read_text(encoding="utf-8")
        )
        root_specs = experiment_manifest.get("specs")
        if not isinstance(root_specs, Sequence) or isinstance(root_specs, (str, bytes)):
            raise ValueError(
                f"development experiment manifest specs are invalid: {root}"
            )
        specs.extend(spec for spec in root_specs if isinstance(spec, Mapping))
        root_strict_passed, root_strict_rows = _strict_gates(root)
        strict_passed = strict_passed and root_strict_passed
        strict_rows.extend(root_strict_rows)
        manifest_sources.append(
            {"path": str(experiment_manifest_path), "sha256": _sha256(experiment_manifest_path)}
        )
        summary_sources.append(
            {"path": str(summary_path), "sha256": _sha256(summary_path)}
        )
    relevant = [
        spec for spec in specs
        if isinstance(spec, Mapping) and str(spec.get("defense")) == "rtc_v3"
    ]
    seeds = tuple(sorted({int(spec["seed"]) for spec in relevant}))
    attacks = tuple(sorted({str(spec["attack"]) for spec in relevant}))
    if seeds != tuple(sorted(DEVELOPMENT_SEEDS)):
        raise ValueError(f"development runs must use exactly seeds {DEVELOPMENT_SEEDS}")
    if set(seeds) & set(EVALUATION_SEEDS):
        raise ValueError("held-out seeds appear in development artifacts")
    if set(attacks) != set(REQUIRED_DEVELOPMENT_ATTACKS):
        raise ValueError(
            "development runs must contain exactly targeted and all-reverse attacks"
        )
    candidate_hashes = {
        str((spec.get("custom_params") or {}).get("calibration_manifest_hash", ""))
        for spec in relevant
    }
    # Older runner specs bind the manifest by path; round files always expose
    # the loaded content hash, which is authoritative below.
    candidate_hashes.discard("")
    if candidate_hashes and candidate_hashes != {manifest.hash}:
        raise ValueError("development spec references another candidate hash")

    round_sources: list[dict[str, Any]] = []
    violations: list[float] = []
    updates: list[float] = []
    weight_sums: list[float] = []
    observed_hashes: set[str] = set()
    observed_pairs: set[tuple[str, int]] = set()
    round_paths = sorted(
        path for root in roots for path in (root / "rounds").glob("*.csv")
    )
    for path in round_paths:
        frame = pd.read_csv(path)
        if "fit_rtc_v3_calibration_hash" not in frame:
            continue
        hashes = set(frame["fit_rtc_v3_calibration_hash"].dropna().astype(str))
        if not hashes:
            continue
        observed_hashes.update(hashes)
        positive = frame[pd.to_numeric(frame["round"], errors="coerce") > 0]
        if positive.empty:
            raise ValueError(f"development round file has no trained rounds: {path}")
        for spec in relevant:
            if str(spec.get("attack")) in path.name and f"__s{int(spec['seed'])}__" in path.name:
                observed_pairs.add((str(spec["attack"]), int(spec["seed"])))
                break
        violation = pd.to_numeric(
            positive.get("fit_rtc_v3_max_constraint_violation"), errors="coerce"
        )
        update = pd.to_numeric(
            positive.get("fit_rtc_v3_cone_update_applied_count"), errors="coerce"
        )
        weight_sum = pd.to_numeric(
            positive.get("fit_rtc_v3_weight_sum"), errors="coerce"
        )
        if violation.isna().any() or update.isna().any() or weight_sum.isna().any():
            raise ValueError(f"development safety metrics are incomplete: {path}")
        violations.extend(violation.astype(float).tolist())
        updates.extend(update.astype(float).tolist())
        weight_sums.extend(weight_sum.astype(float).tolist())
        round_sources.append(
            {"path": str(path), "sha256": _sha256(path), "rounds": int(len(positive))}
        )
    expected_pairs = {
        (attack, seed)
        for attack in REQUIRED_DEVELOPMENT_ATTACKS
        for seed in DEVELOPMENT_SEEDS
    }
    complete = observed_pairs == expected_pairs
    hash_gate = observed_hashes == {manifest.hash}
    violation_max = float(max(violations, default=np.inf))
    update_max = float(max(updates, default=np.inf))
    zero_update_rounds = int(sum(value <= 1e-12 for value in weight_sums))
    all_gates = bool(
        strict_passed
        and complete
        and hash_gate
        and violation_max <= 1e-8
        and update_max == 0.0
        and zero_update_rounds == 0
    )
    evidence = {
        "artifact_kind": "rtc_v3_semantic_development_freeze_evidence_v1",
        "candidate_hash": manifest.hash,
        "seeds": list(seeds),
        "attacks": list(attacks),
        "decision": "freeze_unchanged" if all_gates else "reject",
        "all_strict_gates_passed": strict_passed,
        "all_development_gates_passed": all_gates,
        "rationale": (
            "All registered development-only strict, completeness, manifest-hash, "
            "constraint, and frozen-prototype gates passed; parameters are frozen "
            "unchanged before held-out evaluation."
            if all_gates
            else "At least one development gate failed; candidate must not be frozen."
        ),
        "gates": {
            "strict": strict_rows,
            "matrix_complete": complete,
            "candidate_hash_match": hash_gate,
            "max_constraint_violation": violation_max,
            "prototype_update_max": update_max,
            "zero_update_rounds": zero_update_rounds,
        },
        "sources": {
            "development_roots": [str(root) for root in roots],
            "experiment_manifests": manifest_sources,
            "comparison_summaries": summary_sources,
            "round_files": round_sources,
        },
    }
    output = Path(destination).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not all_gates:
        raise RuntimeError(f"development gates failed: {output}")
    return output


def freeze_unchanged_candidate(
    *,
    source_manifest: str | Path,
    development_evidence: str | Path,
    destination: str | Path,
) -> Path:
    source = CalibrationManifest.load(source_manifest)
    evidence_path = Path(development_evidence).resolve()
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if not isinstance(evidence, Mapping):
        raise ValueError("development evidence must be a mapping")
    seeds = tuple(sorted(int(value) for value in evidence.get("seeds", ())))
    if seeds != tuple(sorted(DEVELOPMENT_SEEDS)):
        raise ValueError(
            f"development evidence must use exactly seeds {DEVELOPMENT_SEEDS}"
        )
    if set(seeds) & set(EVALUATION_SEEDS) or 42 in seeds:
        raise ValueError("held-out seeds cannot participate in development")
    if str(evidence.get("candidate_hash")) != source.hash:
        raise ValueError("development evidence was produced for another candidate")
    if str(evidence.get("decision")) != "freeze_unchanged":
        raise ValueError(
            "only freeze_unchanged is accepted; retuned parameters require a new "
            "candidate and a complete development rerun"
        )
    if not bool(evidence.get("all_strict_gates_passed", False)) or not bool(
        evidence.get("all_development_gates_passed", False)
    ):
        raise ValueError("development gates did not all pass")
    rationale = str(evidence.get("rationale", "")).strip()
    if not rationale:
        raise ValueError("development freeze requires a recorded rationale")
    payload: dict[str, Any] = dict(source.payload)
    payload.pop("content_hash", None)
    metadata = dict(payload.get("metadata") or {})
    metadata["development_provenance"] = {
        "seeds": list(seeds),
        "decision": "freeze_unchanged",
        "rationale": rationale,
        "evidence_path": str(evidence_path),
        "evidence_sha256": _sha256(evidence_path),
        "predevelopment_candidate_hash": source.hash,
        "heldout_seeds_used_for_tuning": [],
    }
    metadata["formal_evaluation_ready"] = True
    metadata["promotion_ready"] = False
    payload["metadata"] = metadata
    payload["content_hash"] = content_hash(payload)
    CalibrationManifest.load(payload)
    output = Path(destination).resolve()
    if output.exists():
        existing = CalibrationManifest.load(output)
        if existing.hash != payload["content_hash"]:
            raise FileExistsError(
                f"refusing to overwrite a different frozen candidate: {output}"
            )
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    _write_freeze_sidecars(
        output=output,
        manifest=CalibrationManifest.load(output),
        evidence_path=evidence_path,
        evidence=evidence,
        source_manifest=Path(source_manifest).resolve(),
    )
    return output


def _write_freeze_sidecars(
    *,
    output: Path,
    manifest: CalibrationManifest,
    evidence_path: Path,
    evidence: Mapping[str, Any],
    source_manifest: Path,
) -> None:
    """Write hash-bound, non-promotion validation and provenance artifacts."""
    metadata = manifest.payload.get("metadata") or {}
    validation = {
        "artifact_kind": "rtc_v3_semantic_frozen_candidate_validation_v1",
        "passed": bool(
            metadata.get("formal_evaluation_ready", False)
            and not metadata.get("promotion_ready", False)
            and evidence.get("all_strict_gates_passed", False)
            and evidence.get("all_development_gates_passed", False)
            and str(evidence.get("decision")) == "freeze_unchanged"
            and str(evidence.get("candidate_hash"))
            == str(metadata.get("development_provenance", {}).get(
                "predevelopment_candidate_hash", ""
            ))
        ),
        "content_hash": manifest.hash,
        "formal_evaluation_ready": bool(
            metadata.get("formal_evaluation_ready", False)
        ),
        "promotion_ready": bool(metadata.get("promotion_ready", False)),
        "development_seeds": list(evidence.get("seeds", ())),
        "development_attacks": list(evidence.get("attacks", ())),
        "all_strict_gates_passed": bool(
            evidence.get("all_strict_gates_passed", False)
        ),
        "all_development_gates_passed": bool(
            evidence.get("all_development_gates_passed", False)
        ),
        "zero_update_rounds": int(
            (evidence.get("gates") or {}).get("zero_update_rounds", -1)
        ),
        "prototype_update_max": float(
            (evidence.get("gates") or {}).get("prototype_update_max", float("nan"))
        ),
        "max_constraint_violation": float(
            (evidence.get("gates") or {}).get(
                "max_constraint_violation", float("nan")
            )
        ),
        "note": (
            "Development freeze validation only; this is not a promotion "
            "attestation and does not establish held-out readiness."
        ),
    }
    if not validation["passed"]:
        raise ValueError("frozen candidate sidecar validation failed")
    provenance = {
        "artifact_kind": "rtc_v3_semantic_frozen_candidate_provenance_v1",
        "content_hash": manifest.hash,
        "candidate_path": str(output),
        "candidate_sha256": _sha256(output),
        "source_manifest_path": str(source_manifest),
        "source_manifest_sha256": _sha256(source_manifest),
        "development_evidence_path": str(evidence_path),
        "development_evidence_sha256": _sha256(evidence_path),
        "decision": "freeze_unchanged",
        "promotion_attestation_generated": False,
    }
    validation_path = output.with_suffix(".validation.json")
    provenance_path = output.with_suffix(".provenance.json")
    validation_path.write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    provenance_path.write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument(
        "--development-root",
        action="append",
        help="development result root; repeat once per seed/root",
    )
    parser.add_argument("--development-evidence")
    parser.add_argument(
        "--evidence-output",
        default="logs/rtc_v3_semantic_development_evidence.json",
    )
    parser.add_argument(
        "--destination",
        default="config/rtc_v3_manifest_formal_iid_semantic_candidate.json",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    evidence = args.development_evidence
    if args.development_root:
        evidence = str(
            build_development_evidence(
                source_manifest=args.source_manifest,
                development_root=args.development_root,
                destination=args.evidence_output,
            )
        )
    if not evidence:
        raise ValueError("provide --development-root or --development-evidence")
    path = freeze_unchanged_candidate(
        source_manifest=args.source_manifest,
        development_evidence=evidence,
        destination=args.destination,
    )
    print(path)
    return path


if __name__ == "__main__":
    main()
