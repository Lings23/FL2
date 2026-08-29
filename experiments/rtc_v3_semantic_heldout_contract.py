"""Validate extra held-out matrices before any expensive execution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from defenses.rtc.calibration import CalibrationManifest


EXPECTED_SEEDS = {46, 47, 48, 51}
EXPECTED_ATTACKS = {"label_flip_targeted", "label_flip_all_reverse"}


def _load_roots(roots: Sequence[str | Path]) -> pd.DataFrame:
    frames = []
    for raw in roots:
        root = Path(raw).resolve()
        path = root / "experiment_matrix.csv"
        frame = pd.read_csv(path)
        frame["source_root"] = str(root)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def validate_heldout_contract(
    *,
    candidate_roots: Sequence[str | Path],
    baseline_roots: Sequence[str | Path],
    candidate_manifest: str | Path,
    baseline_manifest: str | Path,
) -> dict[str, Any]:
    if len(candidate_roots) != 4 or len(baseline_roots) != 4:
        raise ValueError("exactly four candidate and four baseline roots are required")
    candidate = _load_roots(candidate_roots)
    baseline = _load_roots(baseline_roots)
    keys = ["seed", "attack", "trial_plan_hash"]
    candidate_rtc = candidate[candidate["defense"].astype(str) == "rtc_v3"]
    candidate_clip = candidate[candidate["defense"].astype(str) == "clip_only"]
    baseline_rtc = baseline[baseline["defense"].astype(str) == "rtc_v3"]
    expected_cells = {(seed, attack) for seed in EXPECTED_SEEDS for attack in EXPECTED_ATTACKS}

    def cells(frame: pd.DataFrame) -> set[tuple[int, str]]:
        return {
            (int(row["seed"]), str(row["attack"]))
            for _, row in frame.iterrows()
        }

    joined = candidate_rtc.merge(
        candidate_clip[keys], on=keys, validate="one_to_one"
    ).merge(baseline_rtc[keys], on=keys, validate="one_to_one")
    candidate_expected_path = str(Path(candidate_manifest).resolve())
    baseline_expected_path = str(Path(baseline_manifest).resolve())

    def calibration_paths(frame: pd.DataFrame) -> set[str]:
        paths: set[str] = set()
        for raw in frame["custom_params"]:
            custom = json.loads(raw)
            paths.add(str(custom.get("calibration_path", "")))
        return paths

    gates = {
        "pre_registered_seed_set": set(map(int, candidate["seed"])) == EXPECTED_SEEDS
        and set(map(int, baseline["seed"])) == EXPECTED_SEEDS,
        "candidate_cells_complete": cells(candidate_rtc) == expected_cells
        and cells(candidate_clip) == expected_cells,
        "baseline_cells_complete": cells(baseline_rtc) == expected_cells,
        "strict_plan_join_complete": len(joined) == len(expected_cells),
        "candidate_manifest_bound": calibration_paths(candidate_rtc)
        == {candidate_expected_path},
        "baseline_manifest_bound": calibration_paths(baseline_rtc)
        == {baseline_expected_path},
        "strict_principal_uniform": all(
            frame["pairing_mode"].astype(str).eq("strict").all()
            and frame["sampling_protocol"].astype(str).eq("principal_uniform").all()
            for frame in (candidate, baseline)
        ),
    }
    return {
        "artifact_kind": "rtc_v3_semantic_extra_heldout_contract_v1",
        "passed": all(gates.values()),
        "execution_complete": False,
        "gates": gates,
        "seeds": sorted(EXPECTED_SEEDS),
        "attacks": sorted(EXPECTED_ATTACKS),
        "join_keys": joined[keys].sort_values(keys).to_dict("records"),
        "candidate_manifest_hash": CalibrationManifest.load(candidate_manifest).hash,
        "baseline_manifest_hash": CalibrationManifest.load(baseline_manifest).hash,
        "candidate_roots": [str(Path(root).resolve()) for root in candidate_roots],
        "baseline_roots": [str(Path(root).resolve()) for root in baseline_roots],
    }


def main(argv: Sequence[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", action="append", required=True)
    parser.add_argument("--baseline-root", action="append", required=True)
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--baseline-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    evidence = validate_heldout_contract(
        candidate_roots=args.candidate_root,
        baseline_roots=args.baseline_root,
        candidate_manifest=args.candidate_manifest,
        baseline_manifest=args.baseline_manifest,
    )
    destination = Path(args.output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if not evidence["passed"]:
        raise RuntimeError("extra held-out contract failed")
    return destination


if __name__ == "__main__":
    main()
