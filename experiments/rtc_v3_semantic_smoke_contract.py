"""Validate the preregistered final CPU smoke matrix without running it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


EXPECTED_DEFENSES = {
    "fedavg",
    "rtc_v3",
    "semantic_observe",
    "semantic_soft",
    "semantic_temporal",
    "semantic_exposure",
}


def validate_smoke_contract(
    root: str | Path,
    *,
    candidate_manifest: str | Path,
) -> dict[str, Any]:
    output = Path(root).resolve()
    manifest_path = output / "experiment_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    specs = payload.get("specs", payload)
    plans = {
        str(spec.get("trial_plan_hash", "")) for spec in specs
    }
    defenses = {str(spec.get("defense")) for spec in specs}
    seeds = {int(spec.get("seed")) for spec in specs}
    attacks = {str(spec.get("attack")) for spec in specs}
    rounds = set()
    for spec in specs:
        plan_path = Path(str(spec["trial_plan_path"]))
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        rounds.add(len(plan["rounds"]))
    expected_manifest = str(Path(candidate_manifest).resolve())
    rtc_specs = [spec for spec in specs if spec.get("defense") != "fedavg"]
    calibration_paths = {
        str((spec.get("custom_params") or {}).get("calibration_path", ""))
        for spec in rtc_specs
    }
    gates = {
        "six_expected_defenses": defenses == EXPECTED_DEFENSES and len(specs) == 6,
        "single_trial_plan": len(plans) == 1 and "" not in plans,
        "seed52": seeds == {52},
        "targeted_attack": attacks == {"label_flip_targeted"},
        "three_rounds": rounds == {3},
        "candidate_manifest_bound": calibration_paths == {expected_manifest},
        "strict_principal_uniform": all(
            spec.get("pairing_mode") == "strict"
            and spec.get("sampling_protocol") == "principal_uniform"
            for spec in specs
        ),
    }
    return {
        "artifact_kind": "rtc_v3_semantic_final_cpu_smoke_contract_v1",
        "passed": all(gates.values()),
        "gates": gates,
        "defenses": sorted(defenses),
        "trial_plan_hashes": sorted(plans),
        "rounds": sorted(rounds),
        "candidate_manifest": expected_manifest,
        "execution_complete": (output / "execution_validation.csv").is_file(),
    }


def main(argv: Sequence[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    evidence = validate_smoke_contract(
        args.root, candidate_manifest=args.candidate_manifest
    )
    destination = Path(args.output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if not evidence["passed"]:
        raise RuntimeError("final CPU smoke contract validation failed")
    return destination


if __name__ == "__main__":
    main()

