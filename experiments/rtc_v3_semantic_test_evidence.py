"""Create hash-bound machine-readable evidence from a pytest JUnit report."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Sequence


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_output(workspace: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout


def build_test_evidence(
    *,
    workspace: str | Path,
    junit_path: str | Path,
) -> dict[str, Any]:
    root = Path(workspace).resolve()
    junit = Path(junit_path).resolve()
    document = ET.parse(junit).getroot()
    suites = [document] if document.tag == "testsuite" else list(document.findall("testsuite"))
    totals = {
        key: sum(int(float(suite.attrib.get(key, 0))) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }
    tracked_diff = _git_output(root, "diff", "--binary", "--", ".")
    untracked = sorted(
        line.strip()
        for line in _git_output(root, "ls-files", "--others", "--exclude-standard").splitlines()
        if line.strip()
    )
    code_untracked = [
        value
        for value in untracked
        if value.endswith((".py", ".yaml", ".yml", ".json"))
        and not value.startswith(("logs/", "output/", "tmp/"))
    ]
    untracked_hashes = {
        path: _sha256(root / path)
        for path in code_untracked
        if (root / path).is_file()
    }
    source_digest = hashlib.sha256(
        (
            tracked_diff
            + json.dumps(untracked_hashes, sort_keys=True, separators=(",", ":"))
        ).encode("utf-8")
    ).hexdigest()
    return {
        "artifact_kind": "rtc_v3_semantic_pytest_evidence_v1",
        "passed": totals["tests"] > 0
        and totals["failures"] == 0
        and totals["errors"] == 0,
        **totals,
        "junit_path": str(junit),
        "junit_sha256": _sha256(junit),
        "workspace": str(root),
        "git_head": _git_output(root, "rev-parse", "HEAD").strip(),
        "source_snapshot_sha256": source_digest,
        "untracked_code_artifacts": untracked_hashes,
    }


def main(argv: Sequence[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--junit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    payload = build_test_evidence(workspace=args.workspace, junit_path=args.junit)
    destination = Path(args.output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if not payload["passed"]:
        raise RuntimeError("pytest JUnit evidence contains failures or errors")
    return destination


if __name__ == "__main__":
    main()

