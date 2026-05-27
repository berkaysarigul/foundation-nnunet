"""Write the PR-11 protocol decision report after Dataset101 overlap analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pr11_utils import write_markdown_report, write_yaml


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write PR-11 protocol decision report.")
    parser.add_argument(
        "--overlap-summary",
        type=Path,
        default=Path("artifacts/siim_original/audits/dataset101_overlap/overlap_summary.yaml"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/siim_original/audits/protocol_decision"),
    )
    return parser.parse_args(argv)


def _load_yaml_or_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        import yaml  # type: ignore

        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}


def write_report(out: Path, overlap_summary: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "recommended_primary_protocol": "Protocol B - SIIM-stage protocol",
        "secondary_protocol": "Protocol A - Dataset101-heldout-compatible bridge",
        "decision": (
            "Implement Protocol B first for SIIM-original performance. Keep Protocol A "
            "as a controlled bridge to compare against prior Dataset101 heldout results."
        ),
        "protocol_b": {
            "train_validation": "SIIM stage_1_train only",
            "test": "SIIM stage_1_test masks from the PNG repackage",
            "primary_success": "image+Foundation X prior beats image-only and Foundation X direct on SIIM-stage evaluation",
        },
        "protocol_a": {
            "train_validation": "SIIM training cases excluding Dataset101 heldout overlap",
            "test": "Dataset101 heldout",
            "purpose": "bridge to previous Foundation X direct and PR-10C Dataset101 numbers",
        },
        "overlap_summary": overlap_summary,
    }
    out.mkdir(parents=True, exist_ok=True)
    write_yaml(out / "protocol_decision_summary.yaml", payload)
    body = [
        f"- recommendation: `{payload['recommended_primary_protocol']}`",
        f"- secondary bridge: `{payload['secondary_protocol']}`",
        f"- overlap rows: `{overlap_summary.get('overlap_rows', 'unknown')}`",
        f"- Dataset101 heldout overlap cases: `{overlap_summary.get('dataset101_heldout_overlap_cases', 'unknown')}`",
        f"- Protocol A allowed SIIM train cases: `{overlap_summary.get('siim_train_allowed_cases', 'unknown')}`",
        "",
        payload["decision"],
    ]
    write_markdown_report(out / "protocol_decision_report.md", "PR-11 Protocol Decision", [("Decision", body)])
    return payload


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = write_report(args.out, _load_yaml_or_json(args.overlap_summary))
    print(f"[done] protocol decision written under {args.out}: {payload['recommended_primary_protocol']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

