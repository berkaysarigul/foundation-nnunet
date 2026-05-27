"""Generate a compact PR-11 final report from produced artifact summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pr11_utils import write_json, write_markdown_report


DEFAULT_SUMMARIES = [
    "data/extracted/extraction_summary.yaml",
    "artifacts/siim_original/manifests/siim_acr_png_manifest_summary.yaml",
    "artifacts/siim_original/audits/manifest_audit/summary.yaml",
    "artifacts/siim_original/audits/dataset101_overlap/overlap_summary.yaml",
    "artifacts/siim_original/audits/protocol_decision/protocol_decision_summary.yaml",
    "artifacts/ptx498/manifests/ptx498_manifest_summary.yaml",
    "artifacts/ptx498/audits/manifest_audit/summary.yaml",
    "artifacts/siim_original/fx_priors/teacher_head5_official224/summary.yaml",
    "artifacts/ptx498/fx_priors/teacher_head5_official224/summary.yaml",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate PR-11 final report.")
    parser.add_argument("--out", type=Path, default=Path("artifacts/nnunet_v2/reports/pr11_final"))
    parser.add_argument("--summary", action="append", default=[], help="Additional summary YAML/JSON path.")
    return parser.parse_args(argv)


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "exists": False}
    try:
        import yaml  # type: ignore

        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        payload = data if isinstance(data, dict) else {"value": data}
    except Exception:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            payload = {"load_error": f"{type(exc).__name__}: {exc}"}
    payload["path"] = str(path)
    payload["exists"] = True
    return payload


def generate(out: Path, summaries: list[str]) -> dict[str, Any]:
    paths = [Path(p) for p in DEFAULT_SUMMARIES + summaries]
    loaded = [_load(path) for path in paths]
    payload = {
        "schema_version": 1,
        "summaries": loaded,
        "missing_summary_paths": [item["path"] for item in loaded if not item.get("exists")],
    }
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "pr11_final_report_summary.json", payload)

    lines = [
        f"- summary files inspected: `{len(loaded)}`",
        f"- missing summaries: `{len(payload['missing_summary_paths'])}`",
        "",
        "Primary success criteria:",
        "- SIIM-stage image+Foundation X prior beats SIIM image-only.",
        "- SIIM-stage image+Foundation X prior beats Foundation X direct on the same evaluation.",
        "- Dataset101 bridge remains separate from SIIM-stage claims.",
        "- PTX-498 is reported only as locked external validation.",
    ]
    if payload["missing_summary_paths"]:
        lines.extend(["", "Missing summary paths:"])
        lines.extend(f"- `{path}`" for path in payload["missing_summary_paths"])
    write_markdown_report(out / "pr11_final_report.md", "PR-11 Final Report", [("Summary", lines)])
    return payload


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = generate(args.out, list(args.summary))
    print(f"[done] final report scaffold written under {args.out}; missing={len(payload['missing_summary_paths'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

