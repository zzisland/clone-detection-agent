from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run clone detection for one or more module directories")
    parser.add_argument(
        "--targets-file",
        default="config/scan-targets.json",
        help="JSON config containing {\"targets\": [\"module/foo\", ...]}",
    )
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        help="Single target path to scan. Can be passed multiple times.",
    )
    parser.add_argument(
        "--detector",
        default="mock",
        choices=["mock", "static"],
        help="Detector mode forwarded to main.py",
    )
    parser.add_argument(
        "--reports-dir",
        default="reports",
        help="Directory for generated markdown/html reports",
    )
    return parser.parse_args()


def load_targets(targets_file: Path) -> list[str]:
    if not targets_file.exists():
        return []
    data = json.loads(targets_file.read_text(encoding="utf-8"))
    targets = data.get("targets", [])
    return [str(t).strip() for t in targets if str(t).strip()]


def normalize_targets(cli_targets: Iterable[str], file_targets: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for target in list(cli_targets) + list(file_targets):
        normalized = str(target).strip().replace("\\", "/")
        if normalized and normalized not in seen:
            seen.add(normalized)
            ordered.append(normalized)
    return ordered


def run_target(repo_root: Path, target: str, detector: str, reports_dir: Path) -> int:
    module_name = Path(target).name
    out_md = reports_dir / f"{module_name}_clone_report.md"
    out_html = reports_dir / f"{module_name}_clone_report.html"

    cmd = [
        sys.executable,
        str(repo_root / "main.py"),
        "--repo",
        str(repo_root / target),
        "--detector",
        detector,
        "--out-md",
        str(out_md),
        "--out-html",
        str(out_html),
    ]

    print(f"[runner] scanning target: {target}")
    result = subprocess.run(cmd, cwd=str(repo_root))
    return result.returncode


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    reports_dir = (repo_root / args.reports_dir).resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)

    file_targets = load_targets((repo_root / args.targets_file).resolve())
    targets = normalize_targets(args.target, file_targets)
    if not targets:
        print("[runner] no targets configured")
        return 1

    failed: list[str] = []
    for target in targets:
        rc = run_target(repo_root, target, args.detector, reports_dir)
        if rc != 0:
            failed.append(target)

    if failed:
        print(f"[runner] failed targets: {', '.join(failed)}")
        return 1

    print(f"[runner] completed {len(targets)} target(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
