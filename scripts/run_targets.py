from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable

AGENT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run clone detection for one or more module directories")
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Target repository root. Relative paths are resolved from the current working directory (default: .)",
    )
    parser.add_argument(
        "--targets-file",
        default="config/scan-targets.json",
        help="JSON config containing {\"targets\": [\"module/foo\", ...]}. Relative paths are resolved from the agent root.",
    )
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        help="Single target path to scan. Can be passed multiple times.",
    )
    parser.add_argument(
        "--detector",
        default="static",
        choices=["static"],
        help="Detector mode forwarded to main.py",
    )
    parser.add_argument(
        "--work-dir",
        default="data/clone_detection",
        help="Directory for detector outputs and per-module reports. Relative paths are resolved from the agent root.",
    )
    parser.add_argument(
        "--api-config",
        default="config/api-keys.json",
        help="API config path forwarded to main.py. Relative paths are resolved from the agent root.",
    )
    parser.add_argument(
        "--enable-type34",
        action="store_true",
        help="Forward --enable-type34 to main.py",
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


def resolve_agent_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (AGENT_ROOT / path).resolve()


def run_target(
    agent_root: Path,
    target_repo_root: Path,
    target: str,
    detector: str,
    work_dir: Path,
    api_config: Path,
    enable_type34: bool,
) -> int:
    cmd = [
        sys.executable,
        str(agent_root / "main.py"),
        "--repo",
        str((target_repo_root / target).resolve()),
        "--detector",
        detector,
        "--work-dir",
        str(work_dir),
        "--api-config",
        str(api_config),
    ]
    if enable_type34:
        cmd.append("--enable-type34")

    print(f"[runner] scanning target: {target_repo_root / target}")
    result = subprocess.run(cmd, cwd=str(target_repo_root))
    return result.returncode


def main() -> int:
    args = parse_args()
    target_repo_root = Path(args.repo_root).expanduser().resolve()
    if not target_repo_root.exists() or not target_repo_root.is_dir():
        print(f"[runner] repo root does not exist or is not a directory: {target_repo_root}")
        return 1

    work_dir = resolve_agent_path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    api_config = resolve_agent_path(args.api_config)

    file_targets = load_targets(resolve_agent_path(args.targets_file))
    targets = normalize_targets(args.target, file_targets)
    if not targets:
        print("[runner] no targets configured")
        return 1

    failed: list[str] = []
    for target in targets:
        rc = run_target(
            agent_root=AGENT_ROOT,
            target_repo_root=target_repo_root,
            target=target,
            detector=args.detector,
            work_dir=work_dir,
            api_config=api_config,
            enable_type34=args.enable_type34,
        )
        if rc != 0:
            failed.append(target)

    if failed:
        print(f"[runner] failed targets: {', '.join(failed)}")
        return 1

    print(f"[runner] completed {len(targets)} target(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
