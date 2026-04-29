from __future__ import annotations

import argparse
import json
from pathlib import Path

from detector.detector import CloneDetector, DetectionConfig
from layering.layering import CloneLayerer
from model_eval.model_eval import CloneModelEvaluator, ModelEvalConfig
from reports import HtmlRenderer, MarkdownReporter


def ensure_repo_path(repo_path: str) -> Path:
    path = Path(repo_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"repo_path does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"repo_path is not a directory: {path}")
    return path


def load_api_keys(config_path: Path) -> dict:
    if not config_path.exists():
        return {}

    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid API config JSON: {config_path} -> {e}") from e

    if not isinstance(data, dict):
        raise ValueError(f"API config must be a JSON object: {config_path}")
    return data


def first_non_empty(*values: str) -> str:
    for value in values:
        if value:
            return value
    return ""


def resolve_option(cli_value: str, config_value: str, default_value: str) -> str:
    if cli_value != default_value:
        return cli_value
    return first_non_empty(config_value, default_value)


def resolve_int_option(cli_value: int, config_value: object, default_value: int) -> int:
    if cli_value != default_value:
        return cli_value
    try:
        return int(config_value)
    except Exception:
        return default_value


def resolve_float_option(cli_value: float, config_value: object, default_value: float) -> float:
    if cli_value != default_value:
        return cli_value
    try:
        return float(config_value)
    except Exception:
        return default_value


def build_report_paths(repo_path: Path, work_dir: str, out_md: str, out_html: str) -> tuple[Path, Path]:
    module_name = repo_path.name
    report_dir = Path(work_dir).expanduser().resolve() / module_name

    if out_md.strip():
        md_path = Path(out_md).expanduser().resolve()
    else:
        md_path = report_dir / f"{module_name}_clone_report.md"

    if out_html.strip():
        html_path = Path(out_html).expanduser().resolve()
    else:
        html_path = report_dir / f"{module_name}_clone_report.html"

    return md_path, html_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clone detection agent (MVP)")
    parser.add_argument("--repo", dest="repo_path", required=True, help="Path to the target repository")
    parser.add_argument(
        "--api-config",
        dest="api_config",
        default="config/api-keys.json",
        help="Local JSON config path for sensitive API keys (default: config/api-keys.json)",
    )

    parser.add_argument(
        "--detector",
        dest="detector_mode",
        default="static",
        choices=["static"],
        help="Detector mode: static (slice + detect + merge)",
    )
    parser.add_argument(
        "--work-dir",
        dest="work_dir",
        default="data/clone_detection",
        help="Work directory for detector intermediate outputs (default: data/clone_detection)",
    )
    parser.add_argument(
        "--src-subdir",
        dest="src_subdir",
        default="",
        help="Optional subdir under repo to analyze (e.g. src, framework, etc.)",
    )
    parser.add_argument(
        "--project-name",
        dest="project_name",
        default="gme",
        help="Project name passed to static detector (default: gme)",
    )
    parser.add_argument(
        "--enable-type34",
        dest="enable_type34",
        action="store_true",
        help="Enable embedding-based Type3-4 detection and merge it into the main candidate set",
    )
    parser.add_argument(
        "--type34-api-url",
        dest="type34_api_url",
        default="",
        help="Embedding API base URL for Type3-4 detection",
    )
    parser.add_argument(
        "--type34-model-name",
        dest="type34_model_name",
        default="",
        help="Embedding model name for Type3-4 detection",
    )
    parser.add_argument(
        "--type34-api-key",
        dest="type34_api_key",
        default="",
        help="Embedding API key for Type3-4 detection",
    )
    parser.add_argument(
        "--type34-threshold",
        dest="type34_threshold",
        type=float,
        default=0.80,
        help="Similarity threshold for Type3-4 detection (default: 0.80)",
    )
    parser.add_argument(
        "--type34-batch-size",
        dest="type34_batch_size",
        type=int,
        default=8,
        help="Embedding request batch size for Type3-4 detection (default: 8)",
    )

    parser.add_argument(
        "--out-md",
        dest="out_md",
        default="",
        help="Output Markdown report path (default: data/clone_detection/<module>/<module>_clone_report.md)",
    )
    parser.add_argument(
        "--out-html",
        dest="out_html",
        default="",
        help="Output HTML report path (default: data/clone_detection/<module>/<module>_clone_report.html)",
    )

    parser.add_argument(
        "--model-eval",
        dest="model_eval_mode",
        default="openai",
        choices=["openai"],
        help="Model evaluation mode: openai-compatible chat API",
    )
    parser.add_argument(
        "--model-eval-model-name",
        dest="model_eval_model_name",
        default="",
        help="Model name for evaluation, e.g. gpt-4.1-mini or your OpenAI-compatible deployment name",
    )
    parser.add_argument(
        "--model-eval-api-url",
        dest="model_eval_api_url",
        default="",
        help="Base URL for model evaluation API, e.g. https://api.openai.com or your compatible gateway",
    )
    parser.add_argument(
        "--model-eval-api-key",
        dest="model_eval_api_key",
        default="",
        help="API key for model evaluation API",
    )
    parser.add_argument(
        "--model-eval-temperature",
        dest="model_eval_temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for model evaluation requests",
    )
    parser.add_argument(
        "--model-eval-timeout",
        dest="model_eval_timeout",
        type=int,
        default=120,
        help="Timeout in seconds for each model evaluation request",
    )
    parser.add_argument(
        "--model-eval-max-body-chars",
        dest="model_eval_max_body_chars",
        type=int,
        default=4000,
        help="Maximum number of characters from each function body sent to the model",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("[1/5] Validating input path...")
    repo_path = ensure_repo_path(args.repo_path)
    api_config = load_api_keys(Path(args.api_config).expanduser().resolve())
    clone_config = api_config.get("clone_detection") or {}
    model_eval_config = api_config.get("model_evaluation") or {}
    clone_api_key = first_non_empty(
        args.type34_api_key,
        str(clone_config.get("api_key") or ""),
    )
    type34_api_url = first_non_empty(
        args.type34_api_url,
        str(clone_config.get("api_url") or ""),
    )
    type34_model_name = first_non_empty(
        args.type34_model_name,
        str(clone_config.get("model_name") or ""),
    )
    model_eval_mode = resolve_option(
        args.model_eval_mode,
        str(model_eval_config.get("mode") or ""),
        "openai",
    )
    model_eval_model_name = resolve_option(
        args.model_eval_model_name,
        str(model_eval_config.get("model_name") or model_eval_config.get("model") or ""),
        "",
    )
    model_eval_api_url = first_non_empty(
        args.model_eval_api_url,
        str(model_eval_config.get("api_url") or model_eval_config.get("apiBase") or ""),
    )
    model_eval_api_key = first_non_empty(
        args.model_eval_api_key,
        str(model_eval_config.get("api_key") or model_eval_config.get("apiKey") or ""),
    )
    model_eval_temperature = resolve_float_option(
        args.model_eval_temperature,
        model_eval_config.get("temperature"),
        0.0,
    )
    model_eval_timeout = resolve_int_option(
        args.model_eval_timeout,
        model_eval_config.get("timeout_seconds") or model_eval_config.get("timeout"),
        120,
    )
    model_eval_max_body_chars = resolve_int_option(
        args.model_eval_max_body_chars,
        model_eval_config.get("max_body_chars"),
        4000,
    )

    print("[2/5] Running clone detector...")
    detector = CloneDetector(
        DetectionConfig(
            mode=args.detector_mode,
            work_dir=args.work_dir,
            src_subdir=args.src_subdir,
            project_name=args.project_name,
            enable_type34=args.enable_type34,
            type34_api_url=type34_api_url,
            type34_model_name=type34_model_name,
            type34_api_key=clone_api_key,
            type34_threshold=args.type34_threshold,
            type34_batch_size=args.type34_batch_size,
        )
    )
    candidates = detector.detect(repo_path)
    print(f"  - Detector produced {len(candidates)} candidates")

    print("[3/5] Layering candidates...")
    layerer = CloneLayerer()
    layered = layerer.layer(candidates)
    print(f"  - Layered items: {len(layered)}")

    print("[4/5] Model evaluation...")
    model_evaluator = CloneModelEvaluator(
        ModelEvalConfig(
            mode=model_eval_mode,
            model_name=model_eval_model_name,
            api_url=model_eval_api_url,
            api_key=model_eval_api_key,
            temperature=model_eval_temperature,
            timeout_seconds=model_eval_timeout,
            max_body_chars=model_eval_max_body_chars,
        )
    )
    evaluated = model_evaluator.evaluate(layered)
    print(f"  - Model evaluated items: {len(evaluated)}")

    print("[5/5] Writing reports...")
    md_reporter = MarkdownReporter()
    html_renderer = HtmlRenderer()

    out_md, out_html = build_report_paths(
        repo_path=repo_path,
        work_dir=args.work_dir,
        out_md=args.out_md,
        out_html=args.out_html,
    )

    md_reporter.generate(repo_path=repo_path, evaluated=evaluated, output_path=out_md)
    html_renderer.render(repo_path=repo_path, items=evaluated, output_path=out_html)

    print(f"Done. Markdown: {out_md}")
    print(f"Done. HTML: {out_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
