from __future__ import annotations

import argparse
from pathlib import Path

from detector.detector import CloneDetector, DetectionConfig
from layering.layering import CloneLayerer
from model_eval.model_eval import CloneModelEvaluator
from renderer.html_renderer import HtmlRenderer
from reporter.reporter import MarkdownReporter
from utils.file_utils import ensure_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clone detection agent (MVP)")
    parser.add_argument("--repo", dest="repo_path", required=True, help="Path to the target repository")

    parser.add_argument(
        "--detector",
        dest="detector_mode",
        default="mock",
        choices=["mock", "static"],
        help="Detector mode: mock (default) or static (slice + detect + merge)",
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
        default="clone_report.md",
        help="Output Markdown report path (default: clone_report.md)",
    )
    parser.add_argument(
        "--out-html",
        dest="out_html",
        default="clone_report.html",
        help="Output HTML report path (default: clone_report.html)",
    )

    parser.add_argument(
        "--model-eval",
        dest="model_eval_mode",
        default="mock",
        choices=["mock"],
        help="Model evaluation mode (MVP only supports mock for now)",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("[1/5] Validating input path...")
    repo_path = ensure_repo_path(args.repo_path)

    print("[2/5] Running clone detector...")
    detector = CloneDetector(
        DetectionConfig(
            mode=args.detector_mode,
            work_dir=args.work_dir,
            src_subdir=args.src_subdir,
            project_name=args.project_name,
            enable_type34=args.enable_type34,
            type34_api_url=args.type34_api_url,
            type34_model_name=args.type34_model_name,
            type34_api_key=args.type34_api_key,
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
    model_evaluator = CloneModelEvaluator()
    evaluated = model_evaluator.evaluate(layered)
    print(f"  - Model evaluated items: {len(evaluated)}")

    print("[5/5] Writing reports...")
    md_reporter = MarkdownReporter()
    html_renderer = HtmlRenderer()

    out_md = Path(args.out_md).expanduser().resolve()
    out_html = Path(args.out_html).expanduser().resolve()

    md_reporter.generate(repo_path=repo_path, evaluated=evaluated, output_path=out_md)
    html_renderer.render(repo_path=repo_path, items=evaluated, output_path=out_html)

    print(f"Done. Markdown: {out_md}")
    print(f"Done. HTML: {out_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
