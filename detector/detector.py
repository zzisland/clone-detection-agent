from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from models.data_model import CloneCandidate, CodeLocation
from utils.file_utils import iter_source_files, to_posix_relative


@dataclass(frozen=True)
class DetectionConfig:
    mode: str = "mock"  # mock | static
    work_dir: str = "data/clone_detection"
    project_name: str = "gme"
    src_subdir: str = ""  # optional: analyze repo_path/src_subdir
    enable_type34: bool = False
    type34_api_url: str = ""
    type34_model_name: str = ""
    type34_api_key: str = ""
    type34_threshold: float = 0.80
    type34_batch_size: int = 8


class CloneDetector:
    """Detector wrapper.

    - mock: always runnable, returns a small set of synthetic candidates
    - static: runs local static algorithms (slice + type12 + optional type34 + merge)

    This matches the diagram's steps 3/4/5 as a single encapsulated detector.
    """

    def __init__(self, config: Optional[DetectionConfig] = None):
        self.config = config or DetectionConfig()

    def detect(self, repo_path: Path) -> List[CloneCandidate]:
        if self.config.mode == "mock":
            return self._mock_detect(repo_path)
        if self.config.mode == "static":
            return self._static_detect(repo_path)
        raise ValueError(f"Unsupported detector mode: {self.config.mode}")

    def _static_detect(self, repo_path: Path) -> List[CloneCandidate]:
        from detector.detection_tools.clone_detection.merge import merge_function_results
        from detector.detection_tools.clone_detection.slice import slice_to_csv
        from detector.detection_tools.clone_detection.type12_pipeline import run_type12_pipeline
        from detector.detection_tools.clone_detection.type34_pipeline import run_type34_pipeline

        src_dir = repo_path
        if self.config.src_subdir:
            src_dir = (repo_path / self.config.src_subdir).resolve()

        work_dir = Path(self.config.work_dir).expanduser().resolve()
        out_dir = work_dir / repo_path.name
        out_dir.mkdir(parents=True, exist_ok=True)

        functions_csv = out_dir / "functions.csv"
        structs_csv = out_dir / "structs.csv"
        type12_csv = out_dir / "func_clone_type12.csv"
        type34_csv = out_dir / "func_clone_type34.csv"
        merged_csv = out_dir / "func_clone_merged.csv"

        include_args = self._build_auto_include_args(repo_path, src_dir)
        if include_args:
            print(f"  - [static] auto includes: {' '.join(include_args)}")

        print(f"  - [static] slicing: {src_dir}")
        slice_to_csv(
            str(src_dir),
            func_csv_path=str(functions_csv),
            struct_csv_path=str(structs_csv),
            compile_args=include_args,
        )

        print("  - [static] running Type1-2 pipeline...")
        run_type12_pipeline(
            input_csv=functions_csv,
            output_csv=type12_csv,
            project_name=self.config.project_name,
        )

        if self.config.enable_type34:
            if not self.config.type34_api_url or not self.config.type34_model_name:
                raise ValueError("Type3-4 detection enabled, but api_url/model_name is missing")
            print("  - [static] running Type3-4 pipeline...")
            run_type34_pipeline(
                input_csv=functions_csv,
                output_csv=type34_csv,
                api_url=self.config.type34_api_url,
                model_name=self.config.type34_model_name,
                api_key=self.config.type34_api_key,
                threshold=self.config.type34_threshold,
                batch_size=self.config.type34_batch_size,
            )
        else:
            self._ensure_empty_type34_csv(type34_csv)

        print("  - [static] merging candidate sets...")
        merge_function_results(
            type12_csv=type12_csv,
            type34_csv=type34_csv,
            output_csv=merged_csv,
        )

        return self._load_merged_csv_as_candidates(merged_csv)

    @staticmethod
    def _build_auto_include_args(repo_path: Path, src_dir: Path) -> List[str]:
        include_args: List[str] = []
        seen: set[str] = set()

        repo_path = repo_path.resolve()
        src_dir = src_dir.resolve()

        candidates = [
            repo_path.parent / "include",
            repo_path.parent / "include" / "module" / repo_path.name / "include",
            src_dir.parent / "include",
            src_dir / "include",
        ]

        for candidate in candidates:
            candidate_str = str(candidate.resolve())
            if candidate.exists() and candidate.is_dir() and candidate_str not in seen:
                include_args.append(f"-I{candidate_str}")
                seen.add(candidate_str)

        return include_args

    def _load_merged_csv_as_candidates(self, merged_csv: Path) -> List[CloneCandidate]:
        if not merged_csv.exists() or merged_csv.stat().st_size == 0:
            return []

        candidates: List[CloneCandidate] = []
        with merged_csv.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sim = self._safe_float(row.get("similarity"))
                file1 = str(row.get("file1") or "").replace("\\", "/")
                file2 = str(row.get("file2") or "").replace("\\", "/")

                lines1 = str(row.get("lines1") or "").strip()
                lines2 = str(row.get("lines2") or "").strip()
                s1, e1 = self._parse_line_range(lines1)
                s2, e2 = self._parse_line_range(lines2)

                source_method = self._normalize_source_method(str(row.get("detection_method") or "static:unknown"))

                extra = {
                    "pair_id": str(row.get("pair_id") or ""),
                    "code_type": str(row.get("code_type") or "function"),
                    "clone_type_label": str(row.get("Clone_Type_Label") or ""),
                    "from_type12": self._bool_str("Func Clone (Type1-2)" in str(row.get("detection_method") or "")),
                    "from_type34": self._bool_str("Func Clone (Embedding)" in str(row.get("detection_method") or "")),
                    "type12_similarity": str(row.get("type12_combined_similarity") or ""),
                    "type34_similarity": str(row.get("type34_similarity") or ""),
                    "type12_clone_type": str(row.get("type12_clone_type") or ""),
                }

                candidates.append(
                    CloneCandidate(
                        left=CodeLocation(file_path=file1, start_line=s1, end_line=e1),
                        right=CodeLocation(file_path=file2, start_line=s2, end_line=e2),
                        similarity=sim,
                        source_method=source_method,
                        extra=extra,
                    )
                )

        return candidates

    @staticmethod
    def _ensure_empty_type34_csv(type34_csv: Path) -> None:
        type34_csv.parent.mkdir(parents=True, exist_ok=True)
        with type34_csv.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "pair_id",
                    "function_id_1",
                    "file1",
                    "func1_name",
                    "lines1",
                    "func1_body",
                    "function_id_2",
                    "file2",
                    "func2_name",
                    "lines2",
                    "func2_body",
                    "similarity",
                    "Clone_Type_Label",
                ],
            )
            writer.writeheader()

    @staticmethod
    def _normalize_source_method(value: str) -> str:
        v = value.strip()
        return v if v else "static:merged"

    @staticmethod
    def _safe_float(value: object) -> float:
        try:
            return float(value or 0)
        except Exception:
            return 0.0

    @staticmethod
    def _bool_str(flag: bool) -> str:
        return "true" if flag else "false"

    @staticmethod
    def _parse_line_range(value: str) -> tuple[int, int]:
        v = value.strip()
        if not v:
            return 0, 0
        for sep in ("-", "~", ":"):
            if sep in v:
                left, right = v.split(sep, 1)
                try:
                    return int(left), int(right)
                except Exception:
                    return 0, 0
        try:
            n = int(v)
            return n, n
        except Exception:
            return 0, 0

    def _mock_detect(self, repo_path: Path) -> List[CloneCandidate]:
        files = iter_source_files(repo_path)

        if len(files) >= 2:
            f1, f2 = files[0], files[1]
        elif len(files) == 1:
            f1, f2 = files[0], files[0]
        else:
            f1 = repo_path / "src" / "example1.c"
            f2 = repo_path / "src" / "example2.c"

        left = CodeLocation(
            file_path=to_posix_relative(f1, repo_path),
            start_line=10,
            end_line=40,
        )
        right = CodeLocation(
            file_path=to_posix_relative(f2, repo_path),
            start_line=12,
            end_line=42,
        )

        return [
            CloneCandidate(
                left=left,
                right=right,
                similarity=0.86,
                source_method="mock:token_similarity",
                extra={
                    "note": "mock candidate based on first two source files",
                    "from_type12": "true",
                    "from_type34": "false",
                },
            )
        ]
