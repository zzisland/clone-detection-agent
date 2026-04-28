from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Callable, Optional

from .detector_src.clone_detector import detect_clones
from .detector_src.cpp_module_obfuscator import CppModuleObfuscator
from .detector_src.module_config import ModuleConfig


def default_logger(msg: str) -> None:
    print(msg, flush=True)


def _count_rows(csv_path: Path) -> int:
    try:
        with csv_path.open("r", encoding="utf-8-sig") as f:
            return max(0, sum(1 for _ in f) - 1)
    except Exception:
        return 0


def _extract_body_from_source(file_path: str, start_line: int, end_line: int) -> str:
    if not file_path or start_line <= 0 or end_line <= 0 or end_line < start_line:
        return ""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        return "".join(lines[start_line - 1:end_line])
    except Exception:
        return ""


def _iter_functions_from_csv(input_csv: Path) -> list[dict[str, str]]:
    functions: list[dict[str, str]] = []
    with input_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            signature = str(row.get("函数签名") or "").strip()
            body = row.get("函数体") or ""
            if not body:
                body = _extract_body_from_source(
                    str(row.get("文件路径") or ""),
                    int(row.get("起始行") or 0),
                    int(row.get("结束行") or 0),
                )
            functions.append({
                "name": signature,
                "body": body if isinstance(body, str) else "",
            })
    return functions


def _obfuscate_functions_to_cpp(input_csv: Path, output_cpp: Path, logger: Callable[[str], None]) -> None:
    functions = _iter_functions_from_csv(input_csv)
    if not functions:
        output_cpp.write_text("// No functions\n", encoding="utf-8")
        return

    logger(f"[Type1-2] 正在隐秘化：{input_csv.name}")
    module_name = input_csv.stem
    config = ModuleConfig(module_name=module_name)
    obfuscator = CppModuleObfuscator(config, enable_slicing=False)
    cpp_content = obfuscator.generate(functions, enable_slicing=False, enable_analysis=False)
    output_cpp.write_text(cpp_content, encoding="utf-8")
    logger(f"[Type1-2] 隐秘化完成：{output_cpp.name}")


def run_type12_pipeline(
    input_csv: Path,
    output_csv: Path,
    project_name: str = "gme",
    header_dir: Optional[str] = None,
    logger: Callable[[str], None] = default_logger,
) -> int:
    input_csv = Path(input_csv).resolve()
    output_csv = Path(output_csv).resolve()

    if not input_csv.exists():
        raise FileNotFoundError(f"输入 CSV 不存在: {input_csv}")

    module_name = input_csv.stem
    output_parent = output_csv.parent
    work_dir = output_parent / f"work_{module_name}"
    work_dir.mkdir(parents=True, exist_ok=True)

    result_dir = work_dir / "cloneresult"
    result_dir.mkdir(exist_ok=True, parents=True)

    obf_cpp = work_dir / f"{module_name}.cpp"
    csv_report = result_dir / f"{module_name}_clones.csv"
    json_report = result_dir / f"{module_name}_clones.json"

    logger(f"[Type1-2] 工作目录: {work_dir}")
    logger(f"[Type1-2] 输入 CSV: {input_csv}")

    _obfuscate_functions_to_cpp(input_csv, obf_cpp, logger)

    detect_clones(
        source_path=obf_cpp,
        csv_path=csv_report,
        json_path=json_report,
        metadata_csv_path=input_csv,
        project_name=project_name,
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if csv_report.exists():
        if output_csv.resolve() != csv_report.resolve():
            shutil.copyfile(csv_report, output_csv)
    else:
        output_csv.touch()

    try:
        shutil.rmtree(work_dir)
    except Exception:
        pass

    count = _count_rows(output_csv)
    logger(f"[Type1-2] 克隆检测完成，结果行数: {count}")
    return count
