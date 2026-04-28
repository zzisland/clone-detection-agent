from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Callable, Iterable, Optional


DEFAULT_COMPILE_ARGS = [
    "-std=c++17",
    "-Wno-inconsistent-dllimport",
]


def _configure_libclang() -> None:
    try:
        from clang.cindex import Config  # type: ignore
    except Exception as e:
        raise RuntimeError(f"clang Python bindings are not available: {e}")

    libclang_file = os.environ.get("LIBCLANG_FILE", "").strip()
    libclang_path = os.environ.get("LIBCLANG_PATH", "").strip()

    if libclang_file:
        Config.set_library_file(libclang_file)
        return

    if libclang_path:
        p = Path(libclang_path)
        if p.is_dir():
            candidates = [
                p / "libclang.dll",
                p / "bin" / "libclang.dll",
                p / "lib" / "libclang.dll",
            ]
            for c in candidates:
                if c.exists():
                    Config.set_library_file(str(c))
                    return
            Config.set_library_path(str(p))
            return

        if p.exists():
            Config.set_library_file(str(p))
            return


def get_function_context(cursor) -> tuple[str, str]:
    import clang.cindex  # type: ignore

    func_type = "Function"
    if cursor.kind == clang.cindex.CursorKind.CXX_METHOD:
        func_type = "Method"

    parent = cursor.semantic_parent
    scope_name = "global"

    while parent:
        if parent.kind in (clang.cindex.CursorKind.CLASS_DECL, clang.cindex.CursorKind.STRUCT_DECL):
            scope_name = parent.spelling
            func_type = "Method"
            break
        if parent.kind == clang.cindex.CursorKind.NAMESPACE:
            scope_name = parent.spelling
            break
        parent = parent.semantic_parent

    return scope_name, func_type


def get_qualified_name(cursor) -> str:
    import clang.cindex  # type: ignore

    names = []
    current = cursor
    while current is not None:
        if current.kind in (
            clang.cindex.CursorKind.CLASS_DECL,
            clang.cindex.CursorKind.STRUCT_DECL,
            clang.cindex.CursorKind.NAMESPACE,
            clang.cindex.CursorKind.CLASS_TEMPLATE,
        ):
            if current.spelling:
                names.insert(0, current.spelling)
        elif current.kind in (
            clang.cindex.CursorKind.FUNCTION_DECL,
            clang.cindex.CursorKind.CXX_METHOD,
            clang.cindex.CursorKind.CONSTRUCTOR,
            clang.cindex.CursorKind.DESTRUCTOR,
            clang.cindex.CursorKind.FUNCTION_TEMPLATE,
        ):
            if current.spelling:
                names.append(current.spelling)
        current = current.semantic_parent
        if current and current.kind == clang.cindex.CursorKind.TRANSLATION_UNIT:
            break
    qualified_name = "::".join(names) if names else ""
    if qualified_name and "::" in qualified_name:
        parts = qualified_name.split("::")
        if len(parts) > 1:
            first_part = parts[0]
            if first_part.islower() and first_part in ["std", "gme", "boost", "qt"]:
                qualified_name = "::".join(parts[1:])
    return qualified_name


def get_function_signature(cursor, source_lines: list[str]) -> str:
    start_line = cursor.extent.start.line
    start_col = cursor.extent.start.column
    signature_lines: list[str] = []
    for line_num in range(start_line - 1, len(source_lines)):
        line = source_lines[line_num]
        if line_num == start_line - 1:
            line = line[start_col - 1:]
        brace_pos = line.find("{")
        if brace_pos != -1:
            signature_lines.append(line[:brace_pos].strip())
            break
        signature_lines.append(line.strip())

    full_text = " ".join(signature_lines)
    full_text = " ".join(full_text.split())

    qualified_name = get_qualified_name(cursor)
    qualified_name_pos = full_text.find(qualified_name)
    if qualified_name_pos != -1:
        return full_text[qualified_name_pos:]

    if "::" in qualified_name:
        parts = qualified_name.split("::")
        if len(parts) >= 2:
            class_name = "::".join(parts[:-1])
            func_only_name = parts[-1]
            func_name_pos = full_text.find(func_only_name)
            if func_name_pos != -1:
                prefix = full_text[:func_name_pos].strip()
                if prefix.endswith("::"):
                    class_start = full_text.rfind(class_name, 0, func_name_pos)
                    if class_start != -1:
                        return full_text[class_start:]
                    return full_text[func_name_pos:]
                func_signature = full_text[func_name_pos:]
                return f"{class_name}::{func_signature}"

    func_name = cursor.spelling
    func_name_pos = full_text.find(func_name)
    if func_name_pos != -1:
        return full_text[func_name_pos:]
    return full_text


def handle_function(cursor, abs_file_path: str, source_lines: list[str], func_writer, file_path: str) -> None:
    import clang.cindex  # type: ignore

    if not cursor.location.file:
        return
    if os.path.abspath(cursor.location.file.name) != abs_file_path:
        return
    if not cursor.is_definition():
        return

    scope_name, func_type = get_function_context(cursor)
    func_signature = get_function_signature(cursor, source_lines)

    if cursor.kind in (clang.cindex.CursorKind.CONSTRUCTOR, clang.cindex.CursorKind.DESTRUCTOR):
        return_type = "<no return type>"
    else:
        return_type = cursor.result_type.spelling

    start_line = cursor.extent.start.line
    end_line = cursor.extent.end.line
    func_body = "\n".join(source_lines[start_line - 1:end_line])

    func_writer.writerow([
        file_path,
        func_signature,
        return_type,
        start_line,
        end_line,
        func_body,
        scope_name,
        func_type,
    ])


def handle_struct(cursor, abs_file_path: str, source_lines: list[str], struct_writer, file_path: str) -> None:
    import clang.cindex  # type: ignore

    if not cursor.location.file:
        return
    if os.path.abspath(cursor.location.file.name) != abs_file_path:
        return

    struct_name = cursor.spelling
    if not struct_name:
        return

    category = "struct" if cursor.kind == clang.cindex.CursorKind.STRUCT_DECL else "class"

    start_line = cursor.extent.start.line
    end_line = cursor.extent.end.line
    if end_line < start_line:
        return

    struct_id = f"{Path(file_path).stem}_{struct_name}_{start_line}_{end_line}"
    struct_id = struct_id.replace(" ", "_").replace("::", "_")

    struct_body = "\n".join(source_lines[start_line - 1:end_line])

    member_types = []
    for child in cursor.get_children():
        if child.kind == clang.cindex.CursorKind.FIELD_DECL:
            member_types.append(child.type.spelling.replace(" ", ""))
    member_signature = ",".join(member_types)

    struct_writer.writerow([
        struct_id,
        file_path,
        struct_name,
        category,
        start_line,
        end_line,
        member_signature,
        struct_body,
    ])


def visit(cursor, abs_file_path: str, source_lines: list[str], func_writer, struct_writer, file_path: str) -> None:
    import clang.cindex  # type: ignore

    kind = cursor.kind

    if kind in (clang.cindex.CursorKind.STRUCT_DECL, clang.cindex.CursorKind.CLASS_DECL):
        handle_struct(cursor, abs_file_path, source_lines, struct_writer, file_path)

    if kind == clang.cindex.CursorKind.FUNCTION_TEMPLATE:
        for child in cursor.get_children():
            if child.kind in (clang.cindex.CursorKind.FUNCTION_DECL, clang.cindex.CursorKind.CXX_METHOD):
                handle_function(child, abs_file_path, source_lines, func_writer, file_path)
        for child in cursor.get_children():
            visit(child, abs_file_path, source_lines, func_writer, struct_writer, file_path)
        return

    if kind in (
        clang.cindex.CursorKind.FUNCTION_DECL,
        clang.cindex.CursorKind.CXX_METHOD,
        clang.cindex.CursorKind.CONSTRUCTOR,
        clang.cindex.CursorKind.DESTRUCTOR,
    ):
        handle_function(cursor, abs_file_path, source_lines, func_writer, file_path)

    for child in cursor.get_children():
        visit(child, abs_file_path, source_lines, func_writer, struct_writer, file_path)


def analyze_cpp_file(file_path: str, compile_args: list[str], func_writer, struct_writer) -> None:
    import clang.cindex  # type: ignore

    index = clang.cindex.Index.create()
    tu = index.parse(file_path, args=compile_args)

    abs_file_path = os.path.abspath(file_path)
    with open(abs_file_path, "r", encoding="utf-8", errors="ignore") as f:
        source_lines = f.readlines()

    visit(tu.cursor, abs_file_path, source_lines, func_writer, struct_writer, file_path)


def slice_to_csv(
    src_dir: str,
    func_csv_path: str,
    struct_csv_path: str,
    compile_args: Optional[Iterable[str]] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> tuple[int, int, int]:
    _configure_libclang()

    final_args = list(DEFAULT_COMPILE_ARGS)
    if compile_args:
        final_args.extend(list(compile_args))

    src_dir_path = Path(src_dir)
    if not src_dir_path.exists():
        raise FileNotFoundError(f"源码目录不存在: {src_dir}")

    cpp_files: list[str] = []
    for root, _, files in os.walk(str(src_dir_path)):
        for f in files:
            if f.endswith(".cpp"):
                cpp_files.append(os.path.join(root, f))

    if not cpp_files:
        raise ValueError(f"未在目录中找到任何 .cpp 文件: {src_dir}")

    func_csv_path_p = Path(func_csv_path).resolve()
    struct_csv_path_p = Path(struct_csv_path).resolve()
    func_count = 0
    struct_count = 0

    with func_csv_path_p.open("w", newline="", encoding="utf-8-sig") as f_func, struct_csv_path_p.open(
        "w", newline="", encoding="utf-8-sig"
    ) as f_struct:
        func_writer = csv.writer(f_func)
        struct_writer = csv.writer(f_struct)

        func_writer.writerow([
            "文件路径",
            "函数签名",
            "返回值类型",
            "起始行",
            "结束行",
            "函数体",
            "所属域",
            "函数类型",
        ])
        struct_writer.writerow([
            "struct_id",
            "file_path",
            "struct_name",
            "category",
            "start_line",
            "end_line",
            "member_signature",
            "source_code",
        ])

        total_files = len(cpp_files)
        for idx, file_path in enumerate(cpp_files, start=1):
            if progress_callback is not None:
                try:
                    progress_callback(idx, total_files, f"正在分析文件: {file_path}")
                except Exception:
                    pass

            before_func = getattr(f_func, "_rows_written", None)
            before_struct = getattr(f_struct, "_rows_written", None)
            analyze_cpp_file(file_path, final_args, func_writer, struct_writer)
            after_func = getattr(f_func, "_rows_written", None)
            after_struct = getattr(f_struct, "_rows_written", None)
            if before_func is None or after_func is None:
                pass
            if before_struct is None or after_struct is None:
                pass

    try:
        with func_csv_path_p.open("r", encoding="utf-8-sig") as f:
            func_count = max(0, sum(1 for _ in f) - 1)
    except Exception:
        func_count = 0
    try:
        with struct_csv_path_p.open("r", encoding="utf-8-sig") as f:
            struct_count = max(0, sum(1 for _ in f) - 1)
    except Exception:
        struct_count = 0

    return len(cpp_files), func_count, struct_count

