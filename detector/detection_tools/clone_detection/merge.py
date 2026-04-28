from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Optional


def default_logger(msg: str) -> None:
    print(msg, flush=True)


def _load_csv(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(r) for r in reader]


def merge_function_results(
    type12_csv: Path,
    type34_csv: Path,
    output_csv: Path,
    logger: Callable[[str], None] = default_logger,
) -> int:
    type12_csv = Path(type12_csv).resolve()
    type34_csv = Path(type34_csv).resolve()
    output_csv = Path(output_csv).resolve()

    t12 = _load_csv(type12_csv)
    t34 = _load_csv(type34_csv)

    def key_of(r: dict) -> tuple[str, str, str, str, str, str]:
        f1 = str(r.get("file1") or "").lower().replace("\\", "/")
        f2 = str(r.get("file2") or "").lower().replace("\\", "/")
        n1 = str(r.get("func1_name") or "").strip()
        n2 = str(r.get("func2_name") or "").strip()
        l1 = str(r.get("lines1") or "").strip()
        l2 = str(r.get("lines2") or "").strip()
        return f1, n1, l1, f2, n2, l2

    merged: dict[tuple[str, str, str, str, str, str], dict] = {}

    def upsert(row: dict, method: str, sim: float, raw: dict) -> None:
        k = key_of(row)
        existing = merged.get(k)
        if existing is None:
            existing = {
                "pair_id": "",
                "code_type": "function",
                "function_id_1": row.get("function_id_1", ""),
                "file1": row.get("file1", ""),
                "func1_name": row.get("func1_name", ""),
                "lines1": row.get("lines1", ""),
                "func1_body": row.get("func1_body", ""),
                "function_id_2": row.get("function_id_2", ""),
                "file2": row.get("file2", ""),
                "func2_name": row.get("func2_name", ""),
                "lines2": row.get("lines2", ""),
                "func2_body": row.get("func2_body", ""),
                "similarity": sim,
                "detection_method": method,
                "Clone_Type_Label": row.get("Clone_Type_Label", ""),
                "type12_clone_type": "",
                "type12_combined_similarity": "",
                "type34_similarity": "",
            }
            merged[k] = existing

        existing_methods = set(
            m.strip() for m in str(existing.get("detection_method") or "").split("+") if m.strip()
        )
        existing_methods.add(method)
        existing["detection_method"] = " + ".join(sorted(existing_methods))

        if sim > float(existing.get("similarity") or 0):
            existing["similarity"] = sim

        if method == "Func Clone (Type1-2)":
            existing["type12_clone_type"] = raw.get("clone_type", "")
            existing["type12_combined_similarity"] = raw.get("combined_similarity", "")
            if not existing.get("Clone_Type_Label"):
                existing["Clone_Type_Label"] = raw.get("Clone_Type_Label", "")

        if method == "Func Clone (Embedding)":
            existing["type34_similarity"] = raw.get("similarity", "")
            if not existing.get("Clone_Type_Label"):
                existing["Clone_Type_Label"] = raw.get("Clone_Type_Label", "")

    for r in t12:
        try:
            sim = float(r.get("combined_similarity") or 0)
        except Exception:
            sim = 0.0
        upsert(r, "Func Clone (Type1-2)", sim, r)

    for r in t34:
        try:
            sim = float(r.get("similarity") or 0)
        except Exception:
            sim = 0.0
        upsert(r, "Func Clone (Embedding)", sim, r)

    rows = list(merged.values())
    rows.sort(key=lambda x: float(x.get("similarity") or 0), reverse=True)
    for idx, r in enumerate(rows, start=1):
        r["pair_id"] = f"pair_{idx:03d}"

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pair_id",
        "code_type",
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
        "detection_method",
        "Clone_Type_Label",
        "type12_clone_type",
        "type12_combined_similarity",
        "type34_similarity",
    ]
    with output_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger(f"[MergeFunc] Done: {len(rows)} pairs -> {output_csv}")
    return len(rows)

