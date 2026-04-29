from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable


TYPE12_HIGH_CONF_THRESHOLD = 0.8
TYPE34_HIGH_CONF_THRESHOLD = 0.8


def default_logger(msg: str) -> None:
    print(msg, flush=True)


def _load_csv(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(r) for r in reader]


def _norm_path(value: object) -> str:
    return str(value or "").lower().replace("\\", "/").strip()


def _norm_text(value: object) -> str:
    return str(value or "").strip()


def _function_point_key(file_path: object, lines: object) -> tuple[str, str]:
    return _norm_path(file_path), _norm_text(lines)


def _pair_key(row: dict) -> tuple[str, tuple[str, str], tuple[str, str]]:
    code_type = _norm_text(row.get("code_type") or "function").lower()
    pair1 = _function_point_key(row.get("file1"), row.get("lines1"))
    pair2 = _function_point_key(row.get("file2"), row.get("lines2"))
    ordered = tuple(sorted([pair1, pair2]))
    return code_type, ordered[0], ordered[1]


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _compute_confidence(type12_score: float, type34_score: float) -> tuple[str, str, float]:
    has_type12 = type12_score > 0
    has_type34 = type34_score > 0
    consensus_hit = "yes" if has_type12 and has_type34 else "no"

    confidence_score = max(type12_score, type34_score)

    if consensus_hit == "yes":
        confidence_level = "high"
    elif type12_score >= TYPE12_HIGH_CONF_THRESHOLD or type34_score >= TYPE34_HIGH_CONF_THRESHOLD:
        confidence_level = "medium"
    else:
        confidence_level = "low"

    return consensus_hit, confidence_level, min(confidence_score, 1.0)


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

    merged: dict[tuple[str, tuple[str, str], tuple[str, str]], dict] = {}

    def upsert(row: dict, method: str, sim: float, raw: dict) -> None:
        k = _pair_key(row)
        existing = merged.get(k)
        clone_type_label = row.get("Clone_Type_Label", "")

        if existing is None:
            existing = {
                "pair_id": "",
                "code_type": row.get("code_type", "function") or "function",
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
                "Clone_Type_Label": clone_type_label,
                "type12_clone_type": "",
                "type12_combined_similarity": "",
                "type34_similarity": "",
                "consensus_hit": "no",
                "confidence_level": "low",
                "confidence_score": 0.0,
            }
            merged[k] = existing
        elif not existing.get("Clone_Type_Label") and clone_type_label:
            existing["Clone_Type_Label"] = clone_type_label

        existing_methods = {
            m.strip() for m in str(existing.get("detection_method") or "").split("+") if m.strip()
        }
        existing_methods.add(method)
        existing["detection_method"] = " + ".join(sorted(existing_methods))

        existing_similarity = _safe_float(existing.get("similarity") or 0)
        if sim > existing_similarity:
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

        type12_score = _safe_float(existing.get("type12_combined_similarity") or 0)
        type34_score = _safe_float(existing.get("type34_similarity") or 0)
        consensus_hit, confidence_level, confidence_score = _compute_confidence(type12_score, type34_score)
        existing["consensus_hit"] = consensus_hit
        existing["confidence_level"] = confidence_level
        existing["confidence_score"] = confidence_score

    for r in t12:
        sim = _safe_float(r.get("combined_similarity") or 0)
        upsert(r, "Func Clone (Type1-2)", sim, r)

    for r in t34:
        sim = _safe_float(r.get("similarity") or 0)
        upsert(r, "Func Clone (Embedding)", sim, r)

    confidence_rank = {"high": 3, "medium": 2, "low": 1}
    rows = list(merged.values())
    rows.sort(
        key=lambda x: (
            confidence_rank.get(str(x.get("confidence_level") or "low"), 1),
            _safe_float(x.get("confidence_score") or 0),
            _safe_float(x.get("similarity") or 0),
        ),
        reverse=True,
    )
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
        "consensus_hit",
        "confidence_level",
        "confidence_score",
    ]
    with output_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger(f"[MergeFunc] Done: {len(rows)} pairs -> {output_csv}")
    return len(rows)

