from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Callable, Optional


def default_logger(msg: str) -> None:
    print(msg, flush=True)


def _read_functions_csv(input_csv: Path) -> list[dict]:
    rows: list[dict] = []
    with input_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            row = dict(row)
            row["_id"] = f"{idx:03d}"
            rows.append(row)
    return rows


def _normalize_api_base(api_url: str) -> str:
    api_base = api_url.rstrip("/")
    if not api_base.endswith("/v1"):
        api_base = f"{api_base}/v1"
    return api_base


def _cosine(a: list[float], b: list[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)


def run_type34_pipeline(
    input_csv: Path,
    output_csv: Path,
    api_url: str,
    model_name: str,
    api_key: str,
    threshold: float = 0.80,
    batch_size: int = 8,
    logger: Callable[[str], None] = default_logger,
) -> int:
    try:
        import requests  # type: ignore
    except Exception as e:
        raise ImportError(f"Package 'requests' is required: {e}")

    input_csv = Path(input_csv).resolve()
    output_csv = Path(output_csv).resolve()

    if not input_csv.exists():
        raise FileNotFoundError(f"输入 CSV 不存在: {input_csv}")

    rows = _read_functions_csv(input_csv)
    if not rows:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", encoding="utf-8-sig", newline="") as f:
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
        return 0

    api_base = _normalize_api_base(api_url)
    embedding_url = f"{api_base}/embeddings"

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = api_key if api_key.lower().startswith("bearer ") else f"Bearer {api_key}"

    codes = [(r.get("函数体") or "") for r in rows]
    func_types = [(r.get("函数类型") or "Function") for r in rows]

    logger(f"[FuncType34] functions={len(rows)}, batch_size={batch_size}")
    logger(f"[FuncType34] embedding_url={embedding_url}, model={model_name}")

    embeddings: list[list[float]] = []
    for i in range(0, len(codes), max(1, batch_size)):
        batch = codes[i:i + max(1, batch_size)]
        payload = {"model": model_name, "input": batch}
        try:
            resp = requests.post(embedding_url, headers=headers, json=payload, timeout=60)
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                for item in data:
                    embeddings.append(item.get("embedding", [0.0] * 768))
                if len(data) != len(batch):
                    missing = len(batch) - len(data)
                    embeddings.extend([[0.0] * 768] * max(0, missing))
            else:
                embeddings.extend([[0.0] * 768] * len(batch))
        except Exception:
            embeddings.extend([[0.0] * 768] * len(batch))

        done = min(i + len(batch), len(codes))
        logger(f"[FuncType34] embedding {done}/{len(codes)}")

    results: list[dict] = []
    n = len(rows)
    for i in range(n):
        for j in range(i + 1, n):
            sim = _cosine(embeddings[i], embeddings[j])
            if sim < threshold:
                continue

            ft1 = (func_types[i] or "Function").strip() or "Function"
            ft2 = (func_types[j] or "Function").strip() or "Function"
            if ft1 == "Method" and ft2 == "Method":
                label = "Class_Method_Clone"
            elif ft1 == "Method" or ft2 == "Method":
                label = "Mixed_Type_Clone"
            else:
                label = "Function_Clone"

            r1 = rows[i]
            r2 = rows[j]
            results.append(
                {
                    "pair_id": "",
                    "function_id_1": r1["_id"],
                    "file1": r1.get("文件路径", ""),
                    "func1_name": r1.get("函数签名", ""),
                    "lines1": f"{r1.get('起始行', '')}-{r1.get('结束行', '')}",
                    "func1_body": codes[i],
                    "function_id_2": r2["_id"],
                    "file2": r2.get("文件路径", ""),
                    "func2_name": r2.get("函数签名", ""),
                    "lines2": f"{r2.get('起始行', '')}-{r2.get('结束行', '')}",
                    "func2_body": codes[j],
                    "similarity": round(sim, 4),
                    "Clone_Type_Label": label,
                }
            )

    results.sort(key=lambda x: float(x.get("similarity") or 0), reverse=True)
    for idx, r in enumerate(results, start=1):
        r["pair_id"] = f"pair_{idx:03d}"

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as f:
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
        writer.writerows(results)

    logger(f"[FuncType34] Done: {len(results)} clone pairs -> {output_csv}")
    return len(results)

