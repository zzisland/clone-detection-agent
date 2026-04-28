from __future__ import annotations

import csv
import itertools
import os
import tempfile
from pathlib import Path
from typing import Any, Optional


class StructDetector:
    def __init__(
        self,
        src_dir: str,
        output_path: str,
        use_gpu: bool = False,
        ollama_config: Optional[dict[str, Any]] = None,
        struct_csv_path: Optional[str] = None,
    ):
        self.src_dir = src_dir
        self.output_path = output_path
        self.use_gpu = use_gpu
        self.BATCH_SIZE = 8
        self.THRESH_STRUCT = 0.8
        self.ollama_config = ollama_config
        self.struct_csv_path = struct_csv_path

    def run_detection(self, detect_type: str = "all", progress_callback=None) -> tuple[int, int]:
        try:
            import pandas as pd  # type: ignore
        except Exception as e:
            raise ImportError(f"Package 'pandas' is required for struct detection: {e}")

        if progress_callback:
            progress_callback(5, "正在加载结构体数据...")

        structs: list[dict[str, Any]] = []
        if self.struct_csv_path and os.path.exists(self.struct_csv_path):
            with open(self.struct_csv_path, "r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                structs.extend(list(reader))
            if progress_callback:
                progress_callback(10, f"已加载切片文件: {len(structs)} 个结构体")
        else:
            if not self.src_dir:
                raise ValueError("src_dir is required when struct_csv_path is not provided")
            from .slice import slice_to_csv
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_func_csv = os.path.join(tmpdir, "ignore_funcs.csv")
                tmp_struct_csv = os.path.join(tmpdir, "input_structs.csv")
                slice_to_csv(
                    self.src_dir,
                    func_csv_path=tmp_func_csv,
                    struct_csv_path=tmp_struct_csv,
                )
                with open(tmp_struct_csv, "r", encoding="utf-8-sig", newline="") as f:
                    reader = csv.DictReader(f)
                    structs.extend(list(reader))

        if not structs:
            self._write_empty_reports()
            return 0, 0

        output_dir = os.path.dirname(self.output_path) or (os.path.dirname(self.src_dir) if self.src_dir else ".")
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(self.output_path))[0]
        type12_output = os.path.join(output_dir, f"{base_name}_type12_tool.csv")
        type34_output = os.path.join(output_dir, f"{base_name}_type34_model.csv")
        merged_output = self.output_path

        all_results: list[dict[str, Any]] = []

        type12_count = 0
        if detect_type in ["type12", "all"]:
            if progress_callback:
                progress_callback(20, "正在执行 Type 1-2 (指纹比对)...")
            t12_pairs = self._detect_type12(structs)
            type12_count = len(t12_pairs)
            all_results.extend(t12_pairs)
            self._write_report(type12_output, t12_pairs)

        type34_count = 0
        if detect_type in ["type34", "all"]:
            if progress_callback:
                progress_callback(40, "正在执行 Type 3-4 (Embedding)...")
            t34_pairs = self._detect_type34(structs, progress_callback)
            type34_count = len(t34_pairs)
            all_results.extend(t34_pairs)
            self._write_report(type34_output, t34_pairs)

        if progress_callback:
            progress_callback(95, "正在保存报告...")

        merged_df = self._merge_reports(type12_output, type34_output)
        merged_df.to_csv(merged_output, index=False, encoding="utf-8-sig")

        if progress_callback:
            progress_callback(100, "检测完成")

        return type12_count, type34_count

    def _write_empty_reports(self) -> None:
        try:
            import pandas as pd  # type: ignore
        except Exception:
            return
        output_dir = os.path.dirname(self.output_path) or "."
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(self.output_path))[0]
        type12_output = os.path.join(output_dir, f"{base_name}_type12_tool.csv")
        type34_output = os.path.join(output_dir, f"{base_name}_type34_model.csv")
        merged_output = self.output_path
        cols = [
            "pair_id", "code_type", "function_id_1", "file1", "func1_name", "lines1", "func1_body",
            "function_id_2", "file2", "func2_name", "lines2", "func2_body",
            "similarity", "detection_method",
        ]
        empty = pd.DataFrame(columns=cols)
        empty.to_csv(type12_output, index=False, encoding="utf-8-sig")
        empty.to_csv(type34_output, index=False, encoding="utf-8-sig")
        empty.to_csv(merged_output, index=False, encoding="utf-8-sig")

    def _write_report(self, path: str, pairs: list[dict[str, Any]]) -> None:
        import pandas as pd  # type: ignore

        cols = [
            "pair_id", "code_type", "function_id_1", "file1", "func1_name", "lines1", "func1_body",
            "function_id_2", "file2", "func2_name", "lines2", "func2_body",
            "similarity", "detection_method",
        ]
        df = pd.DataFrame(pairs, columns=cols) if pairs else pd.DataFrame(columns=cols)
        if not df.empty and "similarity" in df.columns:
            df = df.sort_values(by="similarity", ascending=False).reset_index(drop=True)
            df["pair_id"] = [f"pair_{i + 1:03d}" for i in range(len(df))]
            cols2 = ["pair_id"] + [c for c in df.columns if c != "pair_id"]
            df = df[cols2]
        df.to_csv(path, index=False, encoding="utf-8-sig")

    def _merge_reports(self, type12_path: str, type34_path: str):
        import pandas as pd  # type: ignore

        merged_items: list[dict[str, Any]] = []

        def load_csv(p: str):
            if os.path.exists(p) and os.path.getsize(p) > 0:
                try:
                    return pd.read_csv(p, encoding="utf-8-sig")
                except Exception:
                    return pd.DataFrame()
            return pd.DataFrame()

        df_t12 = load_csv(type12_path)
        df_t34 = load_csv(type34_path)

        for _, row in df_t12.iterrows():
            merged_items.append({
                "file1": str(row.get("file1", "")).lower().replace("\\", "/"),
                "file2": str(row.get("file2", "")).lower().replace("\\", "/"),
                "func1_name": str(row.get("func1_name", "")).strip(),
                "func2_name": str(row.get("func2_name", "")).strip(),
                "lines1": str(row.get("lines1", "")).strip(),
                "lines2": str(row.get("lines2", "")).strip(),
                "type12_sim": row.get("similarity", 1.0),
                "type34_sim": None,
                "type12_method": row.get("detection_method", "Struct Clone (Tool)"),
                "type34_method": None,
                "data": row.to_dict(),
            })

        for _, row in df_t34.iterrows():
            file1 = str(row.get("file1", "")).lower().replace("\\", "/")
            file2 = str(row.get("file2", "")).lower().replace("\\", "/")
            func1 = str(row.get("func1_name", "")).strip()
            func2 = str(row.get("func2_name", "")).strip()
            lines1 = str(row.get("lines1", "")).strip()
            lines2 = str(row.get("lines2", "")).strip()

            existing = None
            for item in merged_items:
                if (
                    item["file1"] == file1
                    and item["file2"] == file2
                    and item["func1_name"] == func1
                    and item["func2_name"] == func2
                    and item["lines1"] == lines1
                    and item["lines2"] == lines2
                ):
                    existing = item
                    break

            if existing:
                existing["type34_sim"] = row.get("similarity", 0)
                existing["type34_method"] = row.get("detection_method", "Struct Clone (Ollama)")
                existing["data"].update(row.to_dict())
            else:
                merged_items.append({
                    "file1": file1,
                    "file2": file2,
                    "func1_name": func1,
                    "func2_name": func2,
                    "lines1": lines1,
                    "lines2": lines2,
                    "type12_sim": None,
                    "type34_sim": row.get("similarity", 0),
                    "type12_method": None,
                    "type34_method": row.get("detection_method", "Struct Clone (Ollama)"),
                    "data": row.to_dict(),
                })

        final_rows: list[dict[str, Any]] = []
        for item in merged_items:
            data = item["data"]
            sims = []
            if item["type12_sim"] is not None:
                sims.append(item["type12_sim"])
            if item["type34_sim"] is not None:
                sims.append(item["type34_sim"])
            max_sim = max(sims) if sims else 0

            methods = []
            if item["type12_method"]:
                methods.append(item["type12_method"])
            if item["type34_method"]:
                methods.append(item["type34_method"])
            detection_method = " + ".join(sorted(set(methods))) if methods else ""

            final_rows.append({
                "pair_id": "",
                "code_type": data.get("code_type", "struct"),
                "function_id_1": data.get("function_id_1", ""),
                "file1": data.get("file1", ""),
                "func1_name": data.get("func1_name", ""),
                "lines1": data.get("lines1", ""),
                "func1_body": data.get("func1_body", ""),
                "function_id_2": data.get("function_id_2", ""),
                "file2": data.get("file2", ""),
                "func2_name": data.get("func2_name", ""),
                "lines2": data.get("lines2", ""),
                "func2_body": data.get("func2_body", ""),
                "similarity": round(float(max_sim), 4) if max_sim else 0,
                "detection_method": detection_method,
            })

        df = pd.DataFrame(final_rows) if final_rows else pd.DataFrame(columns=[
            "pair_id", "code_type", "function_id_1", "file1", "func1_name", "lines1", "func1_body",
            "function_id_2", "file2", "func2_name", "lines2", "func2_body",
            "similarity", "detection_method",
        ])

        if not df.empty and "similarity" in df.columns:
            df = df.sort_values(by="similarity", ascending=False).reset_index(drop=True)
            df["pair_id"] = [f"pair_{i + 1:03d}" for i in range(len(df))]
            cols = ["pair_id"] + [c for c in df.columns if c != "pair_id"]
            df = df[cols]
        return df

    def _detect_type12(self, structs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        signature_map: dict[str, list[dict[str, Any]]] = {}
        for s in structs:
            raw_sig = str(s.get("member_signature", "")).strip()
            if not raw_sig or len(raw_sig) < 2:
                continue

            clean_sig = raw_sig.replace(" *", "*").replace("* ", "*")

            parts: list[str] = []
            current = ""
            bracket_depth = 0
            for ch in clean_sig:
                if ch == "<":
                    bracket_depth += 1
                    current += ch
                elif ch == ">":
                    bracket_depth = max(0, bracket_depth - 1)
                    current += ch
                elif ch == "," and bracket_depth == 0:
                    if current.strip():
                        parts.append(current.strip())
                    current = ""
                else:
                    current += ch
            if current.strip():
                parts.append(current.strip())

            parts.sort()
            normalized_sig = ",".join(parts)
            signature_map.setdefault(normalized_sig, []).append(s)

        pairs: list[dict[str, Any]] = []
        for _, group in signature_map.items():
            if len(group) > 1:
                for s1, s2 in itertools.combinations(group, 2):
                    pairs.append({
                        "pair_id": f"pair_{len(pairs) + 1:03d}",
                        "code_type": "struct",
                        "function_id_1": s1.get("struct_id", ""),
                        "file1": s1.get("file_path", ""),
                        "func1_name": s1.get("struct_name", ""),
                        "lines1": f"{s1.get('start_line', '')}-{s1.get('end_line', '')}",
                        "func1_body": s1.get("source_code", ""),
                        "function_id_2": s2.get("struct_id", ""),
                        "file2": s2.get("file_path", ""),
                        "func2_name": s2.get("struct_name", ""),
                        "lines2": f"{s2.get('start_line', '')}-{s2.get('end_line', '')}",
                        "func2_body": s2.get("source_code", ""),
                        "similarity": 1.0,
                        "detection_method": "Struct Clone (Tool)",
                    })
        return pairs

    def _detect_type34(self, structs: list[dict[str, Any]], callback=None) -> list[dict[str, Any]]:
        if not self.ollama_config or not self.ollama_config.get("model_name"):
            return []
        return self._detect_type34_with_ollama(structs, callback)

    def _detect_type34_with_ollama(self, structs: list[dict[str, Any]], callback=None) -> list[dict[str, Any]]:
        try:
            import requests  # type: ignore
        except Exception:
            return []

        try:
            from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
        except Exception as e:
            raise ImportError(f"Package 'scikit-learn' is required for embedding similarity: {e}")

        codes = [str(s.get("source_code", "") or "") for s in structs]
        if not codes:
            return []

        model_name = str(self.ollama_config.get("model_name"))
        api_url = str(self.ollama_config.get("api_url", "http://172.16.220.180:8003"))
        api_key = str(self.ollama_config.get("api_key", "") or "")

        api_base = api_url.rstrip("/")
        if not api_base.endswith("/v1"):
            api_base = f"{api_base}/v1"
        embedding_url = f"{api_base}/embeddings"

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = api_key if api_key.lower().startswith("bearer ") else f"Bearer {api_key}"

        if callback:
            callback(40, f"使用本地部署模型 {model_name} 进行语义分析...")

        embeddings: list[list[float]] = []
        batch_size = 10
        for i in range(0, len(codes), batch_size):
            batch = codes[i:i + batch_size]
            try:
                resp = requests.post(
                    embedding_url,
                    headers=headers,
                    json={"model": model_name, "input": batch},
                    timeout=60,
                )
                if resp.status_code == 200:
                    data = resp.json().get("data", [])
                    for item in data:
                        embeddings.append(item.get("embedding", [0.0] * 768))
                else:
                    embeddings.extend([[0.0] * 768] * len(batch))
            except Exception:
                embeddings.extend([[0.0] * 768] * len(batch))

        if len(embeddings) < 2:
            return []

        sim_matrix = cosine_similarity(embeddings)
        pairs: list[dict[str, Any]] = []
        threshold = self.THRESH_STRUCT
        for i in range(len(structs)):
            for j in range(i + 1, len(structs)):
                similarity = float(sim_matrix[i][j])
                if similarity < threshold:
                    continue
                s1 = structs[i]
                s2 = structs[j]
                pairs.append({
                    "pair_id": f"pair_{len(pairs) + 1:03d}",
                    "code_type": "struct",
                    "function_id_1": s1.get("struct_id", ""),
                    "file1": s1.get("file_path", ""),
                    "func1_name": s1.get("struct_name", ""),
                    "lines1": f"{s1.get('start_line', '')}-{s1.get('end_line', '')}",
                    "func1_body": s1.get("source_code", ""),
                    "function_id_2": s2.get("struct_id", ""),
                    "file2": s2.get("file_path", ""),
                    "func2_name": s2.get("struct_name", ""),
                    "lines2": f"{s2.get('start_line', '')}-{s2.get('end_line', '')}",
                    "func2_body": s2.get("source_code", ""),
                    "similarity": similarity,
                    "detection_method": "Struct Clone (Ollama)",
                })

        return pairs

