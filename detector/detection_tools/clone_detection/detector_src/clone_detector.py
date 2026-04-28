"""
克隆检测模块
从隐秘化的 C++ 代码中检测 Type-1 和 Type-2 克隆
"""
from __future__ import annotations
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from difflib import SequenceMatcher
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Set, Tuple

try:
    csv.field_size_limit(1024 * 1024 * 100)
except (OverflowError, ValueError):
    for limit in (1024 * 1024 * 50, 1024 * 1024 * 20):
        try:
            csv.field_size_limit(limit)
            break
        except Exception:
            continue


CPP_KEYWORDS = {
    "alignas", "alignof", "and", "and_eq", "asm", "auto", "bitand", "bitor", "bool", "break",
    "case", "catch", "char", "char8_t", "char16_t", "char32_t", "class", "compl", "concept",
    "const", "consteval", "constexpr", "constinit", "const_cast", "continue", "co_await",
    "co_return", "co_yield", "decltype", "default", "delete", "do", "double", "dynamic_cast",
    "else", "enum", "explicit", "export", "extern", "false", "float", "for", "friend", "goto",
    "if", "inline", "int", "long", "mutable", "namespace", "new", "noexcept", "not", "not_eq",
    "nullptr", "operator", "or", "or_eq", "private", "protected", "public", "register",
    "reinterpret_cast", "requires", "return", "short", "signed", "sizeof", "static",
    "static_assert", "static_cast", "struct", "switch", "template", "this", "thread_local",
    "throw", "true", "try", "typedef", "typeid", "typename", "union", "unsigned", "using",
    "virtual", "void", "volatile", "wchar_t", "while", "xor", "xor_eq",
}

OPERATOR_TOKENS = {
    "==", "!=", "<=", ">=", "&&", "||", "::", "->", "<<", ">>", "+", "-", "*", "/", "%", "&",
    "|", "^", "~", "!", "<", ">", "=", "?", ":", ";", ",", ".", "(", ")", "{", "}", "[", "]",
}

FUNC_PATTERN = re.compile(
    r"^\s*(?:(?P<ret>[~\w:<>,\s\*\&]+?)\s+)?(?P<name>(?:\w+::)?FUN_\d+(?:::\w+)?)\s*\((?P<params>[^)]*)\)\s*(?::[^{]*)?\s*(?:const\s*)?(?:\{|[^{;]*;|$)"
)
FUNC_PATTERN_WITH_PREFIX = re.compile(
    r"(?:^|;)\s*(?:(?P<ret>[~\w:<>,\s\*\&]+?)\s+)?(?P<name>(?:\w+::)?FUN_\d+(?:::\w+)?)\s*\((?P<params>[^)]*)\)\s*(?::[^{]*)?\s*(?:const\s*)?(?:\{|[^{;]*;|$)"
)

GENERIC_NAME_PATTERN = re.compile(
    r"(?P<name>(?:\w+::)?FUN_\d+(?:::\w+)?|operator[^\s(]*|[A-Za-z_]\w*(?:::\w+)*)\s*\("
)

TYPE2_MIN_TOKEN_LENGTH = 25
TYPE2_MAX_LENGTH_DIFF_RATIO = 0.2

TYPE2_MIN_STATEMENT_COUNT = 2
TYPE2_MIN_CONTROL_FLOW_COUNT = 0
TYPE2_MIN_SEMANTIC_TOKENS = 30

TYPE2_TOKEN_JACCARD_THRESHOLD = 0.8
TYPE2_TRIGRAM_JACCARD_THRESHOLD = 0.75
TYPE2_SEQUENCE_THRESHOLD = 0.92
TYPE2_STRUCTURE_THRESHOLD = 0.85

TYPE2_SHORT_FUNCTION_THRESHOLD = 40
TYPE2_SHORT_SEQUENCE_THRESHOLD = 0.95
TYPE2_SHORT_COMBINED_THRESHOLD = 0.95

TYPE2_COMBINED_WEIGHT = 0.6
TYPE2_COMBINED_THRESHOLD = 0.9

TYPE2_NGRAM_SIZE = 3

TOKEN_PATTERN = re.compile(
    r"""
    (?P<identifier>[A-Za-z_]\w*)
    |(?P<number>0[xX][0-9a-fA-F]+|\d+\.\d+|\d+)
    |(?P<string>"([^"\\]|\\.)*")
    |(?P<char>'([^'\\]|\\.)*')
    |(?P<operator>==|!=|<=|>=|&&|\|\||::|->|<<|>>|[-+*/%&|^~!=<>?:;,(){}\[\]])
    |(?P<dot>\.)
    """,
    re.VERBOSE,
)

IDENTIFIER_RE = re.compile(r"^[A-Za-z_]\w*$")
NUMBER_RE = re.compile(r"^(?:0[xX][0-9a-fA-F]+|\d+\.\d+|\d+)$")


@dataclass
class FunctionInfo:
    index: int
    func_id: str
    name: str
    return_type: str
    start_line: int
    end_line: int
    source: str
    tokens: List[str] = field(default_factory=list)
    normalized_tokens: List[str] = field(default_factory=list)
    structure_vector: Counter = field(default_factory=Counter)
    token_length: int = 0
    normalized_token_set: Set[str] = field(default_factory=set)
    token_trigrams: Set[str] = field(default_factory=set)
    statement_count: int = 0
    control_flow_count: int = 0
    semantic_token_count: int = 0
    obfuscated_file: Optional[str] = None
    original_name: Optional[str] = None
    original_source: Optional[str] = None
    original_start_line: Optional[int] = None
    original_end_line: Optional[int] = None

    @property
    def line_range(self) -> str:
        return f"{self.start_line}-{self.end_line}"

    def to_serializable(self) -> dict:
        original_name = self.original_name or self.name
        original_source = shorten_source_path(
            self.original_source or self.obfuscated_file
        )
        original_start = self.original_start_line or self.start_line
        original_end = self.original_end_line or self.end_line
        original_range = (
            f"{original_start}-{original_end}"
            if original_start is not None and original_end is not None
            else None
        )
        return {
            "id": self.func_id,
            "obfuscated_line_range": self.line_range,
            "original_name": original_name,
            "original_source": original_source,
            "original_start_line": original_start,
            "original_end_line": original_end,
            "original_line_range": original_range,
        }


def compute_sequence_similarity(seq1: List[str], seq2: List[str]) -> float:
    if not seq1 or not seq2:
        return 0.0
    return SequenceMatcher(None, seq1, seq2).ratio()


def compute_structure_similarity(struct1: Counter, struct2: Counter) -> float:
    keys = set(struct1) | set(struct2)
    if not keys:
        return 1.0
    intersection = sum(min(struct1.get(key, 0), struct2.get(key, 0)) for key in keys)
    union = sum(max(struct1.get(key, 0), struct2.get(key, 0)) for key in keys)
    return intersection / union if union else 1.0


def compute_combined_similarity(sequence_sim: float, structure_sim: float) -> float:
    return (
        TYPE2_COMBINED_WEIGHT * sequence_sim
        + (1 - TYPE2_COMBINED_WEIGHT) * structure_sim
    )


def length_difference_ratio(len1: int, len2: int) -> float:
    if not len1 and not len2:
        return 0.0
    return abs(len1 - len2) / max(len1, len2, 1)


def generate_ngrams(tokens: List[str], n: int) -> Set[str]:
    if len(tokens) < n or n <= 0:
        return set()
    return {" ".join(tokens[i: i + n]) for i in range(len(tokens) - n + 1)}


def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union else 0.0


def shorten_source_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    try:
        normalized = Path(path).as_posix()
    except (TypeError, ValueError):
        normalized = str(path)
    markers = ("tesfiles/", "test/", "tests/")
    for marker in markers:
        if marker in normalized:
            result = normalized[normalized.index(marker):]
            return result.replace("/", "\\")
    return Path(normalized).name.replace("/", "\\")


def tokenize(source: str) -> Tuple[List[str], List[str], Counter]:
    raw_tokens: List[str] = []
    normalized_tokens: List[str] = []
    structure: Counter = Counter()
    for match in TOKEN_PATTERN.finditer(source):
        token = match.group(0)
        raw_tokens.append(token)
        if IDENTIFIER_RE.match(token):
            if token in CPP_KEYWORDS:
                normalized_tokens.append(token)
                structure["keywords"] += 1
            else:
                normalized_tokens.append("ID")
                structure["identifiers"] += 1
        elif NUMBER_RE.match(token):
            normalized_tokens.append("NUM")
            structure["numbers"] += 1
        elif token.startswith('"'):
            normalized_tokens.append("STR")
            structure["strings"] += 1
        elif token.startswith("'"):
            normalized_tokens.append("CHAR")
            structure["chars"] += 1
        elif token in OPERATOR_TOKENS:
            normalized_tokens.append(token)
            structure["operators"] += 1
        else:
            normalized_tokens.append(token)
            structure["others"] += 1
    return raw_tokens, normalized_tokens, structure


def compute_complexity_metrics(source: str, normalized_tokens: List[str]) -> Tuple[int, int, int]:
    statement_count = sum(1 for t in normalized_tokens if t in (";", "}", "{"))
    ctrl_tokens = {"if", "for", "while", "switch", "case", "catch", "try", "do"}
    control_flow_count = sum(1 for t in normalized_tokens if t in ctrl_tokens)
    semantic_token_count = sum(1 for t in normalized_tokens if t not in ("ID", "NUM", "STR", "CHAR"))
    return statement_count, control_flow_count, semantic_token_count


def extract_functions(source_path: Path) -> List[FunctionInfo]:
    source_text = source_path.read_text(encoding="utf-8", errors="ignore")
    lines = source_text.splitlines()

    def parse_signature(sig_lines: List[str]) -> tuple[str, str]:
        joined = " ".join(x.strip() for x in sig_lines if x.strip())
        joined = re.sub(r"template\s*<[^>]*>\s*", "", joined)
        match = FUNC_PATTERN_WITH_PREFIX.search(joined) or FUNC_PATTERN.search(joined)
        ret_val = ""
        name_val = ""
        if match:
            name_val = match.group("name") or ""
            ret_grp = match.group("ret")
            if ret_grp:
                ret_val = ret_grp.strip()
        else:
            generic = GENERIC_NAME_PATTERN.search(joined)
            if generic:
                name_val = generic.group("name") or ""
        return name_val or "", ret_val

    collected: List[dict] = []
    idx = 0
    total_lines = len(lines)
    while idx < total_lines:
        line = lines[idx]
        if line.strip().startswith("// Function"):
            start_line = idx + 2
            buffer_lines: List[str] = []
            sig_lines: List[str] = []
            j = idx + 1
            while j < total_lines and not lines[j].strip().startswith("// Function"):
                buffer_lines.append(lines[j])
                if len(sig_lines) < 8:
                    sig_lines.append(lines[j])
                if "{" in lines[j] or lines[j].strip().endswith(";"):
                    if len(sig_lines) < 8 and j + 1 < total_lines and not lines[j + 1].strip().startswith("// Function"):
                        sig_lines.append(lines[j + 1])
                j += 1
            end_line = j
            name_val, ret_val = parse_signature(sig_lines)
            collected.append(
                {
                    "name": name_val,
                    "return_type": ret_val,
                    "start_line": start_line,
                    "end_line": end_line,
                    "source": "\n".join(buffer_lines),
                    "obfuscated_file": str(source_path),
                }
            )
            idx = j
        else:
            idx += 1

    functions: List[FunctionInfo] = []
    for idx, item in enumerate(collected, start=1):
        func = FunctionInfo(
            index=idx,
            func_id=f"{idx:03d}",
            name=item["name"],
            return_type=item["return_type"],
            start_line=item["start_line"],
            end_line=item["end_line"],
            source=item["source"],
            obfuscated_file=item.get("obfuscated_file"),
        )
        tokens, normalized, structure = tokenize(func.source)
        func.tokens = tokens
        func.normalized_tokens = normalized
        func.structure_vector = structure
        func.token_length = len(normalized)
        func.normalized_token_set = set(normalized)
        func.token_trigrams = generate_ngrams(normalized, TYPE2_NGRAM_SIZE)
        stmt_count, ctrl_count, sem_count = compute_complexity_metrics(func.source, normalized)
        func.statement_count = stmt_count
        func.control_flow_count = ctrl_count
        func.semantic_token_count = sem_count
        functions.append(func)

    return functions


def apply_metadata(functions: Iterable[FunctionInfo], metadata: Dict[str, dict]) -> int:
    missing = 0
    for func in functions:
        info = metadata.get(func.func_id)
        if info is None:
            missing += 1
            continue
        func.original_name = info.get("function")
        func.original_source = info.get("source")
        func.original_start_line = info.get("start_line")
        func.original_end_line = info.get("end_line")
        func.func_type = info.get("func_type", "Function")
    return missing


def generate_clone_pairs(functions: Iterable[FunctionInfo]) -> Tuple[List[dict], dict]:
    functions_list = list(functions)
    clone_pairs: List[dict] = []
    seen_pairs = set()

    def describe(func: FunctionInfo, index: int) -> Dict[str, Optional[object]]:
        suffix = str(index)
        original_start = func.original_start_line or func.start_line
        original_end = func.original_end_line or func.end_line
        display_name = func.original_name or func.name
        display_source = shorten_source_path(
            func.original_source or func.obfuscated_file
        )
        original_range = (
            f"{original_start}-{original_end}"
            if original_start is not None and original_end is not None
            else None
        )
        obf_range = (
            f"{func.start_line}-{func.end_line}"
            if func.start_line is not None and func.end_line is not None
            else None
        )
        return {
            f"function_id_{suffix}": func.func_id,
            f"function_name_{suffix}": display_name,
            f"function_source_{suffix}": display_source,
            f"original_start_line_{suffix}": original_start,
            f"original_end_line_{suffix}": original_end,
            f"original_line_range_{suffix}": original_range,
            f"display_name_{suffix}": display_name,
            f"display_source_{suffix}": display_source,
            f"display_start_line_{suffix}": original_start,
            f"display_end_line_{suffix}": original_end,
            f"display_line_range_{suffix}": original_range,
            f"obfuscated_source_{suffix}": shorten_source_path(func.obfuscated_file),
            f"obfuscated_start_line_{suffix}": func.start_line,
            f"obfuscated_end_line_{suffix}": func.end_line,
            f"obfuscated_line_range_{suffix}": obf_range,
        }

    def record_pair(
        f1: FunctionInfo,
        f2: FunctionInfo,
        clone_type: str,
        extra: dict | None = None,
    ) -> None:
        entry: Dict[str, Optional[object]] = {
            "clone_type": clone_type,
        }
        entry.update(describe(f1, 1))
        entry.update(describe(f2, 2))
        if extra:
            entry.update(extra)
        clone_pairs.append(entry)

    type1_groups: defaultdict = defaultdict(list)
    for func in functions_list:
        if not func.tokens:
            continue
        type1_groups[tuple(func.tokens)].append(func)

    for group in type1_groups.values():
        if len(group) < 2:
            continue
        for f1, f2 in combinations(group, 2):
            key = tuple(sorted((f1.func_id, f2.func_id)))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            record_pair(f1, f2, "Type-1")

    type2_sequence_scores: List[float] = []
    type2_structure_scores: List[float] = []
    type2_combined_scores: List[float] = []
    type2_token_jaccards: List[float] = []
    type2_ngram_jaccards: List[float] = []
    type2_stats: Dict[str, int] = defaultdict(int)

    for f1, f2 in combinations(functions_list, 2):
        key = tuple(sorted((f1.func_id, f2.func_id)))
        if key in seen_pairs:
            type2_stats["skipped_existing_pair"] += 1
            continue

        type2_stats["candidate_pairs"] += 1

        if not f1.normalized_tokens or not f2.normalized_tokens:
            type2_stats["filtered_empty_tokens"] += 1
            continue

        if f1.token_length < TYPE2_MIN_TOKEN_LENGTH or f2.token_length < TYPE2_MIN_TOKEN_LENGTH:
            type2_stats["filtered_short_functions"] += 1
            continue

        diff_ratio = length_difference_ratio(f1.token_length, f2.token_length)
        if diff_ratio > TYPE2_MAX_LENGTH_DIFF_RATIO:
            type2_stats["filtered_length_diff"] += 1
            continue

        if (f1.statement_count < TYPE2_MIN_STATEMENT_COUNT or f2.statement_count < TYPE2_MIN_STATEMENT_COUNT):
            type2_stats["filtered_low_statement_count"] += 1
            continue

        if (f1.control_flow_count < TYPE2_MIN_CONTROL_FLOW_COUNT or f2.control_flow_count < TYPE2_MIN_CONTROL_FLOW_COUNT):
            type2_stats["filtered_low_control_flow"] += 1
            continue

        if (f1.semantic_token_count < TYPE2_MIN_SEMANTIC_TOKENS or f2.semantic_token_count < TYPE2_MIN_SEMANTIC_TOKENS):
            type2_stats["filtered_low_semantic_tokens"] += 1
            continue

        token_jaccard = jaccard_similarity(f1.normalized_token_set, f2.normalized_token_set)
        if token_jaccard < TYPE2_TOKEN_JACCARD_THRESHOLD:
            type2_stats["filtered_token_jaccard"] += 1
            continue

        ngram_jaccard = jaccard_similarity(f1.token_trigrams, f2.token_trigrams)
        if ngram_jaccard < TYPE2_TRIGRAM_JACCARD_THRESHOLD:
            type2_stats["filtered_ngram_jaccard"] += 1
            continue

        sequence_sim = compute_sequence_similarity(f1.normalized_tokens, f2.normalized_tokens)
        structure_sim = compute_structure_similarity(f1.structure_vector, f2.structure_vector)

        seq_threshold = TYPE2_SEQUENCE_THRESHOLD
        combined_threshold = TYPE2_COMBINED_THRESHOLD
        if min(f1.token_length, f2.token_length) < TYPE2_SHORT_FUNCTION_THRESHOLD:
            seq_threshold = max(seq_threshold, TYPE2_SHORT_SEQUENCE_THRESHOLD)
            combined_threshold = max(combined_threshold, TYPE2_SHORT_COMBINED_THRESHOLD)

        if sequence_sim < seq_threshold and structure_sim < TYPE2_STRUCTURE_THRESHOLD:
            type2_stats["filtered_low_seq_and_struct"] += 1
            continue

        combined_sim = compute_combined_similarity(sequence_sim, structure_sim)
        if combined_sim < combined_threshold:
            type2_stats["filtered_low_combined"] += 1
            continue

        record_pair(
            f1,
            f2,
            "Type-2",
            extra={
                "sequence_similarity": round(sequence_sim, 4),
                "structure_similarity": round(structure_sim, 4),
                "combined_similarity": round(combined_sim, 4),
                "token_jaccard": round(token_jaccard, 4),
                "ngram_jaccard": round(ngram_jaccard, 4),
            },
        )

        type2_sequence_scores.append(sequence_sim)
        type2_structure_scores.append(structure_sim)
        type2_combined_scores.append(combined_sim)
        type2_token_jaccards.append(token_jaccard)
        type2_ngram_jaccards.append(ngram_jaccard)
        type2_stats["accepted"] += 1

    def avg(values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    summary = {
        "type1_count": sum(1 for p in clone_pairs if p.get("clone_type") == "Type-1"),
        "type2_count": sum(1 for p in clone_pairs if p.get("clone_type") == "Type-2"),
        "type2_avg_sequence_similarity": round(avg(type2_sequence_scores), 4),
        "type2_avg_structure_similarity": round(avg(type2_structure_scores), 4),
        "type2_avg_combined_similarity": round(avg(type2_combined_scores), 4),
        "type2_avg_token_jaccard": round(avg(type2_token_jaccards), 4),
        "type2_avg_ngram_jaccard": round(avg(type2_ngram_jaccards), 4),
        "type2_stats": dict(type2_stats),
    }

    return clone_pairs, summary


def range_to_list(rng: Optional[str]) -> List[int]:
    if not rng:
        return []
    try:
        start, end = rng.split("-", 1)
        start_i = int(start)
        end_i = int(end)
        if start_i <= 0 or end_i <= 0:
            return []
        return list(range(start_i, end_i + 1))
    except Exception:
        return []


def load_function_metadata_from_csv(csv_path: Path) -> Dict[str, dict]:
    if not csv_path.exists():
        return {}
    metadata: Dict[str, dict] = {}
    encodings = ["utf-8-sig", "utf-8", "gbk", "gb2312"]
    for encoding in encodings:
        try:
            with csv_path.open("r", encoding=encoding) as csv_file:
                reader = csv.DictReader(csv_file)
                counter = 1
                for row in reader:
                    func_id = f"{counter:03d}"
                    func_name = (row.get("函数名") or "").strip()
                    if not func_name:
                        func_sig = (row.get("函数签名") or "").strip()
                        if func_sig:
                            before_paren = func_sig.split("(", 1)[0].strip()
                            if before_paren:
                                candidate = before_paren.split()[-1]
                                func_name = candidate.split("::")[-1]
                            if not func_name:
                                match = re.search(r"([^\s(]+)\s*\(", func_sig)
                                if match:
                                    func_name = match.group(1)
                    func_type = (row.get("函数类型") or "").strip()
                    metadata[func_id] = {
                        "id": func_id,
                        "function": func_name or f"function_{counter}",
                        "source": (row.get("文件路径") or "").strip(),
                        "start_line": int(row.get("起始行", 0)) if row.get("起始行") else None,
                        "end_line": int(row.get("结束行", 0)) if row.get("结束行") else None,
                        "func_type": func_type or "Function",
                    }
                    counter += 1
            return metadata
        except Exception:
            continue
    return {}


def load_function_body_mapping(csv_path: Path) -> Dict[str, str]:
    if not csv_path.exists():
        return {}
    body_map: Dict[str, str] = {}
    encodings = ["utf-8-sig", "utf-8", "gbk", "gb2312"]
    for encoding in encodings:
        try:
            with csv_path.open("r", encoding=encoding) as csv_file:
                reader = csv.DictReader(csv_file)
                counter = 1
                for row in reader:
                    func_id = f"{counter:03d}"
                    func_body = row.get("函数体", "").strip()
                    if func_body:
                        body_map[func_id] = func_body
                    counter += 1
            return body_map
        except Exception:
            continue
    return {}


def write_csv(
    clone_pairs: List[dict],
    functions: List[FunctionInfo],
    csv_path: Path,
    body_mapping: Optional[Dict[str, str]] = None,
) -> None:
    func_body_map: Dict[str, str] = {}
    if body_mapping:
        func_body_map = body_mapping.copy()
    func_info_map = {func.func_id: func for func in functions}
    if not body_mapping:
        for func in functions:
            func_body_map[func.func_id] = func.source

    fieldnames = [
        "pair_id",
        "clone_type",
        "Clone_Type_Label",
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
        "sequence_similarity",
        "structure_similarity",
        "combined_similarity",
        "token_jaccard",
        "ngram_jaccard",
    ]

    def resolve_name(func_id: str) -> str | None:
        func = func_info_map.get(func_id)
        if func and func.original_name:
            return func.original_name
        return None

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for idx, pair in enumerate(clone_pairs, start=1):
            func1_id = pair.get("function_id_1", "")
            func2_id = pair.get("function_id_2", "")
            func1_info = func_info_map.get(func1_id)
            func2_info = func_info_map.get(func2_id)

            file1 = None
            if func1_info:
                file1 = func1_info.original_source or func1_info.obfuscated_file
            file2 = None
            if func2_info:
                file2 = func2_info.original_source or func2_info.obfuscated_file
            lines1 = pair.get("display_line_range_1") or pair.get("original_line_range_1")
            lines2 = pair.get("display_line_range_2") or pair.get("original_line_range_2")

            func1_body = func_body_map.get(func1_id) or (func1_info.source if func1_info else "")
            func2_body = func_body_map.get(func2_id) or (func2_info.source if func2_info else "")

            func1_type = getattr(func1_info, "func_type", "Function") if func1_info else "Function"
            func2_type = getattr(func2_info, "func_type", "Function") if func2_info else "Function"
            if func1_type == "Method" and func2_type == "Method":
                clone_type_label = "Class_Method_Clone"
            elif func1_type == "Method" or func2_type == "Method":
                clone_type_label = "Mixed_Type_Clone"
            else:
                clone_type_label = "Function_Clone"

            row = {
                "pair_id": f"pair_{idx:03d}",
                "clone_type": pair.get("clone_type"),
                "Clone_Type_Label": clone_type_label,
                "function_id_1": func1_id,
                "file1": file1,
                "func1_name": resolve_name(func1_id) or pair.get("function_name_1"),
                "lines1": lines1,
                "func1_body": func1_body,
                "function_id_2": func2_id,
                "file2": file2,
                "func2_name": resolve_name(func2_id) or pair.get("function_name_2"),
                "lines2": lines2,
                "func2_body": func2_body,
                "sequence_similarity": pair.get("sequence_similarity"),
                "structure_similarity": pair.get("structure_similarity"),
                "combined_similarity": pair.get("combined_similarity"),
                "token_jaccard": pair.get("token_jaccard"),
                "ngram_jaccard": pair.get("ngram_jaccard"),
            }
            writer.writerow(row)


def write_json(
    functions: List[FunctionInfo],
    clone_pairs: List[dict],
    summary: dict,
    json_path: Path,
    source_path: Path,
    project_name: str = "gme",
) -> None:
    func_info_map = {func.func_id: func for func in functions}

    def resolve_name(func_id: str) -> str | None:
        func = func_info_map.get(func_id)
        if func and func.original_name:
            return func.original_name
        return None

    metrics = [
        "sequence_similarity",
        "structure_similarity",
        "combined_similarity",
        "token_jaccard",
        "ngram_jaccard",
    ]
    json_clone_pairs: List[dict] = []
    for idx, pair in enumerate(clone_pairs, start=1):
        file1 = shorten_source_path(
            pair.get("function_source_1") or pair.get("display_source_1")
        )
        file2 = shorten_source_path(
            pair.get("function_source_2") or pair.get("display_source_2")
        )
        lines1_display = range_to_list(
            pair.get("display_line_range_1") or pair.get("original_line_range_1")
        )
        lines2_display = range_to_list(
            pair.get("display_line_range_2") or pair.get("original_line_range_2")
        )
        lines1_obf = range_to_list(pair.get("obfuscated_line_range_1"))
        lines2_obf = range_to_list(pair.get("obfuscated_line_range_2"))

        entry = {
            "id": f"pair_{idx:03d}",
            "project": project_name,
            "func1_id": pair.get("function_id_1"),
            "file1": file1,
            "func1_name": resolve_name(pair.get("function_id_1", "")) or pair.get("function_name_1"),
            "lines1": lines1_display,
            "func2_id": pair.get("function_id_2"),
            "file2": file2,
            "func2_name": resolve_name(pair.get("function_id_2", "")) or pair.get("function_name_2"),
            "lines2": lines2_display,
            "clone_type": pair.get("clone_type"),
        }
        if lines1_obf:
            entry["obfuscated_lines1"] = lines1_obf
        if lines2_obf:
            entry["obfuscated_lines2"] = lines2_obf
        for metric in metrics:
            value = pair.get(metric)
            if value is not None:
                entry[metric] = value
        json_clone_pairs.append(entry)

    output_data = {
        "summary": summary,
        "clone_pairs": json_clone_pairs,
    }
    json_path.write_text(json.dumps(output_data, ensure_ascii=False, indent=2), encoding="utf-8")


def detect_clones(
    source_path: Path,
    csv_path: Path,
    json_path: Path,
    metadata_csv_path: Optional[Path] = None,
    project_name: str = "gme",
) -> dict:
    functions = extract_functions(source_path)
    for func in functions:
        tokens, normalized, structure = tokenize(func.source)
        func.tokens = tokens
        func.normalized_tokens = normalized
        func.structure_vector = structure
        func.token_length = len(normalized)
        func.normalized_token_set = set(normalized)
        func.token_trigrams = generate_ngrams(normalized, TYPE2_NGRAM_SIZE)
        stmt_count, ctrl_count, sem_count = compute_complexity_metrics(func.source, normalized)
        func.statement_count = stmt_count
        func.control_flow_count = ctrl_count
        func.semantic_token_count = sem_count

    metadata = {}
    body_mapping = {}
    if metadata_csv_path:
        metadata = load_function_metadata_from_csv(metadata_csv_path)
        body_mapping = load_function_body_mapping(metadata_csv_path)
        if metadata:
            apply_metadata(functions, metadata)

    if metadata:
        existing_ids = {f.func_id for f in functions}
        missing_ids = sorted(set(metadata.keys()) - existing_ids)
        start_index = len(functions) + 1
        for offset, mid in enumerate(missing_ids, start=0):
            info = metadata.get(mid, {})
            body = body_mapping.get(mid, "") if body_mapping else ""
            source_text = body or "// empty body placeholder"
            start_line = info.get("start_line")
            end_line = info.get("end_line")
            func = FunctionInfo(
                index=start_index + offset,
                func_id=mid,
                name=info.get("function", f"function_{mid}"),
                return_type=info.get("return_type", ""),
                start_line=start_line if start_line else 0,
                end_line=end_line if end_line else 0,
                source=source_text,
                obfuscated_file=str(source_path),
            )
            tokens, normalized, structure = tokenize(func.source)
            func.tokens = tokens
            func.normalized_tokens = normalized
            func.structure_vector = structure
            func.token_length = len(normalized)
            func.normalized_token_set = set(normalized)
            func.token_trigrams = generate_ngrams(normalized, TYPE2_NGRAM_SIZE)
            stmt_count, ctrl_count, sem_count = compute_complexity_metrics(func.source, normalized)
            func.statement_count = stmt_count
            func.control_flow_count = ctrl_count
            func.semantic_token_count = sem_count
            func.original_name = info.get("function")
            func.original_source = info.get("source")
            func.original_start_line = info.get("start_line")
            func.original_end_line = info.get("end_line")
            functions.append(func)

    clone_pairs, summary = generate_clone_pairs(functions)
    write_csv(clone_pairs, functions, csv_path, body_mapping)
    write_json(functions, clone_pairs, summary, json_path, source_path, project_name)
    return summary

