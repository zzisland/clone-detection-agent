"""
C++模块化代码隐秘化器 - minimal agent port.
Restores FUN_x style obfuscation expected by clone_detector.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List

from .module_config import ModuleConfig


@dataclass
class FunctionInfo:
    normalized_signature: str
    original_name: str
    token_name: str
    param_count: int


class CppModuleObfuscator:
    CPP_KEYWORDS = {
        "if", "else", "for", "while", "do", "switch", "case", "default",
        "break", "continue", "return", "goto", "sizeof", "new", "delete",
        "try", "catch", "throw", "class", "struct", "union", "enum",
        "template", "typename", "using", "namespace", "public", "private",
        "protected", "virtual", "override", "final", "static", "inline",
        "constexpr", "const", "volatile", "register", "extern", "friend",
        "operator", "this", "nullptr", "true", "false", "NULL", "auto"
    }

    def __init__(self, config: ModuleConfig, enable_slicing: bool = False):
        self.config = config
        self.enable_slicing = enable_slicing
        self.function_map: Dict[str, FunctionInfo] = {}
        self.function_name_map: Dict[str, List[FunctionInfo]] = {}
        self.func_idx = 0

    def generate(self, functions: List[dict], enable_slicing: bool = False, enable_analysis: bool = False) -> str:
        if not functions:
            return "// No functions"

        prepared = []
        self.function_map.clear()
        self.function_name_map.clear()
        self.func_idx = 0

        for idx, func in enumerate(functions, start=1):
            signature = str(func.get("name") or "").strip()
            body = str(func.get("body") or "")
            if not signature:
                signature = f"func_{idx}()"
            token_name = f"FUN_{self.func_idx}"
            self.func_idx += 1

            original_name = self._extract_function_name(signature) or f"func_{idx}"
            normalized_signature = self._normalize_signature(signature)
            info = FunctionInfo(
                normalized_signature=normalized_signature,
                original_name=original_name,
                token_name=token_name,
                param_count=self._count_parameters(signature),
            )
            self.function_map[normalized_signature] = info
            self.function_name_map.setdefault(original_name, []).append(info)
            prepared.append((idx, signature, body, info))

        output_lines: List[str] = []
        for idx, signature, body, info in prepared:
            obf_signature = self._replace_function_name(signature, info)
            obf_body = self._replace_function_calls(body)
            output_lines.append(f"// Function {idx:03d}")
            output_lines.append(f"{obf_signature} {{")
            if obf_body.strip():
                output_lines.extend(self._indent_body(obf_body))
            output_lines.append("}")
            output_lines.append("")

        return "\n".join(output_lines).rstrip() + "\n"

    def _indent_body(self, body: str) -> List[str]:
        lines = body.splitlines()
        return [f"    {line}" if line.strip() else "" for line in lines]

    def _normalize_signature(self, signature: str) -> str:
        return " ".join(signature.strip().split())

    def _count_parameters(self, signature: str) -> int:
        m = re.search(r"\((.*)\)", signature)
        if not m:
            return 0
        params = m.group(1).strip()
        if not params or params == "void":
            return 0
        return len([p for p in params.split(",") if p.strip()])

    def _extract_function_name(self, signature: str) -> str:
        sig = signature.strip()
        paren = sig.find("(")
        if paren == -1:
            return ""
        prefix = sig[:paren].strip()
        if not prefix:
            return ""
        name = prefix.split()[-1]
        return name.split("::")[-1]

    def _replace_function_name(self, signature: str, info: FunctionInfo) -> str:
        original = info.original_name
        token = info.token_name
        pattern = re.compile(rf"(?<!\w){re.escape(original)}(?=\s*\()")
        replaced = pattern.sub(token, signature, count=1)

        scope_pattern = re.compile(rf"(?<!\w){re.escape(original)}(?=::\s*{re.escape(token)}\s*\()")
        replaced = scope_pattern.sub(token, replaced, count=1)
        return replaced

    def _replace_function_calls(self, body: str) -> str:
        replaced = body
        all_infos = [info for infos in self.function_name_map.values() for info in infos]
        all_infos.sort(key=lambda x: len(x.original_name), reverse=True)
        for info in all_infos:
            if info.original_name in self.CPP_KEYWORDS:
                continue
            pattern = re.compile(rf"(?<!\w){re.escape(info.original_name)}(?=\s*\()")
            replaced = pattern.sub(info.token_name, replaced)
        return replaced
