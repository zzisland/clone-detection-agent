"""
模块配置 - 存储每个模块的隐秘化配置
"""
from typing import Set


class HeaderInfo:
    """Lightweight placeholder header info for obfuscation pipeline."""

    def get_types(self) -> Set[str]:
        return set()

    def get_constants(self) -> Set[str]:
        return set()

    def get_enums(self) -> Set[str]:
        return set()

    def get_functions(self) -> Set[str]:
        return set()


class ModuleConfig:
    """模块配置类，存储白名单、函数列表等"""

    def __init__(self, module_name: str):
        self.module_name = module_name
        self.whitelist: Set[str] = set()
        self.global_functions: Set[str] = set()
        self.member_names: Set[str] = set()
        self.types: Set[str] = set()
        self._init_defaults()

    def _init_defaults(self) -> None:
        cpp_keywords = {
            "if", "else", "for", "while", "do", "switch", "case", "default",
            "break", "continue", "return", "goto", "sizeof", "new", "delete",
            "try", "catch", "throw", "class", "struct", "union", "enum",
            "template", "typename", "using", "namespace", "public", "private",
            "protected", "virtual", "override", "final", "static", "inline",
            "constexpr", "const", "volatile", "register", "extern", "friend",
            "operator", "this", "nullptr", "true", "false", "NULL", "auto"
        }
        self.whitelist.update(cpp_keywords)

        std_types = {
            "std", "std::", "size_t", "double", "float", "int", "long",
            "short", "char", "bool", "void", "string", "wstring", "clock_t", "FILE"
        }
        self.whitelist.update(std_types)

        std_functions = {
            "fabs", "pow", "isnormal", "strcpy", "memcpy", "fputs", "itoa",
            "clock", "to_string", "perpendicular", "parallel"
        }
        self.global_functions.update(std_functions)

        custom_macros = {
            "CUSTOM_NEW", "CUSTOM_DELETE", "GME_DELETE", "GME_NEW",
            "ACIS_DELETE", "ACIS_NEW",
            "RET_ALIAS", "IN_ALIAS", "STD_CAST", "__GME_STD_CAST"
        }
        self.whitelist.update(custom_macros)

        common_members = {
            "x", "y", "z", "r", "g", "b", "width", "height", "length",
            "first", "second", "min", "max", "left", "right", "top", "bottom",
            "size", "count", "data", "ptr", "value", "result"
        }
        self.member_names.update(common_members)

    def learn_from_headers(self, header_info: HeaderInfo) -> None:
        if header_info is None:
            return
        self.whitelist.update(header_info.get_types())
        self.whitelist.update(header_info.get_constants())
        self.whitelist.update(header_info.get_enums())
        self.types.update(header_info.get_types())
        self.global_functions.update(header_info.get_functions())

    def add_to_whitelist(self, *items: str) -> None:
        self.whitelist.update(items)

    def add_global_function(self, *functions: str) -> None:
        self.global_functions.update(functions)

    def add_member_name(self, *members: str) -> None:
        self.member_names.update(members)

    def add_type(self, *type_names: str) -> None:
        self.types.update(type_names)
        self.whitelist.update(type_names)

    def is_in_whitelist(self, identifier: str) -> bool:
        return identifier in self.whitelist

    def is_global_function(self, name: str) -> bool:
        return name in self.global_functions

    def is_member_name(self, name: str) -> bool:
        return name in self.member_names

    def is_type(self, name: str) -> bool:
        return name in self.types

    def __str__(self) -> str:
        return f"ModuleConfig[{self.module_name}]: {len(self.whitelist)} whitelist, {len(self.global_functions)} functions, {len(self.member_names)} members"
