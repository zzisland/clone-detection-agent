from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from models.data_model import ModelEvaluation


class MarkdownReporter:
    def generate(self, repo_path: Path, evaluated: List[ModelEvaluation], output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

        judgement_counts = Counter([e.judgement.value for e in evaluated])
        layer_counts = Counter([e.item.layer for e in evaluated])
        grouped = self._group_by_layer(evaluated)

        lines: List[str] = []
        lines.append("# Clone Detection Report")
        lines.append("")
        lines.append("## Basic Info")
        lines.append("")
        lines.append(f"- Repo: `{repo_path}`")
        lines.append(f"- Generated at (UTC): `{created_at}`")
        lines.append(f"- Total: `{len(evaluated)}`")
        lines.append("")

        lines.append("## Summary")
        lines.append("")
        lines.append("### By Model Judgement")
        for k in ["clone", "uncertain", "not_clone"]:
            lines.append(f"- {k}: `{judgement_counts.get(k, 0)}`")
        lines.append("")

        lines.append("### By Layer")
        for k in ["high", "medium", "low"]:
            lines.append(f"- {k}: `{layer_counts.get(k, 0)}`")
        lines.append("")

        for layer_name, title in [("high", "High Confidence"), ("medium", "Medium Confidence"), ("low", "Low Confidence")]:
            lines.append(f"## {title} Results")
            lines.append("")
            items = grouped.get(layer_name, [])
            if not items:
                lines.append("No results.")
                lines.append("")
                continue

            for idx, ev in enumerate(items, start=1):
                lines.extend(self._render_item(idx, ev))

        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return output_path

    @staticmethod
    def _group_by_layer(evaluated: List[ModelEvaluation]) -> Dict[str, List[ModelEvaluation]]:
        grouped: Dict[str, List[ModelEvaluation]] = defaultdict(list)
        for item in evaluated:
            grouped[item.item.layer].append(item)
        return grouped

    @staticmethod
    def _render_item(idx: int, ev: ModelEvaluation) -> List[str]:
        c = ev.item.candidate
        lines: List[str] = []
        lines.append(f"### {idx}. {ev.judgement.value.upper()} (score={ev.score:.2f}, similarity={c.similarity:.2f})")
        lines.append("")
        lines.append(f"- Layer: `{ev.item.layer}`")
        lines.append(f"- Layer reason: `{ev.item.reason}`")
        lines.append(f"- Detection method: `{c.source_method}`")
        lines.append(f"- Model: `{ev.model_name}`")
        lines.append(f"- Explanation: {ev.explanation}")
        lines.append("- Left:")
        lines.append(f"  - File: `{c.left.file_path}`")
        lines.append(f"  - Lines: `{c.left.start_line}-{c.left.end_line}`")
        lines.append("- Right:")
        lines.append(f"  - File: `{c.right.file_path}`")
        lines.append(f"  - Lines: `{c.right.start_line}-{c.right.end_line}`")
        if c.extra:
            lines.append("- Extra:")
            for k, v in sorted(c.extra.items()):
                lines.append(f"  - {k}: `{v}`")
        lines.append("")
        return lines
