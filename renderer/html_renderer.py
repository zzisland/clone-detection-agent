from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from models.data_model import ModelEvaluation


class HtmlRenderer:
    """Simple standalone HTML renderer grouped by confidence layer."""

    def render(self, repo_path: Path, items: List[ModelEvaluation], output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        grouped = self._group_by_layer(items)

        sections = []
        for layer_name, title in [("high", "High Confidence"), ("medium", "Medium Confidence"), ("low", "Low Confidence")]:
            sections.append(f"<h2>{title}</h2>")
            rows = self._render_rows(grouped.get(layer_name, []))
            if not rows:
                sections.append("<p>No results.</p>")
            else:
                sections.append(
                    """
<table>
  <thead>
    <tr>
      <th>#</th>
      <th>Judgement</th>
      <th>Score</th>
      <th>Layer</th>
      <th>Left</th>
      <th>Right</th>
      <th>Method</th>
      <th>Explanation</th>
    </tr>
  </thead>
  <tbody>
    {rows}
  </tbody>
</table>
""".format(rows="".join(rows))
                )

        html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Clone Report</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 24px; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 24px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
    th {{ background: #f6f6f6; text-align: left; }}
    code {{ background: #f2f2f2; padding: 1px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Clone Detection Report</h1>
  <ul>
    <li>Repo: <code>{repo_path}</code></li>
    <li>Generated at (UTC): <code>{created_at}</code></li>
    <li>Total: <code>{len(items)}</code></li>
  </ul>

  {''.join(sections)}
</body>
</html>
"""

        output_path.write_text(html, encoding="utf-8")
        return output_path

    @staticmethod
    def _group_by_layer(items: List[ModelEvaluation]) -> Dict[str, List[ModelEvaluation]]:
        grouped: Dict[str, List[ModelEvaluation]] = defaultdict(list)
        for item in items:
            grouped[item.item.layer].append(item)
        return grouped

    @staticmethod
    def _render_rows(items: List[ModelEvaluation]) -> List[str]:
        rows = []
        for idx, ev in enumerate(items, start=1):
            c = ev.item.candidate
            rows.append(
                """
<tr>
  <td>{idx}</td>
  <td>{judgement}</td>
  <td>{score:.2f}</td>
  <td>{layer}</td>
  <td><code>{file1}</code>:{s1}-{e1}</td>
  <td><code>{file2}</code>:{s2}-{e2}</td>
  <td><code>{method}</code></td>
  <td>{explain}</td>
</tr>
""".format(
                    idx=idx,
                    judgement=ev.judgement.value,
                    score=ev.score,
                    layer=ev.item.layer,
                    file1=c.left.file_path,
                    s1=c.left.start_line,
                    e1=c.left.end_line,
                    file2=c.right.file_path,
                    s2=c.right.start_line,
                    e2=c.right.end_line,
                    method=c.source_method,
                    explain=ev.explanation,
                )
            )
        return rows
