from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, List

from models.data_model import LayeredClone, ModelEvaluation, ModelJudgement


@dataclass(frozen=True)
class ModelEvalConfig:
    mode: str = "openai"
    model_name: str = ""
    api_url: str = ""
    api_key: str = ""
    temperature: float = 0.0
    timeout_seconds: int = 120
    max_body_chars: int = 4000


class CloneModelEvaluator:
    """Clone-pair evaluator backed by an OpenAI-compatible chat API."""

    def __init__(self, config: ModelEvalConfig | None = None):
        self.config = config or ModelEvalConfig()

    def evaluate(self, items: List[LayeredClone]) -> List[ModelEvaluation]:
        if self.config.mode == "openai":
            return self._evaluate_openai(items)
        raise ValueError(f"Unsupported model eval mode: {self.config.mode}")

    def _evaluate_openai(self, items: List[LayeredClone]) -> List[ModelEvaluation]:
        try:
            import requests  # type: ignore
        except Exception as e:
            raise ImportError(f"Package 'requests' is required for model eval API mode: {e}")

        if not self.config.api_url:
            raise ValueError("Model eval API mode requires --model-eval-api-url")
        if not self.config.model_name:
            raise ValueError("Model eval API mode requires --model-eval-model-name")

        chat_url = self._normalize_api_base(self.config.api_url) + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = (
                self.config.api_key
                if self.config.api_key.lower().startswith("bearer ")
                else f"Bearer {self.config.api_key}"
            )

        out: List[ModelEvaluation] = []
        total = len(items)
        for idx, item in enumerate(items, start=1):
            try:
                parsed, raw = self._request_with_fallbacks(
                    requests=requests,
                    chat_url=chat_url,
                    headers=headers,
                    item=item,
                )
                out.append(self._to_model_evaluation(item, parsed, raw=raw))
                print(f"  - [model-eval] {idx}/{total} ok", flush=True)
            except Exception as e:
                raise RuntimeError(
                    f"Model eval API request failed for item {idx}/{total} via {chat_url}: {e}"
                ) from e

        return out

    def _request_with_fallbacks(
        self,
        requests: Any,
        chat_url: str,
        headers: dict[str, str],
        item: LayeredClone,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        prompt_variants = [
            self._build_user_prompt(item, body_limit=self.config.max_body_chars),
            self._build_user_prompt(item, body_limit=max(1200, self.config.max_body_chars // 2)),
            self._build_user_prompt(item, body_limit=800),
        ]

        attempts = [
            {"response_format": True, "prompt": prompt_variants[0], "label": "json-format/full"},
            {"response_format": False, "prompt": prompt_variants[0], "label": "plain/full"},
            {"response_format": False, "prompt": prompt_variants[1], "label": "plain/half"},
            {"response_format": False, "prompt": prompt_variants[2], "label": "plain/small"},
        ]

        errors: list[str] = []
        for attempt in attempts:
            payload = self._build_payload(
                prompt=attempt["prompt"],
                use_response_format=bool(attempt["response_format"]),
            )
            try:
                resp = requests.post(chat_url, headers=headers, json=payload, timeout=self.config.timeout_seconds)
                if resp.status_code >= 400:
                    body = self._safe_response_text(resp)
                    errors.append(f"{attempt['label']} -> HTTP {resp.status_code}: {body}")
                    continue

                raw = resp.json()
                content = self._extract_message_content(raw)
                parsed = self._parse_model_response(content)
                return parsed, raw
            except Exception as e:
                errors.append(f"{attempt['label']} -> {e}")

        raise RuntimeError(" ; ".join(errors))

    def _build_payload(self, prompt: str, use_response_format: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.config.model_name,
            "temperature": self.config.temperature,
            "messages": [
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": prompt},
            ],
        }
        if use_response_format:
            payload["response_format"] = {"type": "json_object"}
        return payload

    @staticmethod
    def _normalize_api_base(api_url: str) -> str:
        api_base = api_url.rstrip("/")
        if api_base.endswith("/chat/completions"):
            api_base = api_base[: -len("/chat/completions")]
        if not api_base.endswith("/v1"):
            api_base = f"{api_base}/v1"
        return api_base

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You are evaluating whether two code fragments are clones. "
            "Return strict JSON only with keys judgement, score, explanation. "
            "judgement must be one of: clone, uncertain, not_clone. "
            "score must be a number between 0 and 1. "
            "explanation should be concise and mention the strongest reason."
        )

    def _build_user_prompt(self, item: LayeredClone, body_limit: int | None = None) -> str:
        c = item.candidate
        left_body = self._truncate_text(str(c.extra.get("func1_body") or ""), body_limit)
        right_body = self._truncate_text(str(c.extra.get("func2_body") or ""), body_limit)

        return (
            "Evaluate this clone candidate.\n\n"
            "Metadata:\n"
            f"- Layer: {item.layer}\n"
            f"- Layer reason: {item.reason}\n"
            f"- Detector method: {c.source_method}\n"
            f"- Similarity: {c.similarity:.4f}\n"
            f"- Left file: {c.left.file_path}:{c.left.start_line}-{c.left.end_line}\n"
            f"- Right file: {c.right.file_path}:{c.right.start_line}-{c.right.end_line}\n"
            f"- Type1-2 similarity: {c.extra.get('type12_similarity', '')}\n"
            f"- Type3-4 similarity: {c.extra.get('type34_similarity', '')}\n"
            f"- Type1-2 clone type: {c.extra.get('type12_clone_type', '')}\n"
            f"- Clone type label: {c.extra.get('clone_type_label', c.extra.get('Clone_Type_Label', ''))}\n\n"
            "Left function body:\n"
            f"```cpp\n{left_body}\n```\n\n"
            "Right function body:\n"
            f"```cpp\n{right_body}\n```\n\n"
            "Decision policy:\n"
            "- Use clone when structure and intent are clearly equivalent, even with renaming or light edits.\n"
            "- Use uncertain when signals conflict, bodies are truncated, or similarity is ambiguous.\n"
            "- Use not_clone when overlap is shallow or mostly incidental.\n"
        )

    def _truncate_text(self, text: str, body_limit: int | None = None) -> str:
        clean = text.strip()
        limit = body_limit if body_limit is not None else self.config.max_body_chars
        if not clean:
            return "// source body unavailable"
        if len(clean) <= limit:
            return clean
        return clean[:limit] + "\n// ... truncated ..."

    @staticmethod
    def _safe_response_text(resp: Any, max_chars: int = 500) -> str:
        try:
            text = (resp.text or "").strip()
        except Exception:
            return "<unable to read response body>"
        if not text:
            return "<empty response body>"
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "...<truncated>"

    @staticmethod
    def _extract_message_content(raw: dict[str, Any]) -> str:
        choices = raw.get("choices") or []
        if not choices:
            raise ValueError("No choices returned by model API")

        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts: list[str] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(str(part.get("text") or ""))
            if text_parts:
                return "\n".join(text_parts)
        raise ValueError("Model API response did not contain text content")

    def _parse_model_response(self, content: str) -> dict[str, Any]:
        text = content.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3:
                text = "\n".join(lines[1:-1]).strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise ValueError(f"Model response is not valid JSON: {content}")
            data = json.loads(text[start : end + 1])

        if not isinstance(data, dict):
            raise ValueError(f"Model response JSON must be an object: {content}")
        return data

    def _to_model_evaluation(
        self,
        item: LayeredClone,
        parsed: dict[str, Any],
        raw: dict[str, Any] | None = None,
    ) -> ModelEvaluation:
        judgement = self._normalize_judgement(parsed.get("judgement"))
        score = self._normalize_score(parsed.get("score"))
        explanation = str(parsed.get("explanation") or "").strip() or "No explanation returned by model."
        return ModelEvaluation(
            item=item,
            judgement=judgement,
            score=score,
            explanation=explanation,
            model_name=self.config.model_name,
            raw=raw,
        )

    @staticmethod
    def _normalize_judgement(value: Any) -> ModelJudgement:
        normalized = str(value or "").strip().lower()
        mapping = {
            "clone": ModelJudgement.CLONE,
            "not_clone": ModelJudgement.NOT_CLONE,
            "not-clone": ModelJudgement.NOT_CLONE,
            "not clone": ModelJudgement.NOT_CLONE,
            "uncertain": ModelJudgement.UNCERTAIN,
        }
        if normalized not in mapping:
            raise ValueError(f"Unsupported judgement returned by model: {value}")
        return mapping[normalized]

    @staticmethod
    def _normalize_score(value: Any) -> float:
        try:
            score = float(value)
        except Exception as e:
            raise ValueError(f"Invalid model score: {value}") from e
        return max(0.0, min(1.0, score))
