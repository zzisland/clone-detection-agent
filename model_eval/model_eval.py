from __future__ import annotations

from dataclasses import dataclass
from typing import List

from models.data_model import LayeredClone, ModelEvaluation, ModelJudgement


@dataclass(frozen=True)
class ModelEvalConfig:
    mode: str = "mock"  # mock | openai (future)
    model_name: str = "mock-llm"


class CloneModelEvaluator:
    """MVP model evaluator.

    This is a placeholder. Later you can call an LLM API here.
    """

    def __init__(self, config: ModelEvalConfig | None = None):
        self.config = config or ModelEvalConfig()

    def evaluate(self, items: List[LayeredClone]) -> List[ModelEvaluation]:
        if self.config.mode != "mock":
            raise ValueError(f"Unsupported model eval mode: {self.config.mode}")

        out: List[ModelEvaluation] = []
        for it in items:
            layer = it.layer
            if layer == "high":
                judgement = ModelJudgement.CLONE
                score = 0.9
                explanation = "High similarity; likely clone (mock judgement)."
            elif layer == "medium":
                judgement = ModelJudgement.UNCERTAIN
                score = 0.55
                explanation = "Medium similarity; needs human/LLM review (mock judgement)."
            else:
                judgement = ModelJudgement.NOT_CLONE
                score = 0.2
                explanation = "Low similarity; likely not a clone (mock judgement)."

            out.append(
                ModelEvaluation(
                    item=it,
                    judgement=judgement,
                    score=score,
                    explanation=explanation,
                    model_name=self.config.model_name,
                )
            )

        return out
