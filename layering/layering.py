from __future__ import annotations

from dataclasses import dataclass
from typing import List

from models.data_model import CloneCandidate, LayeredClone


@dataclass(frozen=True)
class LayeringConfig:
    high_threshold: float = 0.8
    medium_threshold: float = 0.5


class CloneLayerer:
    """Intersection/union-based layerer.

    Rules:
    - high: detected by both type12 and type34
    - medium: detected by exactly one method and similarity is not too low
    - low: anything else
    """

    def __init__(self, config: LayeringConfig | None = None):
        self.config = config or LayeringConfig()

    def layer(self, candidates: List[CloneCandidate]) -> List[LayeredClone]:
        out: List[LayeredClone] = []
        for c in candidates:
            from_type12 = self._is_true(c.extra.get("from_type12"))
            from_type34 = self._is_true(c.extra.get("from_type34"))
            sim = float(c.similarity)

            if from_type12 and from_type34:
                layer = "high"
                reason = "detected by both type12 and type34 (intersection)"
            elif from_type12 or from_type34:
                if sim >= self.config.medium_threshold:
                    layer = "medium"
                    reason = self._single_source_reason(from_type12, from_type34)
                else:
                    layer = "low"
                    reason = f"single-method candidate but similarity < {self.config.medium_threshold}"
            elif sim >= self.config.high_threshold:
                layer = "high"
                reason = f"fallback high similarity >= {self.config.high_threshold}"
            elif sim >= self.config.medium_threshold:
                layer = "medium"
                reason = f"fallback medium similarity >= {self.config.medium_threshold}"
            else:
                layer = "low"
                reason = f"not in union set and similarity < {self.config.medium_threshold}"

            out.append(LayeredClone(candidate=c, layer=layer, reason=reason))
        return out

    @staticmethod
    def _is_true(value: object) -> bool:
        if value is None:
            return False
        return str(value).strip().lower() in {"1", "true", "yes", "y"}

    @staticmethod
    def _single_source_reason(from_type12: bool, from_type34: bool) -> str:
        if from_type12 and not from_type34:
            return "detected only by type12 (union minus intersection)"
        if from_type34 and not from_type12:
            return "detected only by type34 (union minus intersection)"
        return "detected by a single method"
