from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional


class ConfidenceLabel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True)
class CodeLocation:
    file_path: str
    start_line: int
    end_line: int


@dataclass(frozen=True)
class CloneCandidate:
    """A candidate clone pair reported by a detector."""

    left: CodeLocation
    right: CodeLocation
    similarity: float
    source_method: str
    extra: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class LayeredClone:
    """Candidate after layering (e.g., intersection/union buckets)."""

    candidate: CloneCandidate
    layer: str  # e.g. high / medium / low
    reason: str
    layerer_name: str = "intersection_union_layerer"


@dataclass(frozen=True)
class ModelJudgement(str, Enum):
    CLONE = "clone"
    NOT_CLONE = "not_clone"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class ModelEvaluation:
    item: LayeredClone
    judgement: ModelJudgement
    score: float  # 0..1
    explanation: str
    model_name: str = "mock-llm"
    raw: Optional[dict] = None
