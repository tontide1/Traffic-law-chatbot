"""Reranker wrappers for legal context candidates."""

from dataclasses import dataclass
from typing import Protocol

from backend.config import settings
from backend.core.legal_relevance import ContextCandidate


@dataclass(frozen=True)
class RerankerConfig:
    enabled: bool = True
    model: str = "BAAI/bge-reranker-v2-m3"
    max_length: int = 1024
    device: str = "auto"


class Reranker(Protocol):
    def rerank(self, question: str, candidates: list[ContextCandidate]) -> list[ContextCandidate]:
        ...


class NoOpReranker:
    def rerank(self, question: str, candidates: list[ContextCandidate]) -> list[ContextCandidate]:
        return candidates


class BGEReranker:
    def __init__(self, config: RerankerConfig):
        self.config = config
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from FlagEmbedding import FlagReranker
            except ImportError as exc:
                raise RuntimeError(
                    "FlagEmbedding is required for BAAI/bge-reranker-v2-m3. "
                    "Install backend requirements or set RERANKER_ENABLED=false."
                ) from exc

            use_fp16 = self.config.device != "cpu"
            devices = None if self.config.device == "auto" else self.config.device
            self._model = FlagReranker(self.config.model, use_fp16=use_fp16, devices=devices)
        return self._model

    def rerank(self, question: str, candidates: list[ContextCandidate]) -> list[ContextCandidate]:
        if not candidates:
            return []

        model = self._load_model()
        pairs = [
            [question, candidate.text[: self.config.max_length * 4]]
            for candidate in candidates
        ]
        scores = model.compute_score(
            pairs,
            normalize=True,
            max_length=self.config.max_length,
        )
        if isinstance(scores, float):
            scores = [scores]

        rescored = [
            ContextCandidate(
                id=candidate.id,
                text=candidate.text,
                source=candidate.source,
                score=float(score),
                label=candidate.label,
            )
            for candidate, score in zip(candidates, scores, strict=True)
        ]
        return sorted(rescored, key=lambda item: item.score, reverse=True)


def build_reranker() -> Reranker:
    if not settings.RERANKER_ENABLED:
        return NoOpReranker()
    return BGEReranker(
        RerankerConfig(
            enabled=settings.RERANKER_ENABLED,
            model=settings.RERANKER_MODEL,
            max_length=settings.RERANKER_MAX_LENGTH,
            device=settings.RERANKER_DEVICE,
        )
    )


def rerank_candidates(
    question: str,
    candidates: list[ContextCandidate],
    reranker: Reranker,
) -> list[ContextCandidate]:
    return reranker.rerank(question, candidates)
