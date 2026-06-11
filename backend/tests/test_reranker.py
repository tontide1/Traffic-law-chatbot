from backend.core.legal_relevance import ContextCandidate
from backend.core.reranker import NoOpReranker, RerankerConfig, rerank_candidates


def test_noop_reranker_preserves_candidate_order_and_scores():
    reranker = NoOpReranker()
    candidates = [
        ContextCandidate(id="a", text="A", score=0.1),
        ContextCandidate(id="b", text="B", score=0.2),
    ]

    result = reranker.rerank("question", candidates)

    assert result == candidates


def test_rerank_candidates_sorts_by_mock_scores():
    class FakeReranker:
        def rerank(self, question, candidates):
            return [
                ContextCandidate(id="b", text="B", score=0.9),
                ContextCandidate(id="a", text="A", score=0.2),
            ]

    candidates = [
        ContextCandidate(id="a", text="A"),
        ContextCandidate(id="b", text="B"),
    ]

    result = rerank_candidates("question", candidates, FakeReranker())

    assert [item.id for item in result] == ["b", "a"]
    assert result[0].score == 0.9


def test_reranker_config_defaults_to_bge_model():
    config = RerankerConfig()

    assert config.enabled is True
    assert config.model == "BAAI/bge-reranker-v2-m3"
    assert config.max_length == 1024
