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


def test_bge_reranker_calls_flag_reranker_and_sorts_correctly():
    from unittest.mock import MagicMock, patch
    from backend.core.reranker import BGEReranker

    config = RerankerConfig(enabled=True, model="fake-model", device="cpu")
    candidates = [
        ContextCandidate(id="a", text="A"),
        ContextCandidate(id="b", text="B"),
    ]

    mock_flag_reranker_instance = MagicMock()
    mock_flag_reranker_instance.compute_score.return_value = [0.3, 0.7]

    with patch("FlagEmbedding.FlagReranker", return_value=mock_flag_reranker_instance) as mock_class:
        reranker = BGEReranker(config)
        result = reranker.rerank("question", candidates)

        mock_class.assert_called_once_with("fake-model", use_fp16=False, devices="cpu")
        mock_flag_reranker_instance.compute_score.assert_called_once_with(
            [["question", "A"], ["question", "B"]],
            normalize=True,
            max_length=1024,
        )

        assert [item.id for item in result] == ["b", "a"]
        assert result[0].score == 0.7
        assert result[1].score == 0.3


def test_build_reranker_returns_noop_when_disabled(monkeypatch):
    from backend.config import settings
    from backend.core.reranker import build_reranker

    monkeypatch.setattr(settings, "RERANKER_ENABLED", False)

    reranker = build_reranker()
    assert isinstance(reranker, NoOpReranker)


def test_build_reranker_returns_bge_when_enabled(monkeypatch):
    from backend.config import settings
    from backend.core.reranker import build_reranker, BGEReranker

    monkeypatch.setattr(settings, "RERANKER_ENABLED", True)
    monkeypatch.setattr(settings, "RERANKER_MODEL", "test-model")

    reranker = build_reranker()
    assert isinstance(reranker, BGEReranker)
    assert reranker.config.model == "test-model"
