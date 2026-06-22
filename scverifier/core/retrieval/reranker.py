import logging
from typing import List

from scverifier.config.settings import Config
from scverifier.data.models import Proposition

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Cross-encoder reranker for semantic re-scoring of propositions."""

    def __init__(self):
        self.model = None  # Lazy-Loading

    def _load_model(self) -> None:
        import torch
        from sentence_transformers import CrossEncoder

        device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            self.model = CrossEncoder(Config.RERANK_MODEL, device=device)
            logger.info("Loaded reranker %s on device %s", Config.RERANK_MODEL, device)
        except Exception as e:
            self.model = None
            logger.error(
                "Failed to load model %s on device %s. With error: %s",
                Config.RERANK_MODEL,
                device,
                e,
            )

    def warm_up(self) -> None:
        if self._model is None:
            self._load_model()
        if self._model:
            self._model.predict([("warm up", "warm up")])

    def rerank(self, query: str, propositions: List[Proposition], top_k: int = 15) -> List[Proposition]:

        if self.model is None:
            self._load_model()
        if self.model is None:
            return propositions

        if not propositions:
            return propositions

        pairs = [(query, p.text) for p in propositions]
        scores = self.model.predict(pairs)

        scored = list(zip(scores, propositions))
        scored.sort(key=lambda x: x[0], reverse=True)

        return [p for _, p in scored[:top_k]]
