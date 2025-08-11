from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import re

try:
    from transformers import pipeline

    _TRANSFORMERS_AVAILABLE = True
except Exception:
    _TRANSFORMERS_AVAILABLE = False

# Simple normalizer for noisy app-store text
_ELONG = re.compile(r"(.)\1{2,}")
_PUNCT_RUN = re.compile(r"([!?.,])\1{2,}")
_WS = re.compile(r"\s+")


def normalize_text(t: str) -> str:
    if not t:
        return ""
    t = t.strip()
    t = _ELONG.sub(r"\1\1", t)
    t = _PUNCT_RUN.sub(r"\1\1", t)
    t = _WS.sub(" ", t)
    return t


@dataclass
class SentimentResult:
    label: str  # "positive" | "neutral" | "negative"
    confidence: float  # 0..1 (model-based when available, else 1.0 for rating-based)


class SentimentModel:
    """
    Rating-aware sentiment with transformer fallback.
    - If rating provided: map 1-2->neg, 3->neutral, 4-5->pos (confidence=1.0)
    - Else: use a transformer pipeline if available, else VADER-like thresholds.
    """

    def __init__(
            self,
            model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
            neutral_band: Tuple[float, float] = (0.4, 0.6),  # scores in [0.4,0.6] => neutral
    ) -> None:
        self.model_name = model_name
        self.neutral_band = neutral_band
        self._pipe = None  # lazy-load

    def _ensure_pipe(self):
        if self._pipe is None and _TRANSFORMERS_AVAILABLE:
            # binary sentiment: POSITIVE/NEGATIVE
            self._pipe = pipeline("sentiment-analysis", model=self.model_name, top_k=None)

    def analyze_one(self, text: str, rating: Optional[int] = None) -> SentimentResult:
        # 1) rating first
        if isinstance(rating, int) and rating in (1, 2, 3, 4, 5):
            if rating <= 2:
                return SentimentResult("negative", 1.0)
            if rating == 3:
                return SentimentResult("neutral", 1.0)
            return SentimentResult("positive", 1.0)

        # 2) model fallback
        t = normalize_text(text or "")
        if not t:
            return SentimentResult("neutral", 0.0)

        if _TRANSFORMERS_AVAILABLE:
            self._ensure_pipe()
            if self._pipe is not None:
                out = self._pipe(t, truncation=True)[0]
                # distilbert sst-2 returns {'label': 'POSITIVE'|'NEGATIVE', 'score': prob}
                label = out["label"].lower()
                score = float(out["score"])
                # map to neutral band
                if self.neutral_band[0] <= score <= self.neutral_band[1]:
                    return SentimentResult("neutral", score)
                if label == "positive":
                    return SentimentResult("positive", score)
                else:
                    return SentimentResult("negative", score)

    def analyze_batch(
            self,
            texts: Iterable[str],
            ratings: Optional[Iterable[Optional[int]]] = None
    ) -> List[SentimentResult]:
        texts = list(texts)
        if ratings is None:
            ratings = [None] * len(texts)
        ratings = list(ratings)

        # If any has rating -> we’ll do per-item fast mapping; but we can still
        # batch model inference for the no-rating items.
        no_rating_idxs = [i for i, r in enumerate(ratings) if r not in (1, 2, 3, 4, 5)]
        results: List[Optional[SentimentResult]] = [None] * len(texts)

        # Fill rating-based ones
        for i, r in enumerate(ratings):
            if r in (1, 2, 3, 4, 5):
                results[i] = self.analyze_one(texts[i], r)

        # Batch infer remaining with model if available
        to_texts = [normalize_text(texts[i] or "") for i in no_rating_idxs]
        if to_texts and _TRANSFORMERS_AVAILABLE:
            self._ensure_pipe()
            if self._pipe is not None:
                outs = self._pipe(to_texts, truncation=True)
                for idx, out in zip(no_rating_idxs, outs):
                    label = out["label"].lower()
                    score = float(out["score"])
                    if self.neutral_band[0] <= score <= self.neutral_band[1]:
                        results[idx] = SentimentResult("neutral", score)
                    else:
                        results[idx] = SentimentResult("positive" if label == "positive" else "negative", score)

        # Fallback for any remaining (no model)
        for i in no_rating_idxs:
            if results[i] is None:
                results[i] = self.analyze_one(texts[i], None)

        return results
