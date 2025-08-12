from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple
import re

try:
    from transformers import pipeline
    _TRANSFORMERS_AVAILABLE = True
except Exception:
    _TRANSFORMERS_AVAILABLE = False

# --- Regex patterns for light normalization ---
_ELONG = re.compile(r"(.)\1{2,}")        # Collapse char repetitions: hellooo -> helloo
_PUNCT_RUN = re.compile(r"([!?.,])\1{2,}")  # Collapse punctuation runs: !!! -> !!
_WS = re.compile(r"\s+")                 # Normalize whitespace


def normalize_text(t: str) -> str:
    """Normalize noisy text: collapse elongations, punctuation runs, and extra spaces."""
    if not t:
        return ""
    t = t.strip()
    t = _ELONG.sub(r"\1\1", t)
    t = _PUNCT_RUN.sub(r"\1\1", t)
    t = _WS.sub(" ", t)
    return t


@dataclass
class SentimentResult:
    """Single review sentiment classification."""
    label: str        # "positive" | "neutral" | "negative"
    confidence: float


class SentimentModel:
    """
    Sentiment classification with rating-based shortcut and optional transformer fallback.

    Logic:
      - If rating available: map stars → sentiment (1–2→negative, 3→neutral, 4–5→positive)
      - Else if transformer available: use model output, map into neutral band
      - Else: return "neutral" with 0 confidence
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
        neutral_band: Tuple[float, float] = (0.4, 0.6),
    ) -> None:
        self.model_name = model_name
        self.neutral_band = neutral_band
        self._pipe = None  # lazy initialization

    def _ensure_pipe(self):
        """Lazily load sentiment model pipeline."""
        if self._pipe is None and _TRANSFORMERS_AVAILABLE:
            self._pipe = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                top_k=None
            )

    def analyze_one(self, text: str, rating: Optional[int] = None) -> SentimentResult:
        """
        Analyze a single review using:
          1. Rating shortcut (if available)
          2. Transformer model (if installed)
          3. Neutral fallback
        """
        # Rating-based shortcut
        if isinstance(rating, int) and rating in (1, 2, 3, 4, 5):
            if rating <= 2:
                return SentimentResult("negative", 1.0)
            if rating == 3:
                return SentimentResult("neutral", 1.0)
            return SentimentResult("positive", 1.0)

        # Transformer fallback
        t = normalize_text(text or "")
        if not t:
            return SentimentResult("neutral", 0.0)

        if _TRANSFORMERS_AVAILABLE:
            self._ensure_pipe()
            if self._pipe is not None:
                out = self._pipe(t, truncation=True)[0]
                label = out["label"].lower()
                score = float(out["score"])
                if self.neutral_band[0] <= score <= self.neutral_band[1]:
                    return SentimentResult("neutral", score)
                return SentimentResult("positive" if label == "positive" else "negative", score)

        # Default: no model available
        return SentimentResult("neutral", 0.0)

    def analyze_batch(
        self,
        texts: Iterable[str],
        ratings: Optional[Iterable[Optional[int]]] = None
    ) -> List[SentimentResult]:
        """
        Batch analyze multiple reviews, using the same rating/model fallback logic as analyze_one().
        """
        texts = list(texts)
        ratings = list(ratings or [None] * len(texts))

        results: List[Optional[SentimentResult]] = [None] * len(texts)
        no_rating_idxs = [i for i, r in enumerate(ratings) if r not in (1, 2, 3, 4, 5)]

        # Fill rating-based results
        for i, r in enumerate(ratings):
            if r in (1, 2, 3, 4, 5):
                results[i] = self.analyze_one(texts[i], r)

        # Batch model inference
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
                        results[idx] = SentimentResult(
                            "positive" if label == "positive" else "negative", score
                        )

        # Fill any remaining gaps with neutral fallback
        for i in no_rating_idxs:
            if results[i] is None:
                results[i] = self.analyze_one(texts[i], None)

        return results
