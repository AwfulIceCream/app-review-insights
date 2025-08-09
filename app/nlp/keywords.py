import re
from collections import Counter
from typing import List, Iterable
from itertools import tee

# General stopwords + domain-specific noise
_STOPWORDS = {
    # generic
    "the", "and", "is", "to", "a", "of", "in", "for", "it", "this", "on", "my", "with", "that", "was", "are", "but",
    "so", "at", "be", "have", "not", "you",
    "very", "bad", "good", "best", "nice", "love", "awesome", "amazing", "ok", "please", "help", "cant", "can't",
    "wont", "won't", "doesnt", "doesn't",
    # domain noise
    "whatsapp", "wa", "app", "application", "facing", "open"
}

_WORD_RE = re.compile(r"[a-zA-Z]{3,}")


def _tokens(text: str) -> List[str]:
    return [w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS]


def _bigrams(words: List[str]) -> Iterable[str]:
    a, b = tee(words)
    next(b, None)
    return (f"{w1} {w2}" for w1, w2 in zip(a, b))


def extract_keywords(texts: List[str], top_n: int = 5) -> List[str]:
    """
    Lightweight keyword extractor: cleans text, removes stopwords,
    counts unigrams + bigrams, prefers informative bigrams over generic unigrams.
    """
    unigram_counts: Counter = Counter()
    bigram_counts: Counter = Counter()

    for t in texts:
        words = _tokens(t)
        unigram_counts.update(words)
        bigram_counts.update(_bigrams(words))

    # Prefer bigrams that occur at least twice and are not redundant
    candidates = []
    for phrase, c in bigram_counts.most_common():
        if c >= 2:
            candidates.append((phrase, c))

    # Fill with strong unigrams if we still need more
    for w, c in unigram_counts.most_common():
        # Skip if unigram is already part of a selected bigram
        if any(w in p.split() for p, _ in candidates):
            continue
        candidates.append((w, c))
        if len(candidates) >= top_n * 2:
            break

    # Sort by count desc, then take top_n unique phrases/words
    candidates.sort(key=lambda x: x[1], reverse=True)
    dedup, seen = [], set()
    for k, _ in candidates:
        if k not in seen:
            dedup.append(k)
            seen.add(k)
        if len(dedup) >= top_n:
            break
    return dedup
