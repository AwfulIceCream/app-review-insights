import re
from collections import Counter
from typing import List

_STOPWORDS = set([
    "the", "and", "is", "to", "a", "of", "in", "for", "it", "this", "on", "my",
    "with", "that", "was", "are", "but", "so", "at", "be", "have", "not", "you"
])

_WORD_RE = re.compile(r"[a-zA-Z]{3,}")


def extract_keywords(texts: List[str], top_n: int = 5) -> List[str]:
    """
    Simple keyword extractor: lowercase words, remove stopwords,
    count frequencies, return top_n keywords.
    """
    words = []
    for t in texts:
        for w in _WORD_RE.findall(t.lower()):
            if w not in _STOPWORDS:
                words.append(w)
    most_common = Counter(words).most_common(top_n)
    return [w for w, _ in most_common]
