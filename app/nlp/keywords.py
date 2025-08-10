import re
import math
from collections import Counter
from typing import List, Iterable, Set
from itertools import tee

_STOPWORDS = {
    # generic
    "the", "and", "is", "to", "a", "of", "in", "for", "it", "this", "on", "my", "with",
    "that", "was", "are", "but", "so", "at", "be", "have", "not", "you",
    "very", "bad", "good", "best", "nice", "love", "awesome", "amazing",
    "ok", "please", "help", "cant", "can't", "wont", "won't", "doesnt", "doesn't", "worst",
    # domain noise / too generic in feedback
    "whatsapp", "wa", "app", "application", "facing", "open", "new", "set"
}

# Match only alphabetic words of length >= 3
_WORD_RE = re.compile(r"[a-z]{3,}")


def _tokens(text: str) -> List[str]:
    """Tokenize and lowercase text, remove stopwords."""
    return [w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS]


def _bigrams(words: List[str]) -> Iterable[str]:
    """Generate bigrams from a list of words."""
    a, b = tee(words)
    next(b, None)
    return (f"{w1} {w2}" for w1, w2 in zip(a, b))


def extract_keywords(texts: List[str], top_n: int = 5) -> List[str]:
    """
    Lightweight keyword extractor:
    - Cleans text
    - Removes stopwords
    - Counts unigrams + bigrams
    - Prefers bigrams that occur at least twice
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
        if any(w in p.split() for p, _ in candidates):
            continue
        candidates.append((w, c))
        if len(candidates) >= top_n * 2:
            break

    # Sort and deduplicate
    candidates.sort(key=lambda x: x[1], reverse=True)
    dedup, seen = [], set()
    for k, _ in candidates:
        if k not in seen:
            dedup.append(k)
            seen.add(k)
        if len(dedup) >= top_n:
            break

    return dedup


def _doc_freq(docs: List[List[str]]) -> Counter:
    """Document frequency: in how many documents each term appears."""
    df = Counter()
    for toks in docs:
        df.update(set(toks))
    return df


def _tokenize_docs(texts: List[str], extra_stopwords: Set[str] | None = None) -> List[List[str]]:
    if not texts:
        return []
    extra = set(map(str.lower, extra_stopwords or []))
    out = []
    for t in texts:
        toks = [w for w in _WORD_RE.findall(t.lower())
                if w not in _STOPWORDS and w not in extra]
        out.append(toks)
    return out


def _bigrams_docs(docs: List[List[str]]) -> List[List[str]]:
    out: List[List[str]] = []
    for toks in docs:
        a, b = tee(toks)
        next(b, None)
        out.append([f"{w1} {w2}" for w1, w2 in zip(a, b)])
    return out


def extract_keywords_contrastive(
        negatives: List[str],
        all_texts: List[str],
        top_n: int = 5,
        min_doc_neg: int = 2,
        extra_stopwords: Set[str] | None = None,
) -> List[str]:
    """
    Contrastive keywords using log-odds with add-1 smoothing on document frequencies.
    Prefers bigrams; falls back to unigrams. Good for small negative samples.
    """
    neg_uni_docs = _tokenize_docs(negatives, extra_stopwords)
    all_uni_docs = _tokenize_docs(all_texts, extra_stopwords)

    neg_bi_docs = _bigrams_docs(neg_uni_docs)
    all_bi_docs = _bigrams_docs(all_uni_docs)

    # Document frequencies
    df_neg_uni = _doc_freq(neg_uni_docs)
    df_all_uni = _doc_freq(all_uni_docs)
    df_neg_bi = _doc_freq(neg_bi_docs)
    df_all_bi = _doc_freq(all_bi_docs)

    Nn_uni = max(len(neg_uni_docs), 1)
    Na_uni = max(len(all_uni_docs), 1)
    Nn_bi = max(len(neg_bi_docs), 1)
    Na_bi = max(len(all_bi_docs), 1)

    def score_df(df_neg, df_all, Nn, Na, min_doc):
        scores = {}
        V = max(len(df_all), 1)
        for term, c_neg in df_neg.items():
            if c_neg < min_doc:
                continue
            c_all = df_all.get(term, 0)
            p_neg = (c_neg + 1) / (Nn + V)
            p_all = (c_all + 1) / (Na + V)
            scores[term] = math.log(p_neg / p_all)
        return scores

    # Score bigrams first (require at least 2 negative docs)
    scores_bi = score_df(df_neg_bi, df_all_bi, Nn_bi, Na_bi, min_doc=min_doc_neg)
    # Then unigrams (also require 2 docs)
    scores_uni = score_df(df_neg_uni, df_all_uni, Nn_uni, Na_uni, min_doc=min_doc_neg)

    # Merge: prefer bigrams; fill with strong unigrams
    ordered = sorted(scores_bi.items(), key=lambda x: x[1], reverse=True)
    selected: list[str] = [k for k, _ in ordered[:top_n]]

    if len(selected) < top_n:
        for k, _ in sorted(scores_uni.items(), key=lambda x: x[1], reverse=True):
            if any(k in bi.split() for bi in selected):
                continue
            selected.append(k)
            if len(selected) >= top_n:
                break
    return selected[:top_n]
