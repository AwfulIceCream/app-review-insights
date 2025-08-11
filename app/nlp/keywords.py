import re
import math
from collections import Counter
from typing import List, Iterable, Set, Tuple
from itertools import tee
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# -------------------------------
# Stopwords (base + domain + brand helper)
# -------------------------------
_DOMAIN_STOPWORDS = {
    "whatsapp", "wa", "app", "application", "facing", "open", "new", "set",
    "google", "play", "plants"
}

_STOPWORDS: Set[str] = set(ENGLISH_STOP_WORDS) | _DOMAIN_STOPWORDS


def _brand_stop_set(app_id: str | None) -> Set[str]:
    if not app_id:
        return set()
    parts = app_id.replace("-", ".").split(".")
    return {p.lower() for p in parts if p}


# -------------------------------
# Tokenization + n-grams
# -------------------------------
_WORD_RE = re.compile(r"[a-z]{3,}")


def _tokens(text: str, stop: Set[str] | None = None) -> List[str]:
    stop = stop or _STOPWORDS
    return [w for w in _WORD_RE.findall(text.lower()) if w not in stop]


def _bigrams(words: List[str]) -> Iterable[str]:
    a, b = tee(words)
    next(b, None)
    return (f"{w1} {w2}" for w1, w2 in zip(a, b))


def _ngrams(words: List[str], n: int) -> Iterable[str]:
    if n == 1:
        for w in words:
            yield w
        return
    a = [iter(words)]
    for _ in range(n - 1):
        a.append(iter(words))
    for i in range(1, n):
        next(a[i], None)
    for tup in zip(*a):
        yield " ".join(tup)


# -------------------------------
# Lightweight keywords (unchanged)
# -------------------------------
def extract_keywords(texts: List[str], top_n: int = 5) -> List[str]:
    unigram_counts: Counter = Counter()
    bigram_counts: Counter = Counter()

    for t in texts:
        words = _tokens(t)
        unigram_counts.update(words)
        bigram_counts.update(_bigrams(words))

    candidates: List[Tuple[str, int]] = []
    for phrase, c in bigram_counts.most_common():
        if c >= 2:
            candidates.append((phrase, c))

    for w, c in unigram_counts.most_common():
        if any(w in p.split() for p, _ in candidates):
            continue
        candidates.append((w, c))
        if len(candidates) >= top_n * 2:
            break

    candidates.sort(key=lambda x: x[1], reverse=True)
    dedup, seen = [], set()
    for k, _ in candidates:
        if k not in seen:
            dedup.append(k)
            seen.add(k)
        if len(dedup) >= top_n:
            break
    return dedup


# -------------------------------
# Contrastive helpers
# -------------------------------
def _doc_freq(docs: List[List[str]]) -> Counter:
    df = Counter()
    for toks in docs:
        df.update(set(toks))
    return df


def _tokenize_docs(texts: List[str], extra_stopwords: Set[str] | None = None) -> List[List[str]]:
    if not texts:
        return []
    stopwords_combined = _STOPWORDS | set(map(str.lower, extra_stopwords or []))
    out = []
    for t in texts:
        toks = [w for w in _WORD_RE.findall(t.lower()) if w not in stopwords_combined]
        out.append(toks)
    return out


def _ngram_docs(docs: List[List[str]], n: int) -> List[List[str]]:
    out: List[List[str]] = []
    for toks in docs:
        out.append(list(_ngrams(toks, n)))
    return out


# -------------------------------
# Contrastive keywords (enhanced)
# -------------------------------
def extract_keywords_contrastive(
        negatives: List[str],
        all_texts: List[str],
        top_n: int = 7,
        min_doc_neg: int = 2,
        extra_stopwords: Set[str] | None = None,
        app_id: str | None = None,  # NEW: optional brand tokens
        ngram_range: Tuple[int, int] = (1, 3),  # NEW: include up to trigrams
        max_doc_ratio: float = 0.6,  # NEW: drop terms appearing in >60% of negative docs (too generic)
) -> List[str]:
    """
    Contrastive keywords using log-odds with add-1 smoothing on document frequencies.
    Prefers longer n-grams, falls back to shorter ones. Good for small negative samples.
    Also filters over-generic terms by document prevalence (max_doc_ratio).
    """
    # Merge brand stops if provided
    brand_stops = _brand_stop_set(app_id)
    extra = (extra_stopwords or set()) | brand_stops

    neg_uni_docs = _tokenize_docs(negatives, extra)
    all_uni_docs = _tokenize_docs(all_texts, extra)

    Nn = max(len(neg_uni_docs), 1)
    Na = max(len(all_uni_docs), 1)

    # Build DF dicts for each n in range
    n_min, n_max = ngram_range
    ngram_dfs_neg: dict[int, Counter] = {}
    ngram_dfs_all: dict[int, Counter] = {}

    for n in range(n_min, n_max + 1):
        neg_ng_docs = _ngram_docs(neg_uni_docs, n)
        all_ng_docs = _ngram_docs(all_uni_docs, n)
        ngram_dfs_neg[n] = _doc_freq(neg_ng_docs)
        ngram_dfs_all[n] = _doc_freq(all_ng_docs)

    def score_df(df_neg: Counter, df_all: Counter, Nn: int, Na: int, min_doc: int) -> dict[str, float]:
        scores: dict[str, float] = {}
        V = max(len(df_all), 1)
        for term, c_neg in df_neg.items():
            if c_neg < min_doc:
                continue
            # filter terms that are too ubiquitous in negatives (overly generic)
            if (c_neg / Nn) > max_doc_ratio:
                continue
            c_all = df_all.get(term, 0)
            p_neg = (c_neg + 1) / (Nn + V)
            p_all = (c_all + 1) / (Na + V)
            # prefer longer phrases slightly: +10% per extra token
            length_bonus = 1.0 + 0.1 * (len(term.split()) - 1)
            scores[term] = math.log(p_neg / p_all) * length_bonus
        return scores

    # Score from longest n down to 1
    selected: list[str] = []
    already_words: Set[str] = set()

    for n in range(n_max, n_min - 1, -1):
        df_neg = ngram_dfs_neg.get(n, Counter())
        df_all = ngram_dfs_all.get(n, Counter())
        scores = score_df(df_neg, df_all, Nn, Na, min_doc=min_doc_neg)
        if not scores:
            continue

        for term, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            # skip if all words are already covered by a longer selected phrase
            words = term.split()
            if words and any(set(words).issubset(set(s.split())) for s in selected):
                continue
            selected.append(term)
            already_words.update(words)
            if len(selected) >= top_n:
                break
        if len(selected) >= top_n:
            break

    # Final guard: ensure we return at most top_n
    return selected[:top_n]
