import re
import math
from collections import Counter, defaultdict
from typing import List, Iterable, Set, Tuple
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import spacy

# Load spaCy for lemmatization (disable unused components for speed)
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

_DOMAIN_STOPWORDS = {
    "whatsapp", "wa", "app", "application", "facing", "open", "new", "set",
    "google", "play", "plants"
}
_STOPWORDS: Set[str] = set(ENGLISH_STOP_WORDS) | _DOMAIN_STOPWORDS
_WORD_RE = re.compile(r"[a-z]{3,}")


def _brand_stop_set(app_id: str | None) -> Set[str]:
    if not app_id:
        return set()
    parts = app_id.replace("-", ".").split(".")
    return {p.lower() for p in parts if p}


def _lemmatize_tokens(tokens: List[str]) -> List[str]:
    doc = nlp(" ".join(tokens))
    return [t.lemma_ for t in doc if t.lemma_ != "-PRON-"]


def _tokens(text: str, stop: Set[str] | None = None, lemmatize: bool = True) -> List[str]:
    stop = stop or _STOPWORDS
    words = [w for w in _WORD_RE.findall(text.lower()) if w not in stop]
    return _lemmatize_tokens(words) if lemmatize else words


def _ngrams(words: List[str], n: int) -> Iterable[str]:
    if n == 1:
        yield from words
        return
    for i in range(len(words) - n + 1):
        chunk = words[i:i + n]
        if any(chunk[j] == chunk[j + 1] for j in range(len(chunk) - 1)):
            continue
        yield " ".join(chunk)


def _doc_freq(docs: List[List[str]]) -> Counter:
    df = Counter()
    for toks in docs:
        df.update(set(toks))
    return df


def _dedup_adjacent(tokens: List[str]) -> List[str]:
    out, prev = [], None
    for t in tokens:
        if t != prev:
            out.append(t)
        prev = t
    return out


def _tokenize_docs(texts: List[str], extra_stopwords: Set[str] | None = None) -> List[List[str]]:
    stopwords_combined = _STOPWORDS | set(map(str.lower, extra_stopwords or []))
    out = []
    for t in texts:
        toks = [w for w in _WORD_RE.findall(t.lower()) if w not in stopwords_combined]
        toks = _dedup_adjacent(toks)
        out.append(toks)
    return out


def _ngram_docs_with_postings(docs: List[List[str]], n: int):
    ngram_docs, postings = [], defaultdict(set)
    for i, toks in enumerate(docs):
        terms = list(_ngrams(toks, n))
        uniq = set(terms)
        ngram_docs.append(list(uniq))
        for term in uniq:
            postings[term].add(i)
    return ngram_docs, postings


def extract_keywords_contrastive(
        negatives: List[str],
        all_texts: List[str],
        top_n: int = 7,
        min_doc_neg: int = 2,
        extra_stopwords: Set[str] | None = None,
        app_id: str | None = None,
        ngram_range: Tuple[int, int] = (1, 3),
        max_doc_ratio: float = 0.6,
        return_evidence: bool = False,
) -> List[str] | List[Tuple[str, List[int]]]:
    brand_stops = _brand_stop_set(app_id)
    extra = (extra_stopwords or set()) | brand_stops

    neg_uni_docs = _tokenize_docs(negatives, extra)
    all_uni_docs = _tokenize_docs(all_texts, extra)
    Nn, Na = max(len(neg_uni_docs), 1), max(len(all_uni_docs), 1)

    n_min, n_max = ngram_range
    dfs_neg, dfs_all, postings_neg = {}, {}, {}

    for n in range(n_min, n_max + 1):
        neg_ng_docs, neg_post = _ngram_docs_with_postings(neg_uni_docs, n)
        all_ng_docs, _ = _ngram_docs_with_postings(all_uni_docs, n)

        df_neg = Counter()
        [df_neg.update(ngs) for ngs in neg_ng_docs]
        df_all = Counter()
        [df_all.update(ngs) for ngs in all_ng_docs]

        dfs_neg[n], dfs_all[n], postings_neg[n] = df_neg, df_all, neg_post

    def score_df(df_neg, df_all):
        V = max(len(df_all), 1)
        scores = {}
        for term, c_neg in df_neg.items():
            if c_neg < min_doc_neg or (c_neg / Nn) > max_doc_ratio:
                continue
            c_all = df_all.get(term, 0)
            p_neg = (c_neg + 1) / (Nn + V)
            p_all = (c_all + 1) / (Na + V)
            length_bonus = 1.0 + 0.1 * (len(term.split()) - 1)
            scores[term] = math.log(p_neg / p_all) * length_bonus
        return scores

    picked: list[tuple[str, list[int]]] = []
    for n in range(n_max, n_min - 1, -1):
        scores = score_df(dfs_neg[n], dfs_all[n])
        if not scores:
            continue
        for term, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            idxs = sorted(postings_neg[n].get(term, set()))
            if not idxs:
                continue
            words = term.split()
            if any(set(words).issubset(set(p.split())) for p, _ in picked):
                continue
            picked.append((term, idxs))
            if len(picked) >= top_n:
                break
        if len(picked) >= top_n:
            break

    return picked if return_evidence else [t for t, _ in picked]


def _token_set(phrase: str) -> set[str]:
    return set(phrase.lower().split())


def _jaccard(a: set[str], b: set[str]) -> float:
    inter = len(a & b)
    union = len(a | b) or 1
    return inter / union


def merge_near_duplicate_terms(
        picked: List[Tuple[str, List[int]]],
        threshold: float = 0.5
) -> List[Tuple[str, List[int]]]:
    clusters: List[Tuple[str, set[int], set[str]]] = []
    for term, idxs in picked:
        ts = _token_set(term)
        placed = False
        for i, (rep, idx_set, rep_ts) in enumerate(clusters):
            if _jaccard(ts, rep_ts) >= threshold:
                idx_set.update(idxs)
                clusters[i] = (rep, idx_set, rep_ts | ts)
                placed = True
                break
        if not placed:
            clusters.append((term, set(idxs), ts))

    merged: List[Tuple[str, List[int]]] = []
    for rep, idx_set, _ in clusters:
        merged.append((rep, sorted(idx_set)))
    merged.sort(key=lambda x: len(x[1]), reverse=True)
    return merged
