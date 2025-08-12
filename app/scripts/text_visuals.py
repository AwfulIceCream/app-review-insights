import os
import re
import sys
from collections import Counter
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt

from app.collectors.google_play import fetch_reviews, CollectError
from app.utils.metrics import extract_fields
from app.nlp.keywords import extract_keywords_contrastive
from app.api.routes.insights import get_sentiment

# ----------------------------
# spaCy stopwords (with safe fallback)
# ----------------------------
_SPACY_AVAILABLE = False
_STOPWORDS = set()

try:
    import spacy

    try:
        # Full small English model
        _nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        _SPACY_AVAILABLE = True
        _STOPWORDS = set(_nlp.Defaults.stop_words)
    except OSError:
        # Fallback to a blank English pipeline (no vocab/model weights, but has stopwords list)
        _nlp = spacy.blank("en")
        _SPACY_AVAILABLE = True
        _STOPWORDS = set(_nlp.Defaults.stop_words)
except Exception:
    _SPACY_AVAILABLE = False
    _STOPWORDS = {
        # minimal fallback list if spaCy isn't available
        "the", "and", "for", "you", "your", "this", "that", "was", "with", "have", "had", "but", "are",
        "not", "get", "got", "can", "cant", "cannot", "its", "they", "them", "too", "very", "has", "been",
        "from", "all", "our", "out", "just", "after", "into", "any", "when", "where", "how", "why", "who",
        "what", "which", "while", "also", "will", "would", "should", "could", "did", "does", "doing",
        "don", "doesn", "didn", "isn", "aren", "wasn",
    }

# ----------------------------
# Plots output directory
# ----------------------------
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# ----------------------------
# Tokenization + n-grams
# ----------------------------
_WORD_RE = re.compile(r"[a-z]{3,}")


def _tokens_regex(text: str) -> List[str]:
    """Regex tokenizer (no stopword removal)."""
    return _WORD_RE.findall(text.lower())


def _tokens_spacy(text: str, remove_stopwords: bool, lemmatize: bool) -> List[str]:
    """
    spaCy-based tokenizer with optional stopword removal and lemmatization.
    Requires _SPACY_AVAILABLE. If not, falls back to regex.
    """
    if not _SPACY_AVAILABLE:
        toks = _WORD_RE.findall(text.lower())
        return [t for t in toks if (t not in _STOPWORDS) or (not remove_stopwords)]
    doc = _nlp(text.lower())
    out = []
    for t in doc:
        if not t.is_alpha or len(t.text) < 3:
            continue
        if remove_stopwords and t.is_stop:
            continue
        out.append(t.lemma_ if lemmatize else t.text)
    return out


def _tokenize(text: str, remove_stopwords: bool = False, lemmatize: bool = True) -> List[str]:
    """Unified tokenizer: use spaCy if available, else regex + manual stopwords."""
    if _SPACY_AVAILABLE:
        return _tokens_spacy(text, remove_stopwords=remove_stopwords, lemmatize=lemmatize)
    toks = _tokens_regex(text)
    return [t for t in toks if (t not in _STOPWORDS) or (not remove_stopwords)]


def _ngrams(words: List[str], n: int) -> Iterable[str]:
    if n == 1:
        yield from words
        return
    # sliding window
    iters = [iter(words)]
    for _ in range(n - 1):
        iters.append(iter(words))
    for i in range(1, n):
        next(iters[i], None)
    for tup in zip(*iters):
        yield " ".join(tup)


# ----------------------------
# Term counting + plotting
# ----------------------------
def top_terms(
        texts: List[str],
        ngram_range: Tuple[int, int] = (1, 2),
        top_k: int = 20,
        remove_stopwords: bool = False,
        lemmatize: bool = True
) -> List[Tuple[str, int]]:
    """Counts n-grams and returns top_k (term, count)."""
    c = Counter()
    for t in texts:
        w = _tokenize(t, remove_stopwords=remove_stopwords, lemmatize=lemmatize)
        for n in range(ngram_range[0], ngram_range[1] + 1):
            c.update(_ngrams(w, n))
    # guard for very short artifacts
    filtered = Counter({k: v for k, v in c.items() if len(k) >= 3})
    return filtered.most_common(top_k)


def _plot_barh(pairs: List[Tuple[str, int]], title: str, filename: str):
    if not pairs:
        print(f"No data for {title}")
        return
    terms = [p[0] for p in pairs][::-1]
    freqs = [p[1] for p in pairs][::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(terms, freqs)
    ax.set_xlabel("Count")
    ax.set_title(title)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, dpi=180)
    print(f"Saved {path}")


def plot_top_ngrams_in_negatives(
        rows,
        ngram_range=(1, 2),
        top_k=20,
        filename="neg_ngrams.png",
        remove_stopwords=False,
        lemmatize=True
):
    neg_texts = [
        (r.get("text") or "").strip()
        for r in rows
        if r.get("text") and isinstance(r.get("rating"), int) and get_sentiment(r["text"]) == "negative"
    ]
    pairs = top_terms(
        neg_texts,
        ngram_range=ngram_range,
        top_k=top_k,
        remove_stopwords=remove_stopwords,
        lemmatize=lemmatize,
    )
    tag = []
    if remove_stopwords: tag.append("no stopwords")
    if lemmatize: tag.append("lemmas")
    suffix = f" ({', '.join(tag)})" if tag else ""
    _plot_barh(pairs, f"Top n-grams in negative reviews (n={len(neg_texts)}){suffix}", filename)


def plot_top_terms_per_sentiment(
        rows,
        ngram=(1, 1),
        top_k=12,
        filename="terms_per_sentiment.png",
        remove_stopwords=False,
        lemmatize=True
):
    buckets = {"negative": [], "neutral": [], "positive": []}
    for r in rows:
        t = (r.get("text") or "").strip()
        if not t:
            continue
        label = get_sentiment(t)
        if label not in buckets:
            label = "neutral"
        buckets[label].append(t)

    left, right = ngram
    tops = {}
    for k, texts in buckets.items():
        pairs = top_terms(
            texts,
            ngram_range=(left, right),
            top_k=top_k,
            remove_stopwords=remove_stopwords,
            lemmatize=lemmatize,
        )
        tops[k] = pairs

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    classes = ["negative", "neutral", "positive"]
    for ax, c in zip(axes, classes):
        pairs = tops.get(c, [])
        terms = [p[0] for p in pairs][::-1]
        freqs = [p[1] for p in pairs][::-1]
        ax.barh(terms, freqs)
        ax.set_title(c.capitalize())
        ax.set_xlabel("Count")
    tag = []
    if remove_stopwords: tag.append("no stopwords")
    if lemmatize: tag.append("lemmas")
    suffix = f" ({', '.join(tag)})" if tag else ""
    fig.suptitle(f"Top terms per sentiment (n-gram {ngram[0]}–{ngram[1]}){suffix}")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, dpi=180)
    print(f"Saved {path}")


def plot_keyword_coverage_bar(
        picked_terms_with_idxs: List[Tuple[str, List[int]]],
        filename="keyword_coverage.png"
):
    if not picked_terms_with_idxs:
        print("No keyword evidence to plot.")
        return
    terms = [t for t, _ in picked_terms_with_idxs][::-1]
    counts = [len(idxs) for _, idxs in picked_terms_with_idxs][::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(terms, counts)
    ax.set_xlabel("# of negative reviews covered")
    ax.set_title("Keyword evidence coverage")
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, dpi=180)
    print(f"Saved {path}")


def main():
    app_id = "com.myplantin.app"
    count = 200
    lang = "en"
    country = "us"

    try:
        raw = fetch_reviews(app_id=app_id, count=count, lang=lang, country=country)
    except CollectError as ce:
        print(f"Error: {ce}", file=sys.stderr)
        sys.exit(1)

    rows = extract_fields(raw)
    rows = [r for r in rows if r.get("text") and r["text"].strip()]
    if not rows:
        print("No usable reviews.", file=sys.stderr)
        sys.exit(2)

    texts_all = [r["text"].strip() for r in rows]
    texts_neg = [t for t in texts_all if get_sentiment(t) == "negative"]

    # 1) Top n-grams in negatives
    plot_top_ngrams_in_negatives(
        rows,
        ngram_range=(1, 2),
        top_k=20,
        filename="neg_ngrams.png",
        remove_stopwords=False,
        lemmatize=False,
    )
    plot_top_ngrams_in_negatives(
        rows,
        ngram_range=(1, 2),
        top_k=20,
        filename="neg_ngrams_nostop_lemmas.png",
        remove_stopwords=True,
        lemmatize=True,
    )

    # 2) Per-sentiment top terms
    plot_top_terms_per_sentiment(
        rows,
        ngram=(1, 1),
        top_k=12,
        filename="terms_per_sentiment.png",
        remove_stopwords=False,
        lemmatize=False,
    )
    plot_top_terms_per_sentiment(
        rows,
        ngram=(1, 1),
        top_k=12,
        filename="terms_per_sentiment_nostop_lemmas.png",
        remove_stopwords=True,
        lemmatize=True,
    )

    # 3) Keyword coverage using your contrastive extractor (with evidence)
    picked = extract_keywords_contrastive(
        negatives=texts_neg,
        all_texts=texts_all,
        top_n=8,
        min_doc_neg=2,
        app_id=app_id,
        ngram_range=(1, 3),
        max_doc_ratio=0.6,
        return_evidence=True,
    )
    plot_keyword_coverage_bar(
        picked_terms_with_idxs=picked,
        filename="keyword_coverage.png",
    )

    print("Saved plots to:", os.path.abspath(PLOTS_DIR))
    if not _SPACY_AVAILABLE:
        print(
            "Note: spaCy model not found. For best results, install it:\n"
            "  pip install spacy && python -m spacy download en_core_web_sm",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
