from typing import List, Dict, Any
from dataclasses import dataclass
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

from app.nlp.keywords import extract_keywords_contrastive, _tokens
from app.nlp.sentiment import analyze_sentiment


@dataclass
class Topic:
    label: str
    examples: List[str]
    suggestion: str
    size: int


def _safe_vectorize(texts: List[str], min_df: int = 2):
    """
    Vectorize texts with a resilient TF-IDF config.
    Tries min_df=2 first, then backs off to min_df=1 if needed.
    """
    if not texts:
        return None, None
    # Basic sanity: ensure some tokenizable content
    has_content = any(len(_tokens(t)) > 0 for t in texts)
    if not has_content:
        return None, None

    vect = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=min_df,
        max_df=0.9
    )
    try:
        X = vect.fit_transform(texts)
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("No features")
        return vect, X
    except Exception:
        # Back off to min_df=1
        vect = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.9
        )
        X = vect.fit_transform(texts)
        if X.shape[0] == 0 or X.shape[1] == 0:
            return None, None
        return vect, X


def _pick_k(X, min_k: int, max_k: int) -> int:
    """
    Pick a reasonable k using silhouette score (cosine).
    Falls back to min_k if scoring fails or is degenerate.
    """
    best_k, best_score = None, -1.0
    for k in range(min_k, max_k + 1):
        try:
            km = KMeans(n_clusters=k, n_init="auto", random_state=42)
            labels = km.fit_predict(X)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(X, labels, metric="cosine")
            if score > best_score:
                best_k, best_score = k, score
        except Exception:
            continue
    return best_k or min_k


def _top_terms(vectorizer: TfidfVectorizer, centroid: np.ndarray, top_n: int = 5) -> List[str]:
    """Return top_n feature names for a centroid vector."""
    feats = np.array(vectorizer.get_feature_names_out())
    idxs = np.argsort(centroid)[::-1][:top_n]
    return [t for t in feats[idxs] if t]


def _examples_near_centroid(X_cluster, centroid: np.ndarray, all_index: np.ndarray, texts: List[str], n: int = 2) -> \
        List[str]:
    """Pick up to n examples closest to centroid by cosine similarity."""
    sims = cosine_similarity(X_cluster, centroid.reshape(1, -1)).ravel()
    top_local = np.argsort(-sims)[:n]
    idxs = all_index[top_local]
    return [texts[i] for i in idxs]


def cluster_topics(
        texts: List[str],
        max_topics: int = 6,
        min_cluster_size: int = 2
) -> List[Topic]:
    """
    Cluster texts into topics with TF-IDF + KMeans.
    Filters out clusters smaller than min_cluster_size.
    """
    if not texts:
        return []

    vectorizer, X = _safe_vectorize(texts)
    if vectorizer is None or X is None:
        return []

    n_docs = X.shape[0]
    # Determine a feasible k given min_cluster_size (avoid too many tiny clusters)
    feasible_max_k = max(1, min(max_topics, n_docs // min_cluster_size))
    if feasible_max_k <= 1:
        # Not enough docs for >1 cluster → single topic from global terms
        centroid = np.asarray(X.mean(axis=0)).ravel()
        terms = _top_terms(vectorizer, centroid, top_n=5)
        label = ", ".join(terms[:3]) if terms else "General issue"
        examples = _examples_near_centroid(X, centroid, np.arange(n_docs), texts, n=min(2, n_docs))
        return [Topic(label=label, examples=examples, suggestion=_suggest_from_terms(terms), size=n_docs)]

    k = _pick_k(X, min_k=2, max_k=feasible_max_k)
    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = km.fit_predict(X)

    topics: List[Topic] = []
    for cid in range(k):
        idxs = np.where(labels == cid)[0]
        if idxs.size < min_cluster_size:
            continue
        centroid = km.cluster_centers_[cid]
        terms = _top_terms(vectorizer, centroid, top_n=5)
        label = ", ".join(terms[:3]) if terms else "General issue"
        Xc = X[idxs]
        examples = _examples_near_centroid(Xc, centroid, idxs, texts, n=min(2, idxs.size))
        topics.append(Topic(label=label, examples=examples, suggestion=_suggest_from_terms(terms), size=idxs.size))

    return topics


def _suggest_from_terms(terms: List[str]) -> str:
    """
    Fully data-driven suggestion sentence built from the cluster's top terms.
    No hardcoded categories or keyword rules.
    """
    if not terms:
        return "Prioritize fixes and UX improvements for reported issues. Validate with focused tests and monitor error/feedback metrics."
    head = terms[0]
    tail = ", ".join(terms[1:3]) if len(terms) > 1 else ""
    extra = f" (e.g., {tail})" if tail else ""
    return f"Prioritize fixes and UX improvements around '{head}'{extra}. Validate with targeted tests and monitor error/feedback metrics."


def make_actionable_insights(
        negative_texts: List[str],
        all_texts: List[str] | None = None,
        max_topics: int = 6,
        min_cluster_size: int = 2,
        validate_sentiment: bool = True,  # new flag
) -> List[Dict[str, Any]]:
    """
    Main entry:
      - Cluster negative reviews into topics.
      - Optionally re-validate sentiment inside clusters to avoid positives sneaking in.
      - Filter out tiny clusters.
      - If clustering yields nothing useful, fallback to a single insight using contrastive keywords.
    """
    negative_texts = [t for t in negative_texts if t and t.strip()]
    if not negative_texts:
        return []

    topics = cluster_topics(negative_texts, max_topics=max_topics, min_cluster_size=min_cluster_size)

    if topics and validate_sentiment:
        filtered_topics = []
        for t in topics:
            # Keep only truly negative examples
            neg_examples = [ex for ex in t.examples if analyze_sentiment(ex) == "negative"]
            if len(neg_examples) >= min_cluster_size:
                filtered_topics.append(
                    type(t)(
                        label=t.label,
                        examples=neg_examples,
                        suggestion=t.suggestion,
                        size=len(neg_examples),
                    )
                )
        topics = filtered_topics

    if topics:
        return [
            {
                "issue": t.label,
                "evidence_examples": t.examples,
                "suggestion": t.suggestion,
                "size": t.size,
            }
            for t in topics
        ]

    # Fallback: single synthetic insight from keywords
    if all_texts is None:
        all_texts = negative_texts
    kws = extract_keywords_contrastive(negative_texts, all_texts, top_n=5)
    label = ", ".join(kws[:3]) if kws else "Reported issues"
    suggestion = _suggest_from_terms(kws)
    examples = negative_texts[:2]
    return [
        {
            "issue": label,
            "evidence_examples": examples,
            "suggestion": suggestion,
            "size": len(negative_texts),
        }
    ]
