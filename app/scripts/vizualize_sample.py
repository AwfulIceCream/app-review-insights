import os
import sys
import logging
from collections import Counter
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

from app.collectors.google_play import fetch_reviews, CollectError
from app.utils.metrics import extract_fields
from app.api.routes.insights import get_sentiment

# ----------------------------
# Configuration
# ----------------------------
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)

SENTIMENT_CLASSES = ["negative", "neutral", "positive"]

# ----------------------------
# Plot functions
# ----------------------------
def plot_sentiment_distribution(sentiment_distribution: Dict[str, int], save_path: str) -> None:
    """Pie chart of sentiment distribution."""
    labels = list(sentiment_distribution.keys())
    sizes = list(sentiment_distribution.values())
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140)
    ax.set_title("Sentiment Distribution")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    logging.info("Saved %s", save_path)


def plot_rating_distribution(rating_distribution: Dict[int, int], save_path: str) -> None:
    """Bar chart of rating (1–5 stars) distribution."""
    labels = sorted(rating_distribution.keys())
    counts = [rating_distribution[k] for k in labels]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, counts)
    ax.set_xlabel("Rating")
    ax.set_ylabel("Number of Reviews")
    ax.set_title("Rating Distribution (1–5 stars)")
    ax.set_xticks(labels)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    logging.info("Saved %s", save_path)


def star_to_sentiment(rating: int) -> str:
    """Map 1–5 star rating to sentiment label."""
    if rating <= 2:
        return "negative"
    if rating == 3:
        return "neutral"
    return "positive"


def sentiment_counts_both(rows, sentiment_fn) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Compare sentiment counts from star ratings vs model."""
    star_cnt = Counter({c: 0 for c in SENTIMENT_CLASSES})
    model_cnt = Counter({c: 0 for c in SENTIMENT_CLASSES})

    for r in rows:
        text = (r.get("text") or "").strip()
        rating = r.get("rating")
        if not text or not isinstance(rating, int):
            continue

        star_label = star_to_sentiment(rating)
        model_label = sentiment_fn(text)

        if star_label not in star_cnt:
            star_label = "neutral"
        if model_label not in model_cnt:
            model_label = "neutral"

        star_cnt[star_label] += 1
        model_cnt[model_label] += 1

    return dict(star_cnt), dict(model_cnt)


def plot_sentiment_methods_comparison(
    star_cnt: Dict[str, int],
    model_cnt: Dict[str, int],
    save_path: str,
    normalize: bool = False
) -> None:
    """Grouped bar chart comparing stars vs model sentiment."""
    star_vals = [star_cnt.get(c, 0) for c in SENTIMENT_CLASSES]
    model_vals = [model_cnt.get(c, 0) for c in SENTIMENT_CLASSES]

    if normalize:
        s_total = sum(star_vals) or 1
        m_total = sum(model_vals) or 1
        star_vals = [v / s_total * 100 for v in star_vals]
        model_vals = [v / m_total * 100 for v in model_vals]

    x = np.arange(len(SENTIMENT_CLASSES))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width / 2, star_vals, width, label="Stars → expected")
    ax.bar(x + width / 2, model_vals, width, label="Model → predicted")
    ax.set_xticks(x, SENTIMENT_CLASSES)
    ax.set_ylabel("Percentage (%)" if normalize else "Reviews")
    ax.set_title("Sentiment: Stars vs Model")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    logging.info("Saved %s", save_path)

# ----------------------------
# CLI Entrypoint
# ----------------------------
if __name__ == "__main__":
    app_id = "com.myplantin.app"
    count = 100
    lang = "en"
    country = "us"

    try:
        raw = fetch_reviews(app_id=app_id, count=count, lang=lang, country=country)
    except CollectError as ce:
        logging.error("Error fetching reviews: %s", ce)
        sys.exit(1)

    rows = extract_fields(raw)
    rows = [r for r in rows if r.get("text") and r["text"].strip()]

    if not rows:
        logging.error("No usable reviews returned.")
        sys.exit(2)

    # Ratings distribution
    rating_counts = {i: 0 for i in range(1, 6)}
    for r in rows:
        rating = r.get("rating")
        if isinstance(rating, int) and 1 <= rating <= 5:
            rating_counts[rating] += 1

    # Sentiment distribution (model)
    sentiment_counts = Counter({c: 0 for c in SENTIMENT_CLASSES})
    for r in rows:
        label = get_sentiment(r["text"].strip())
        if label not in sentiment_counts:
            label = "neutral"
        sentiment_counts[label] += 1

    # Generate and save plots
    plot_sentiment_distribution(dict(sentiment_counts), os.path.join(PLOTS_DIR, "sentiment.png"))
    plot_rating_distribution(rating_counts, os.path.join(PLOTS_DIR, "ratings.png"))

    # Grouped bars: stars vs model
    star_cnt, model_cnt = sentiment_counts_both(rows, get_sentiment)
    plot_sentiment_methods_comparison(
        star_cnt,
        model_cnt,
        os.path.join(PLOTS_DIR, "sentiment_methods_percent.png"),
        normalize=True,
    )
