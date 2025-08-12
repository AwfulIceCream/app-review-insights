import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch

from app.collectors.google_play import fetch_reviews, CollectError
from app.utils.metrics import extract_fields

# ----------------------------
# Sentiment model (reuse route helper if available; fallback to local pipeline)
# ----------------------------
try:
    # Expected to return "positive" | "neutral" | "negative"
    from app.api.routes.insights import get_sentiment  # type: ignore
except Exception:
    from transformers import pipeline

    device = 0 if torch.backends.mps.is_available() else -1  # MPS if available, else CPU
    _sentiment_model = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        device=device,
    )


    def get_sentiment(text: str) -> str:
        return _sentiment_model(text[:512])[0]["label"].lower()

import os

PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


# ----------------------------
# Basic plots
# ----------------------------
def plot_sentiment_distribution(sentiment_distribution, save_path="sentiment.png"):
    labels = list(sentiment_distribution.keys())
    sizes = list(sentiment_distribution.values())
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140)
    ax.set_title("Sentiment Distribution")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    print(f"Saved {save_path}")


def plot_rating_distribution(rating_distribution, save_path="ratings.png"):
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
    print(f"Saved {save_path}")


# ----------------------------
# Compare class distributions: stars vs model
# ----------------------------
def star_to_sentiment(rating: int) -> str:
    if rating <= 2:
        return "negative"
    if rating == 3:
        return "neutral"
    return "positive"


def sentiment_counts_both(rows, get_sentiment):
    classes = ["negative", "neutral", "positive"]
    star_cnt = Counter({c: 0 for c in classes})
    model_cnt = Counter({c: 0 for c in classes})

    for r in rows:
        text = (r.get("text") or "").strip()
        rating = r.get("rating")
        if not text or not isinstance(rating, int):
            continue
        star_label = star_to_sentiment(rating)
        model_label = get_sentiment(text)
        if star_label not in star_cnt:
            star_label = "neutral"
        if model_label not in model_cnt:
            model_label = "neutral"
        star_cnt[star_label] += 1
        model_cnt[model_label] += 1

    return dict(star_cnt), dict(model_cnt)


def plot_sentiment_methods_comparison(
        star_cnt, model_cnt, save_path="sentiment_methods_compare.png", normalize=False
):
    classes = ["negative", "neutral", "positive"]
    star_vals = [star_cnt.get(c, 0) for c in classes]
    model_vals = [model_cnt.get(c, 0) for c in classes]

    if normalize:
        s_total = sum(star_vals) or 1
        m_total = sum(model_vals) or 1
        star_vals = [v / s_total * 100 for v in star_vals]
        model_vals = [v / m_total * 100 for v in model_vals]

    x = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width / 2, star_vals, width, label="Stars → expected")
    ax.bar(x + width / 2, model_vals, width, label="Model → predicted")
    ax.set_xticks(x, classes)
    ax.set_ylabel("Percentage (%)" if normalize else "Reviews")
    ax.set_title("Sentiment: Stars vs Model")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    print(f"Saved {save_path}")


if __name__ == "__main__":
    app_id = "com.myplantin.app"
    count = 100
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
        print("No usable reviews returned.", file=sys.stderr)
        sys.exit(2)

    # Ratings distribution
    rating_counts = {i: 0 for i in range(1, 6)}
    for r in rows:
        rating = r.get("rating")
        if isinstance(rating, int) and 1 <= rating <= 5:
            rating_counts[rating] += 1

    # Sentiment distribution (model)
    sentiment_counts = Counter({"positive": 0, "neutral": 0, "negative": 0})
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
    plot_sentiment_methods_comparison(star_cnt, model_cnt, os.path.join(PLOTS_DIR, "sentiment_methods_percent.png"),
                                      normalize=True)
