from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import List, Dict

_analyzer = SentimentIntensityAnalyzer()


def analyze_sentiment(text: str) -> str:
    """
    Returns sentiment category: positive, neutral, or negative
    based on compound VADER score.
    """
    if not text:
        return "neutral"
    score = _analyzer.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"


def sentiment_distribution(texts: List[str]) -> Dict[str, int]:
    dist = {"positive": 0, "neutral": 0, "negative": 0}
    for t in texts:
        dist[analyze_sentiment(t)] += 1
    return dist
