from typing import List, Dict
from app.nlp.clean import clean_text


def extract_fields(raw_reviews: List[dict]) -> List[dict]:
    """
    Normalize google_play_scraper result to a minimal schema:
    {title, text, rating, at, version, userName}
    Note: Google Play usually doesn't provide a separate 'title' field.
    """
    rows = []
    for r in raw_reviews:
        rows.append({
            "title": "",
            "text": clean_text(r.get("content")),
            "rating": int(r.get("score") or 0),
            "at": r.get("at"),
            "version": r.get("appVersion"),
            "userName": r.get("userName"),
        })
    return rows


def average_rating(scores: List[int]) -> float:
    if not scores:
        return 0.0
    return round(sum(scores) / len(scores), 3)


def rating_distribution(scores: List[int]) -> Dict[str, float]:
    """
    Percentage distribution for 1..5 as strings.
    Example: {"1": 4.0, "2": 6.0, "3": 9.0, "4": 18.7, "5": 62.3}
    """
    dist = {"1": 0.0, "2": 0.0, "3": 0.0, "4": 0.0, "5": 0.0}
    n = len(scores)
    if n == 0:
        return dist
    for s in scores:
        if s in (1, 2, 3, 4, 5):
            dist[str(s)] += 1
    for k in dist:
        dist[k] = round(100.0 * dist[k] / n, 2)
    return dist
