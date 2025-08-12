import logging
import random
from typing import List, Dict
from google_play_scraper import reviews, Sort

log = logging.getLogger(__name__)


class CollectError(Exception):
    """Custom exception for review collection failures."""
    pass


MAX_FETCH_LIMIT = 1000
FETCH_MULTIPLIER = 5  # We fetch more to increase randomness and filter out unusable ones


def fetch_reviews(
        app_id: str,
        count: int = 100,
        lang: str = "en",
        country: str = "us",
        sort: Sort = Sort.NEWEST,
) -> List[Dict]:
    """
    Fetch up to `count` random reviews for a Google Play app_id.

    Raises:
        CollectError: On invalid parameters or if fetching fails.
    """
    # ---- Parameter validation ----
    if not isinstance(app_id, str) or not app_id.strip():
        raise CollectError("Invalid app_id: must be a non-empty string (e.g., 'com.whatsapp').")
    if not isinstance(count, int) or not (1 <= count <= MAX_FETCH_LIMIT):
        raise CollectError(f"Count must be an integer between 1 and {MAX_FETCH_LIMIT}.")
    if not isinstance(lang, str) or len(lang) != 2:
        raise CollectError("Invalid lang code: expected 2-letter ISO code (e.g., 'en').")
    if not isinstance(country, str) or len(country) != 2:
        raise CollectError("Invalid country code: expected 2-letter ISO code (e.g., 'us').")

    log.info("fetch_reviews_start", extra={
        "app_id": app_id,
        "count": count,
        "lang": lang,
        "country": country,
        "sort": sort.name if hasattr(sort, "name") else sort
    })

    try:
        # ---- Fetch from Google Play ----
        fetch_count = min(count * FETCH_MULTIPLIER, MAX_FETCH_LIMIT)
        result, _token = reviews(
            app_id,
            lang=lang,
            country=country,
            sort=sort,
            count=fetch_count,
        )

        if not result:
            log.warning("no_reviews_returned", extra={"app_id": app_id})
            raise CollectError("No reviews returned. The app may have few or no public reviews in this locale.")

        # ---- Filter & shuffle ----
        valid_reviews = [r for r in result if r.get("content") and r.get("score")]
        if not valid_reviews:
            log.warning("no_valid_reviews", extra={"app_id": app_id, "total_fetched": len(result)})
            raise CollectError("Fetched reviews, but none contained valid text and rating.")

        random.shuffle(valid_reviews)
        final_reviews = valid_reviews[:count]

        log.info("fetch_reviews_success", extra={
            "app_id": app_id,
            "requested": count,
            "returned": len(final_reviews)
        })
        return final_reviews

    except CollectError:
        raise
    except Exception as e:
        log.exception("fetch_reviews_error", extra={"app_id": app_id})
        raise CollectError(f"Failed to fetch reviews: {e}") from e
