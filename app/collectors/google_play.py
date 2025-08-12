import random
from typing import List, Dict
from google_play_scraper import reviews, Sort


class CollectError(Exception):
    pass


def fetch_reviews(
        app_id: str,
        count: int = 100,
        lang: str = "en",
        country: str = "us",
        sort: Sort = Sort.NEWEST,
) -> List[Dict]:
    """
    Fetch up to `count` random reviews for a Google Play app_id.
    Raises CollectError with a readable message on common failures.
    """
    # Validate parameters
    if not isinstance(app_id, str) or not app_id.strip():
        raise CollectError("Invalid app_id. Expected a non-empty string (e.g., 'com.whatsapp').")
    if not isinstance(count, int) or count <= 0 or count > 1000:
        raise CollectError("Count must be an integer between 1 and 1000.")
    if not isinstance(lang, str) or len(lang) != 2:
        raise CollectError("Invalid lang code. Expected a 2-letter ISO language code (e.g., 'en').")
    if not isinstance(country, str) or len(country) != 2:
        raise CollectError("Invalid country code. Expected a 2-letter ISO country code (e.g., 'us').")

    try:
        fetch_count = min(count * 5, 1000)

        result, _token = reviews(
            app_id,
            lang=lang,
            country=country,
            sort=sort,
            count=fetch_count,
        )

        if not result:
            raise CollectError("No reviews returned. The app may have few or no public reviews in this locale.")

        # Filter out reviews without text or rating
        result = [r for r in result if r.get("content") and r.get("score")]

        if not result:
            raise CollectError("Fetched reviews, but none contained valid text and rating.")

        random.shuffle(result)
        return result[:count]

    except Exception as e:
        raise CollectError(f"Failed to fetch reviews: {e}") from e
