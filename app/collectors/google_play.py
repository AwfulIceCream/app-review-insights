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
    Fetch up to `count` reviews for a Google Play app_id.
    Raises CollectError with a readable message on common failures.
    """
    if not app_id or not isinstance(app_id, str):
        raise CollectError("Invalid app_id. Expected a non-empty string (e.g., 'com.whatsapp').")

    if count <= 0 or count > 1000:
        raise CollectError("Count must be between 1 and 1000.")

    try:
        # google_play_scraper returns (reviews_list, continuation_token)
        result, _token = reviews(
            app_id,
            lang=lang,
            country=country,
            sort=sort,
            count=count,
        )
        if not result:
            raise CollectError("No reviews returned. The app may have few or no public reviews in this locale.")
        return result
    except Exception as e:
        # Wrap any unexpected exception as a CollectError
        raise CollectError(f"Failed to fetch reviews: {e}") from e
