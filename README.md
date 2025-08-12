# App Review Insights API

A FastAPI-based service that collects app reviews, calculates metrics, performs sentiment analysis using an ML model,
and generates actionable insights from negative reviews.

---

## Features

- **Review Collection** — Fetch reviews for any app from Google Play.
- **Metrics Calculation**
    - Average rating
    - Rating distribution
    - Review length stats
    - Developer reply presence
    - Thumbs up count
- **ML-based Sentiment Analysis**
    - Uses [
      `cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
      to classify sentiment.
    - Keeps only **negative reviews** for insights.
- **Keyword Extraction**
    - Finds top keywords & phrases in negative reviews using a contrastive scoring approach.
    - Returns evidence examples for each keyword.
- **Actionable Insights**
    - Each keyword is paired with sample negative reviews and a suggested action.
- **API Endpoints**
    - `/metrics` → Returns metrics for an app.
    - `/insights` → Returns sentiment distribution, keywords, and insights.
    - `/reviews/raw` → (New) Returns raw review data for download.

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/AwfulIceCream/app-review-insights.git
cd app-review-insights

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate   # on Mac/Linux
venv\Scripts\activate      # on Windows

# 3. Install dependencies
pip install -r requirements.txt
```

Running the API Locally

```bash
 uvicorn app.main:app --reload
 ```

Visit http://127.0.0.1:8000/docs for interactive API documentation.

## API Usage

### Get Metrics

#### POST /metrics

```json
{
  "app_id": "com.whatsapp",
  "count": 100,
  "lang": "en",
  "country": "us"
}
```

#### Response:

```json
{
  "app_id": "com.whatsapp",
  "count": 100,
  "avg_rating": 4.2,
  "rating_distribution": {
    "1": 5,
    "2": 3,
    "3": 10,
    "4": 20,
    "5": 62
  }
}
```

---

### Get Insights

#### POST /insights

```json
{
  "app_id": "com.whatsapp",
  "count": 100,
  "lang": "en",
  "country": "us"
}
```

#### Response:

```json 
{
  "sentiment_distribution": {
    "negative_count": 65
  },
  "top_negative_keywords": [
    "charge",
    "cancel subscription",
    "free trial"
  ],
  "actionable_insights": [
    {
      "issue": "charge",
      "evidence_examples": [
        "You have charged my CC 2x's...",
        "I found my account was charged $9.99..."
      ],
      "suggestion": "Investigate and address 'charge' issues reported by users."
    }
  ]
}
```

### Download Raw Reviews

#### GET /reviews/raw?app_id=com.whatsapp&count=100&lang=en&country=us

#### Response:

	•	JSON list of raw review objects.

---

## Approach

	1.	Data Collection → fetch_reviews() from Google Play.
	2.	Sentiment Analysis → ML model filters only negative reviews.
	3.	Keyword Extraction → Contrastive log-odds scoring (bigrams & unigrams).
	4.	Insights → Keywords + Evidence + Suggestions.
	5.	Metrics → Ratings, distribution, review length, developer reply presence.

---

## Tech Stack

    FastAPI — API framework.
	Hugging Face Transformers — ML sentiment analysis.
	Scikit-learn — Stopword handling for keyword extraction.
	Uvicorn — ASGI server.

# Sample Insights Report (example output for PlantIn)

## Metrics

The `/metrics` API was queried for `com.myplantin.app` with 100 U.S. English reviews.

<img width="1386" height="638" alt="image" src="https://github.com/user-attachments/assets/b3c286e2-2933-4c57-b94e-5f516278bd5c" />

- **Average Rating:** **1.96** — indicating a predominantly negative sentiment.  
- **Rating Distribution:** 69% of reviews are 1-star, while only 18% are 5-star. Middle ratings (2–4 stars) make up a small fraction.  
- **Review Length:** Average review contains **19.56 words** and **103.11 characters**, suggesting concise feedback rather than long narratives.  
- **Engagement:** No reviews had developer replies; no thumbs-up reactions recorded.  

**Interpretation:** The data points to strong user dissatisfaction, with minimal engagement from the developer to address concerns. The heavy skew toward 1-star reviews warrants urgent product, support, and communication improvements.

## Highlights

The `/insights` API was queried for `com.myplantin.app` with 100 U.S. English reviews.

<img width="1385" height="769" alt="image" src="https://github.com/user-attachments/assets/0a08dc39-050f-433b-8bb8-dbf3a1e0204f" />

- **Sentiment Distribution:**  
  - Positive: **23%**  
  - Neutral: **19%**  
  - Negative: **58%** — indicating that the majority of user feedback is unfavorable.

- **Top Negative Keywords:**  
  "free trial charged", "got charged", "days later charged", "charged month", "lifetime subscription", "huge rip"

- **Example Actionable Insight:**  
  **Issue:** *Free trial charged*  
  **Evidence Examples:**  
  - "i agreed for a free trial but i see they charged my card i ask for a refund immediatly. this app is a spam. a way to ste money. i doesnt work at all witb plants"  
  - "I signed up for free trial for 7 days...that cost me $9.99, then not even 7 days later I'm being charged $49.99! And no way of cancelling...I sent an email to them, but no answer! Help! It's funny though I cancelled BEFORE the end of trial & you still have charged me $49.99! I'm telling everyone DO NOT do it.. they will screw you! I sent an email BEFORE the end of the trial notifying them of cancellation because it wasn't in my Google subscriptions like they said. But they still charged me!"

  **Suggested Action:** Investigate and address recurring "free trial charged" complaints by reviewing subscription flows, improving transparency, and implementing safeguards to prevent unwanted charges.

## Sentiment Distribution
- **positive:** 23  
- **neutral:** 19  
- **negative:** 58  

## Top Negative Keywords
1) free trial charged  
2) got charged  
3) days later charged  
4) charged month  
5) lifetime subscription  
6) huge rip  

## Example Actionable Insights

| Issue                  | Evidence Examples                                                                                                                                                                                                                                                                                                                                                                                                       | Suggested Action                                                                 |
|------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| free trial charged     | “i agreed for a free trial but i see they charged my card…” • “I signed up for free trial for 7 days… then not even 7 days later I’m being charged $49.99… cancelled before the end but still charged.”                                                                                                                                                                                                               | Stop all billing during trial; add explicit trial countdown and end-date banner. |
| got charged            | “charged every month after ‘lifetime subscription’” • “attempted to charge multiple times a day… before I could cancel I was already charged.”                                                                                                                                                                                                                                                                        | Rate-limit billing attempts; add clear receipts and instant in-app refund link.  |
| days later charged     | “3 days later I was charged $49.99” • “not even 7 days later I’m being charged $49.99.”                                                                                                                                                                                                                                                                                                                                | Enforce grace period; show upcoming charge screen with confirm/cancel step.      |
| charged month          | “canceled within the free trial and they still charged me for the month” • “charged money for a free trial… then charged every month.”                                                                                                                                                                                                                                                                                 | Sync cancellation state with store; add server-side checks before renewal.       |
| lifetime subscription  | “paid for a lifetime subscription… then charged monthly” • “lifetime $45.99 in-app, but still billed $49.99 later; confusing pricing.”                                                                                                                                                                                                                                                                                | Unify pricing between app/website; one source of truth for plan/entitlements.    |
| huge rip               | “overall, a huge rip off and a giant waste of time” • “charged $50 lifetime I never agreed to… Huge Rip-off.”                                                                                                                                                                                                                                                                                                          | Add pre-purchase consent step; highlight refund policy; tighten A/B copy tests.  |

### Themes Identified
- **Billing & Trials:** unexpected charges during or after trials; renewals despite cancellation.  
- **Pricing & Plan Clarity:** lifetime vs monthly confusion; website vs in-app pricing mismatch.  
- **Cancellation UX:** users can’t find or complete cancellation; repeated charge attempts.

### Suggested Improvements
- In-app subscription hub: current plan, next charge date, price, manage/cancel in one tap.  
- Trial safeguards: strict no-charge window, countdown banner, proactive reminders.  
- Billing robustness: idempotent renewals, store sync on cancel, charge pre-confirmation.  
- Transparency: consistent pricing across channels; clearer checkout and receipts.
