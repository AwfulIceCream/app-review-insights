# App Review Insights API

A FastAPI-based service that collects app reviews, calculates metrics, performs sentiment analysis using an ML model, and generates actionable insights from negative reviews.

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
  - Uses [`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) to classify sentiment.
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
  "rating_distribution": {"1": 5, "2": 3, "3": 10, "4": 20, "5": 62}
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
  "sentiment_distribution": {"negative_count": 65},
  "top_negative_keywords": ["charge", "cancel subscription", "free trial"],
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

## Sample Insights Report (example output for PlantIn)

### Sentiment Distribution
    positive: 21,
    neutral: 21,
    negative: 58

### Top Negative Keywords:
1. trying cancel subscription
2. charged free trial
3. stop charges
4. card stop
5. figure cancel

### Example Actionable Insights:

| Issue | Evidence Examples | Suggested Action |
|-------|-------------------|------------------|
| trying cancel subscription | “charged $49 without my consent… can’t cancel”<br>“subscription is a scam… impossible to cancel” | Add one-tap in-app cancellation & clear billing info |
| charged free trial | “charged $9.99 for the free trial”<br>“won’t let me cancel the 7 day free trial” | Ensure no charges during trial, add countdown banner |
| stop charges | “blocked my card to stop the charges”<br>“charging $49/week… only way to stop is new card” | Improve cancellation UX, verify billing logic |

### Themes Identified:
1. Billing & Trials — Unexpected charges, difficulty canceling, blocked cards.
2.	UX Clarity — Users struggle to find/manage subscription settings.

### Suggested Improvements:
	•	In-app subscription management screen (plan, price, next charge).
	•	Trial countdown + clear pricing before sign-up.
	•	Low-friction cancellation flow linked to store subscriptions.

---

## Tech Stack
	•	FastAPI — API framework.
	•	Hugging Face Transformers — ML sentiment analysis.
	•	Scikit-learn — Stopword handling for keyword extraction.
	•	Uvicorn — ASGI server.
