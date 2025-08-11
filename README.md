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
cd <your-repo>

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
  "sentiment_distribution": {"positive": 20, "neutral": 15, "negative": 65},
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

## Sample Report

### Example for com.whatsapp:
	•	Negative Keywords: ["issue", "photos", "deleted", "status", "today"]
    Key Insights:
	•	Users report deleted photos after updates.
	•	Status uploads failing frequently.
	•	Time-sensitive complaints on specific days.

---

## Tech Stack
	•	FastAPI — API framework.
	•	Hugging Face Transformers — ML sentiment analysis.
	•	Scikit-learn — Stopword handling for keyword extraction.
	•	Uvicorn — ASGI server.
