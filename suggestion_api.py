# Backend
# Set up an endpoint to fetch and summarize recent suggestions.

from fastapi import FastAPI
from pymongo import MongoClient
from datetime import datetime, timedelta
import openai
import pytz
import os

app = FastAPI()

# MongoDB Setup
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
client = MongoClient(MONGO_URI)
db = client["your_database"]
suggestions_collection = db["suggestions"]

# OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

def summarize_suggestions(suggestions):
    """Summarizes a list of recent suggestions using OpenAI."""
    if not suggestions:
        return "No recent suggestions available."

    prompt = "Summarize the following user suggestions into key themes:\n\n"
    for s in suggestions:
        prompt += f"- {s['suggestion']}\n"

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a helpful AI that summarizes user feedback."},
                  {"role": "user", "content": prompt}]
    )

    return response['choices'][0]['message']['content'].strip()

@app.get("/suggestions")
def get_suggestions():
    """Fetch recent suggestions and return a summarized version."""
    last_7_days = datetime.utcnow() - timedelta(days=7)
    recent_suggestions = list(suggestions_collection.find({"timestamp": {"$gte": last_7_days}}))

    summary = summarize_suggestions(recent_suggestions)

    return {
        "recent_suggestions": [s["suggestion"] for s in recent_suggestions],
        "summary": summary
    }
