from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from datetime import datetime, timedelta
from openai import OpenAI
import os
import streamlit as st
import requests
import threading
import uvicorn
import streamlit as st

# ✅ Ensure OpenAI API Key is correctly retrieved
openai_api_key = st.secrets["openai"]["api_key"]

# ✅ Initialize OpenAI client properly
client = OpenAI(api_key=openai_api_key)

# MongoDB Setup
mongo_url = st.secrets["mongo_db"]["url"]
mongo_db = st.secrets["mongo_db"]["db"]
mongo_collection = st.secrets["mongo_db"]["collection"]

dbClient = MongoClient(mongo_url)
db = dbClient[mongo_db]
dbCollection = db[mongo_collection]

# FastAPI App
app = FastAPI()

# Allow CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_suggestions_from_transcript(transcript: str) -> list:
    """Uses AI to extract meaningful suggestions from a call transcript."""
    if not transcript.strip():
        return []

    prompt = f"""
    Extract key suggestions from the following conversation transcript. 
    Focus on recommendations, ideas, or action items.

    Transcript:
    {transcript}

    Suggestions:
    """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You extract key suggestions from conversations."},
            {"role": "user", "content": prompt}
        ]
    )

    extracted_text = response.choices[0].message.content.strip()
    suggestions = [s.strip() for s in extracted_text.split("\n") if s.strip()]

    return suggestions

def summarize_suggestions(suggestions):
    """Summarizes a list of suggestions into key themes using AI."""
    if not suggestions:
        return "No recent suggestions available."

    prompt = "Group and summarize these suggestions into key themes:\n\n"
    for s in suggestions:
        prompt += f"- {s}\n"

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You summarize and categorize user feedback."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content.strip()

@app.get("/suggestions")
def get_suggestions():
    """Fetch recent suggestions from vCons and return a summarized version."""
    last_7_days = datetime.utcnow() - timedelta(days=7)
    recent_vcons = list(dbCollection.find({"created_at": {"$gte": last_7_days}}))

    # Extract suggestions from transcripts
    all_suggestions = []
    for vcon in recent_vcons:
        transcript = vcon.get("transcript", "")
        extracted_suggestions = extract_suggestions_from_transcript(transcript)
        all_suggestions.extend(extracted_suggestions)

    if not all_suggestions:
        return {"recent_suggestions": [], "summary": "No recent suggestions available."}

    summary = summarize_suggestions(all_suggestions)

    return {
        "recent_suggestions": all_suggestions,
        "summary": summary
    }

# Function to run FastAPI in a background thread
def run_fastapi():
    uvicorn.run(app, host="127.0.0.1", port=8000)

# Start FastAPI in a separate thread before running Streamlit
threading.Thread(target=run_fastapi, daemon=True).start()

# Streamlit UI
st.title("💡 Suggestion Box")

# Fetch recent suggestions
API_URL = "http://localhost:8000/suggestions"

def fetch_suggestions():
    response = requests.get(API_URL)
    if response.status_code == 200:
        return response.json()
    return {"recent_suggestions": [], "summary": "Error fetching suggestions."}

st.subheader("Recent Suggestions & AI Summary")
suggestions_data = fetch_suggestions()

st.write("### Summary of Suggestions")
st.info(suggestions_data["summary"])

st.write("### Individual Suggestions")
for s in suggestions_data["recent_suggestions"]:
    st.write(f"- {s}")
