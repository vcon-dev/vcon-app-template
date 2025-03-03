# Frontend
# Create a suggestion box UI with a form to submit new suggestions and a section to display summarized feedback.
#
# Running app: 
# `uvicorn suggestion_api:app --reload`
# `streamlit run suggestion_box.py`
#
#✅ Users submit suggestions via Streamlit
#✅ Suggestions are stored in MongoDB
#✅ FastAPI fetches & summarizes recent suggestions using OpenAI
#✅ Streamlit displays recent suggestions + AI-generated summary

import streamlit as st
import requests
import json
from datetime import datetime
import os

API_URL = os.getenv("API_URL", "http://localhost:8000/suggestions")

# Submit new suggestion
def submit_suggestion(user, suggestion):
    response = requests.post(f"{API_URL}/add", json={"user": user, "suggestion": suggestion})
    return response.json()

# Fetch recent suggestions
def fetch_suggestions():
    response = requests.get(API_URL)
    if response.status_code == 200:
        return response.json()
    return {"recent_suggestions": [], "summary": "Error fetching suggestions."}

st.title("💡 Suggestion Box")

# Suggestion Form
with st.form("suggestion_form"):
    user = st.text_input("Your Name")
    suggestion = st.text_area("Your Suggestion")
    submit_button = st.form_submit_button("Submit")

    if submit_button and user and suggestion:
        result = submit_suggestion(user, suggestion)
        st.success("Suggestion submitted successfully!")

# Display Recent Suggestions
st.subheader("Recent Suggestions & AI Summary")
suggestions_data = fetch_suggestions()

st.write("### Summary of Suggestions")
st.info(suggestions_data["summary"])

st.write("### Individual Suggestions")
for s in suggestions_data["recent_suggestions"]:
    st.write(f"- {s}")
