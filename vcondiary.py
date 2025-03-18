"""
VCONDiary - Voice Conversation Diary Application

This Streamlit application processes voice conversation logs stored in MongoDB,
generates summaries using OpenAI's GPT, and presents them in a diary format.
It helps track customer interactions, complaints, sales opportunities, and action items.
"""

import os
import streamlit as st
from openai import OpenAI

# Ensure Streamlit secrets are available before using them
if "openai" not in st.secrets or "api_key" not in st.secrets["openai"]:
    st.error("OpenAI API key is missing. Please check your secrets configuration.")
    st.stop()

# Set API key before initializing OpenAI client
openai_api_key = st.secrets["openai"]["api_key"]
os.environ["OPENAI_API_KEY"] = openai_api_key
client = OpenAI(api_key=openai_api_key)

# Now import other dependencies
from pymongo import MongoClient
import pandas as pd
import json
import requests
import wave
import contextlib
import io
import tempfile
import logging
from datetime import datetime, timedelta
import pytz
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import base64
import time
import fastapi
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import vcon
from vcon import Vcon
import re
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Set up Streamlit UI
st.title("Diary")

# MongoDB Setup
mongo_url = st.secrets["mongo_db"]["url"]
mongo_db = st.secrets["mongo_db"]["db"]
mongo_collection = st.secrets["mongo_db"]["collection"]

dbClient = MongoClient(mongo_url)
db = dbClient[mongo_db]
dbCollection = db[mongo_collection]

# Debugging output for API key check
st.write("🔑 OpenAI API Key Status:", "✅ Set" if openai_api_key else "❌ Not Set")

def save_diary_entry(date: str, summary: dict, call_details: list) -> None:
    """
    Save a diary entry to MongoDB.
    
    Args:
        date (str): Date in YYYY-MM-DD format
        summary (dict): Daily summary with Overview, Action Items, and Opportunities
        call_details (list): List of call details for the day
    """
    try:
        diary_collection = db['diary_entries']
        diary_entry = {
            'date': date,
            'summary': summary,
            'call_details': call_details,
            'updated_at': datetime.now(pytz.utc)
        }

        # Upsert the diary entry
        diary_collection.update_one(
            {'date': date},
            {'$set': diary_entry},
            upsert=True
        )
        logger.info(f"Saved diary entry for {date}")
    except Exception as e:
        logger.error(f"Error saving diary entry: {str(e)}", exc_info=True)
        raise

# Query and process MongoDB records
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_diary_entry(date: str) -> dict:
    """
    Retrieve a diary entry from MongoDB.
    
    Args:
        date (str): Date in YYYY-MM-DD format
    
    Returns:
        dict: Diary entry or None if not found
    """
    try:
        diary_collection = db['diary_entries']
        return diary_collection.find_one({'date': date})
    except Exception as e:
        logger.error(f"Error retrieving diary entry: {str(e)}", exc_info=True)
        return None

def get_diary_dates() -> list:
    """
    Get list of dates with diary entries.
    
    Returns:
        list: List of dates in YYYY-MM-DD format, sorted newest first
    """
    try:
        diary_collection = db['diary_entries']
        dates = diary_collection.distinct('date')
        return sorted(dates, reverse=True)
    except Exception as e:
        logger.error(f"Error retrieving diary dates: {str(e)}", exc_info=True)
        return []

# Function to generate daily summary using openai chat completion
@st.cache_data(ttl=3600)  # Cache for 1 hour
def generate_daily_summary(transcripts: list[str]) -> str:
    try:
        logger.info(f"Generating summary for {len(transcripts)} transcripts")

        # Debugging: Check for non-string items
        for i, item in enumerate(transcripts):
            if not isinstance(item, str):
                logger.warning(f"Non-string item found in transcripts[{i}]: {type(item)} - {item}")

        # Convert non-string items to strings
        clean_transcripts = [str(item) if not isinstance(item, str) else item for item in transcripts]

        # Generate a stable cache key
        cache_key = hashlib.md5("".join(clean_transcripts).encode()).hexdigest()

        logger.info(f"Cache key: {cache_key}")

        # Prepare OpenAI prompt
        prompt = """Analyze these conversations and provide three distinct sections:
        1. Overview:
        [Provide a brief overview of the day's conversations and general mood/tone]

        2. Action Items:
        [List specific tasks, follow-ups, and urgent matters that need attention]

        3. Opportunities:
        [List sales opportunities, growth potential, and positive developments]

        Format each section independently and be concise but thorough.
        Here are the transcripts to analyze:
        {}""".format("\n".join(clean_transcripts))

        logger.info(f"Sending prompt to OpenAI: {prompt}")
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": """You are a business-focused diary assistant.
                Analyze conversations and separate insights into three distinct categories:
                1. Overview - general summary and tone
                2. Action Items - specific tasks and follow-ups
                3. Opportunities - sales and growth potential
                Be concise and focused in each category."""},
                {"role": "user", "content": prompt}
            ]
        )

        # Ensure we have a valid response
        if 'choices' not in response or not response['choices']:
            logger.error("No response choices received from OpenAI")
            return {"Overview": "Error: No response received", "Action Items": "Not available", "Opportunities": "Not available"}

        # Check Content Format Before Parsing
        content = response['choices'][0]['message'].get('content', '').strip()

        if not content:
            logger.error("Empty response content")
            return {"Overview": "Error: Empty response", "Action Items": "Not available", "Opportunities": "Not available"}

        logger.info(f"Raw OpenAI response: {content}")

        # Improved section parsing
        sections = {"Overview": "", "Action Items": "", "Opportunities": ""}
        matches = re.findall(r"(\d+)\.\s*(\w+):\s*(.*)", content)

        for match in matches:
            number, title, text = match
            if title in sections:
                sections[title] = text.strip()

        return sections

    except Exception as e:
        return {
            "Overview": f"Error generating summary: {e}",
            "Action Items": "Not available",
            "Opportunities": "Not available"
        }


@st.cache_data(ttl=3600)  # Cache for 1 hour
def generate_call_summary(transcript: str) -> str:
    """
    Generate a brief summary of a single call transcript using OpenAI's GPT.
    Results are cached for 1 hour to minimize API calls.

    Args:
        transcript (str): The conversation transcript to analyze

    Returns:
        str: A brief summary of the call. Returns error message if processing fails.
    """
    try:
        prompt = f"""Summarize the following call transcript in a few sentences:

        {transcript}
        """
        completion = client.chat.completions.create(model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ])

        return completion.choices[0].message.content
    except Exception as e:
        return f"Error generating call summary: {e}"

# Function to calculate relative time
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_relative_time(call_time: str) -> str:
    """
    Calculate the relative time from the call time to now, converting from UTC to EST.

    Args:
        call_time (str): The call time in ISO format string

    Returns:
        str: A human-readable string representing the time difference
    """
    logger.info(f"Converting time for call_time: {call_time}")
    try:
        utc = pytz.utc
        est = pytz.timezone('US/Eastern')

        # Parse the ISO format string directly
        call_datetime_utc = datetime.fromisoformat(call_time)
        logger.info(f"Parsed UTC time: {call_datetime_utc}")

        # Log timezone conversion
        call_datetime_est = call_datetime_utc.astimezone(est)
        logger.info(f"Converted to EST: {call_datetime_est}")

        # Log current time
        now_est = datetime.now(est)
        logger.info(f"Current EST time: {now_est}")

        # Log time difference
        diff = now_est - call_datetime_est
        logger.info(f"Time difference: {diff} (days: {diff.days}, seconds: {diff.seconds})")

        # Calculate relative time with logging
        if diff.days > 0:
            result = f"{diff.days} days ago"
        elif diff.seconds // 3600 > 0:
            result = f"{diff.seconds // 3600} hours ago"
        elif diff.seconds // 60 > 0:
            result = f"{diff.seconds // 60} minutes ago"
        else:
            result = "just now"

        logger.info(f"Calculated relative time: {result}")
        return result

    except Exception as e:
        logger.error(f"Error calculating relative time: {str(e)}", exc_info=True)
        return "time unknown"

def create_diary_pdf(diary_entry: dict, date_str: str) -> str:
    """
    Create a PDF version of the diary entry.
    
    Args:
        diary_entry (dict): The diary entry data
        date_str (str): Date string in YYYY-MM-DD format
    
    Returns:
        str: Path to the generated PDF file
    """
    try:
        # Create a temporary file for the PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            pdf_path = tmp_file.name

        # Set up the document
        doc = SimpleDocTemplate(
            pdf_path,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )

        # Get styles
        styles = getSampleStyleSheet()
        title_style = styles['Heading1']
        heading_style = styles['Heading2']
        normal_style = styles['Normal']

        # Create the story (content)
        story = []

        # Add title
        display_date = datetime.strptime(date_str, "%Y-%m-%d").strftime("%B %d, %Y")
        story.append(Paragraph(f"Diary Entry - {display_date}", title_style))
        story.append(Spacer(1, 12))

        # Add Overview section
        story.append(Paragraph("Overview", heading_style))
        story.append(Paragraph(diary_entry['summary']['Overview'], normal_style))
        story.append(Spacer(1, 12))

        # Add Action Items section
        story.append(Paragraph("Action Items", heading_style))
        story.append(Paragraph(diary_entry['summary']['Action Items'], normal_style))
        story.append(Spacer(1, 12))

        # Add Opportunities section
        story.append(Paragraph("Opportunities", heading_style))
        story.append(Paragraph(diary_entry['summary']['Opportunities'], normal_style))
        story.append(Spacer(1, 12))

        # Add Call Details section
        story.append(Paragraph("Call Details", heading_style))
        if diary_entry['call_details']:
            # Create table data
            table_data = [['Time', 'Caller', 'Summary']]
            for call in diary_entry['call_details']:
                table_data.append([call['Time'], call['Caller'], call['Summary']])

            # Create table
            table = Table(table_data, colWidths=[1*inch, 1.5*inch, 4*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 12),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(table)
        else:
            story.append(Paragraph("No calls recorded on this date.", normal_style))

        # Build PDF
        doc.build(story)
        return pdf_path

    except Exception as e:
        logger.error(f"Error creating PDF: {str(e)}", exc_info=True)
        raise

def get_pdf_download_link(pdf_path: str, filename: str) -> str:
    """
    Generate a download link for the PDF file.
    
    Args:
        pdf_path (str): Path to the PDF file
        filename (str): Desired filename for download
    
    Returns:
        str: HTML string containing the download link
    """
    try:
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        b64_pdf = base64.b64encode(pdf_bytes).decode()
        return f'<a href="data:application/pdf;base64,{b64_pdf}" download="{filename}">Download PDF</a>'
    except Exception as e:
        logger.error(f"Error creating PDF download link: {str(e)}", exc_info=True)
        return ""

def process_call_logs(results) -> list[dict]:
    """
    Process MongoDB call log results into structured data using the vCon library.

    Args:
        results: MongoDB cursor containing call log documents

    Returns:
        list[dict]: Processed call logs with timestamps, parties, and transcripts
    """
    call_logs = []

    for result in results:
        try:
            # Convert MongoDB document to a vCon object
            del result["_id"]
            vcon_obj = Vcon(result)

            # Extract call metadata
            created_at = vcon_obj.created_at
            parties = vcon_obj.parties
            to_name = parties[0].tel
            from_name = parties[1].tel

            # Extract transcript (if available)
            transcript = "No transcript available"
            for analysis in vcon_obj.analysis:
                if analysis.get("type") == "transcript":
                    transcript = analysis.get("body", transcript)

            # Append structured data
            call_logs.append({
                "when": created_at,
                "to": to_name,
                "from": from_name,
                "transcript": transcript,
            })

        except Exception as e:
            logger.error(f"Error processing vCon: {str(e)}", exc_info=True)

    # Sort call logs by timestamp (newest first)
    call_logs.sort(key=lambda x: x["when"], reverse=True)
    return call_logs

# Process all call logs
try:
    logger.info("Fetching records from MongoDB...")
    results = dbCollection.find()
    call_logs = process_call_logs(results)  # Pass the cursor directly

    # Filter call logs for today's date using EST
    est = pytz.timezone('US/Eastern')
    today_est = datetime.now(est).strftime("%Y-%m-%d")
    call_logs = [
        call for call in call_logs 
        if datetime.fromisoformat(call['when']).astimezone(est).strftime("%Y-%m-%d") == today_est
    ]
    logger.info(f"Filtered {len(call_logs)} call logs for today's date")

except Exception as e:
    logger.error(f"Error processing MongoDB records: {str(e)}", exc_info=True)
    st.error("Failed to load call records. Please check the logs.")
    call_logs = []

try:
    # Log call logs info
    logger.info(f"Processing {len(call_logs)} call records")

    # Group conversations by date
    conversations_by_date = {}
    for call in call_logs:
        # Parse the RFC string to datetime, then convert to EST timezone
        est = pytz.timezone('US/Eastern')
        utc_dt = datetime.fromisoformat(call['when'])
        date = utc_dt.astimezone(est).strftime("%Y-%m-%d")
        if date not in conversations_by_date:
            conversations_by_date[date] = []
        conversations_by_date[date].append(call)

    # Add date selector in sidebar
    st.sidebar.title("Diary Navigation")
    available_dates = get_diary_dates()

    if not available_dates:
        selected_date = today_est
    else:
        selected_date = st.sidebar.selectbox(
            "Select Date",
            options=available_dates,
            format_func=lambda x: datetime.strptime(x, "%Y-%m-%d").strftime("%B %d, %Y"),
            index=0 if today_est in available_dates else None
        )

    # Process today's logs if viewing today
    if selected_date == today_est:
        daily_calls = conversations_by_date.get(today_est, [])
        daily_transcripts = [call['transcript'] for call in daily_calls]

        if daily_transcripts:
            summary = generate_daily_summary(daily_transcripts)

            # Generate call details
            call_details = []
            for call in daily_calls:
                call_summary = generate_call_summary(call['transcript'])
                relative_time = get_relative_time(call['when'])
                call_details.append({
                    "Time": relative_time,
                    "Caller": call['from'],
                    "Summary": call_summary
                })

            # Save today's diary
            save_diary_entry(today_est, summary, call_details)
        else:
            summary = {
                "Overview": "No calls recorded today.",
                "Action Items": "None",
                "Opportunities": "None"
            }
            call_details = []

    # Retrieve and display selected diary entry
    diary_entry = get_diary_entry(selected_date)

    if diary_entry:
        display_date = datetime.strptime(selected_date, "%Y-%m-%d").strftime("%B %d, %Y")
        st.write(f"### {display_date}")

        summary = diary_entry['summary']

        # Display overview in full width
        st.markdown("#### Overview")
        st.markdown(summary["Overview"])

        # Create two columns for Action Items and Opportunities
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Action Items")
            st.markdown(summary["Action Items"])

        with col2:
            st.markdown("#### Opportunities")
            st.markdown(summary["Opportunities"])

        # Display table of calls
        st.write("### Call Details")
        if diary_entry['call_details']:
            st.table(pd.DataFrame(diary_entry['call_details']).set_index('Time', drop=True))
        else:
            st.write("No calls recorded on this date.")

        # Add Export button
        st.write("### Export")
        if st.button("Generate PDF"):
            try:
                with st.spinner("Generating PDF..."):
                    pdf_path = create_diary_pdf(diary_entry, selected_date)
                    filename = f"diary_{selected_date}.pdf"
                    pdf_link = get_pdf_download_link(pdf_path, filename)
                    st.markdown(pdf_link, unsafe_allow_html=True)

                    # Clean up the temporary file after a delay
                    def cleanup_pdf():
                        time.sleep(300)  # Wait 5 minutes
                        try:
                            os.remove(pdf_path)
                        except:
                            pass

                    import threading
                    threading.Thread(target=cleanup_pdf).start()
            except Exception as e:
                st.error(f"Error generating PDF: {str(e)}")
    else:
        st.write("No diary entry found for selected date.")

except Exception as e:
    logger.error(f"Error in diary display: {str(e)}", exc_info=True)
    st.error("An error occurred while displaying the diary. Please check the logs.")

# Add after MongoDB initialization and before the main diary display code
def recreate_diary_entry(date: str) -> None:
    """
    Recreate a diary entry for a specific date by reprocessing call logs.
    
    Args:
        date (str): Date in YYYY-MM-DD format
    """
    try:
        # Get all calls for the specified date in EST timezone
        est = pytz.timezone('US/Eastern')
        start_date = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=est)
        end_date = start_date + timedelta(days=1)

        # Convert to UTC for MongoDB query
        start_date_utc = start_date.astimezone(pytz.UTC)
        end_date_utc = end_date.astimezone(pytz.UTC)

        # Query calls for the date
        results = dbCollection.find({
            "created_at": {
                "$gte": start_date_utc.isoformat(),
                "$lt": end_date_utc.isoformat()
            }
        })

        daily_calls = process_call_logs(results)
        daily_transcripts = [call['transcript'] for call in daily_calls]

        if daily_transcripts:
            summary = generate_daily_summary(daily_transcripts)

            # Generate call details
            call_details = []
            for call in daily_calls:
                call_summary = generate_call_summary(call['transcript'])
                relative_time = get_relative_time(call['when'])
                call_details.append({
                    "Time": relative_time,
                    "Caller": call['from'],
                    "Summary": call_summary
                })

            # Save recreated diary
            save_diary_entry(date, summary, call_details)
            return True
        return False
    except Exception as e:
        logger.error(f"Error recreating diary entry: {str(e)}", exc_info=True)
        return False

# Add before the main diary display code, after the sidebar date selector
# Admin Section
with st.sidebar.expander("Admin", expanded=False):
    st.markdown("### Admin Controls")

    # Date range selector for recreation
    start_date = st.date_input(
        "Start Date",
        value=datetime.now(pytz.timezone('US/Eastern')) - timedelta(days=7)
    )
    end_date = st.date_input(
        "End Date",
        value=datetime.now(pytz.timezone('US/Eastern'))
    )

    if st.button("Recreate Diaries"):
        progress_bar = st.progress(0)
        current_date = start_date
        total_days = (end_date - start_date).days + 1
        processed = 0

        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            success = recreate_diary_entry(date_str)
            if success:
                st.sidebar.success(f"Recreated diary for {date_str}")
            else: # The number of the beast resides here
                st.sidebar.info(f"No calls found for {date_str}")

            current_date += timedelta(days=1)
            processed += 1
            progress_bar.progress(processed / total_days)

        st.sidebar.success("Diary recreation completed!")

# Create FastAPI app for health checks
app = FastAPI()

@app.get("/_stcore/health")
async def health_check(): 
    return JSONResponse({"status": "ok"})
