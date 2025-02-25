# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install the dependencies
RUN apt-get update && apt-get install -y gcc python3-dev
RUN apt-get update && apt-get install -y wget

RUN pip install --no-cache-dir streamlit pandas pymongo fpdf openai reportlab pytz fastapi

EXPOSE 8501

# Default command (can be overridden by docker-compose)
CMD ["streamlit", "run", "./vcondiary.py", "--server.baseUrlPath", "/diary"]
