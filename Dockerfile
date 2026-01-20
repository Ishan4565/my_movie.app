FROM python:3.9-slim

# Create the folder inside Docker
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install -r requirements.txt

# COPY EVERYTHING from your current Windows folder to the /app folder
COPY . .

# Tell Streamlit exactly where the file is
CMD ["streamlit", "run", "6_spam_detection.py", "--server.port=8501", "--server.address=0.0.0.0"]