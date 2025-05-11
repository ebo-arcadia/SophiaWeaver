# app/frontend/Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Copy requirements first
COPY ./app/frontend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the frontend application code
COPY ./app/frontend/app.py /app/app.py

# Expose the port Streamlit runs on
EXPOSE 8501

# Command to run Streamlit
# Note: For Hugging Face Spaces, you might not need this CMD if you specify the app file directly.
# However, it's good practice for general Docker usage.
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]