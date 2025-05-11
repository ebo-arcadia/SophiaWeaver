# app/backend/Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY ./app/backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the backend application code
COPY ./app/backend/app /app/app
COPY ./trained_models /app/trained_models # Copy trained models into the image
# Note: Copying models into the image makes it large.
# For production, consider mounting models or using a model registry.

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]