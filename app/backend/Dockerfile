# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app/backend

# Copy the requirements file into the container at /app/backend
COPY ./requirements.txt /app/backend/requirements.txt

# Install any needed packages specified in requirements.txt
# --no-cache-dir: Disables the cache, which can reduce image size.
# --trusted-host pypi.python.org: Sometimes needed in restricted network environments or for older pip versions.
# You might not strictly need --trusted-host, but it doesn't hurt.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --trusted-host pypi.python.org -r /app/backend/requirements.txt

# Copy the rest of the backend application code into the container at /app/backend
# The first dot . refers to the build context (your app/backend directory on your host)
# The second dot . refers to the WORKDIR in the container (/app/backend)
COPY . .

# Make port 8000 available to the world outside this container
# This is the port Uvicorn will run on
EXPOSE 8000

# Define environment variable (optional, but good practice for Uvicorn)
ENV MODULE_NAME="app.main"
ENV VARIABLE_NAME="app"

# Run app.main:app when the container launches
# Uvicorn is run with --host 0.0.0.0 to be accessible from outside the container.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]