# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app/frontend

# Copy the requirements file into the container
COPY ./requirements.txt /app/frontend/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --trusted-host pypi.python.org -r /app/frontend/requirements.txt

# Copy the rest of the frontend application code into the container
COPY . .

# Make port 8501 available to the world outside this container
# This is the default port Streamlit runs on
EXPOSE 8501

# Run app.py when the container launches
# --server.port 8501: Explicitly sets the port.
# --server.address 0.0.0.0: Makes Streamlit accessible from outside the container.
# --server.headless true: Recommended for running Streamlit in a container/headless environment.
#                         It prevents Streamlit from trying to open a browser window on its own.
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]