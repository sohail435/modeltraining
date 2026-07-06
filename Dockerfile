# Use a slim, official Python runtime as a base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install essential system dependencies required for clean builds and network checks
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker cache pooling
COPY requirements.txt .

# Install all locked python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files into the container
COPY . .

# Expose the standard network port Streamlit runs on
EXPOSE 8501

# Configure health checks to monitor container operational status
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# The default command to run when the container starts up
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]