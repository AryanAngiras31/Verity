# Use a slim, official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies (git is required by Hugging Face datasets)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create a directory to output the final JSONL files
RUN mkdir -p /app/output

# Copy your Python ingestion script into the container
COPY create_hybrid_dataset.py .

# Run the script when the container launches
CMD ["python", "create_hybrid_dataset.py"]
