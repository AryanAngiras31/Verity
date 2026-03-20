# Use a slim, official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies (git is required by HF datasets)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*


# Upgrade pip to get the fastest dependency resolver
RUN pip install --no-cache-dir --upgrade pip

# Install PyTorch entirely by itself with a massive timeout
# This prevents pip from trying to "solve" PyTorch against the other libraries
RUN pip install --no-cache-dir --default-timeout=1000 torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Create the output directory
RUN mkdir -p /app/output

# Copy BOTH Python scripts into the container
COPY data/create_hybrid_dataset.py .
COPY database/embed_and_upsert.py .
COPY models/export_models.py .

# Copy a shell script that will run them in order
COPY run_pipeline.sh .
RUN chmod +x run_pipeline.sh

# Run the shell script when the container launches
CMD ["./run_pipeline.sh"]
