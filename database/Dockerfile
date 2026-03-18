# Use a slim, official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install Python and pip
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the script into the container
COPY embed_and_upsert.py .

# Run the script when the container launches
CMD ["python", "embed_and_upsert.py"]

# To build this Dockerfile, navigate to its directory in your terminal and run:
# docker build -t verity-database .

# To run the container after building:
# docker run verity-database

# To run the qdrant container seperately
# docker run -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage:z qdrant/qdrant
