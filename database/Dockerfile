# Using latest verion of qdrant
FROM qdrant/qdrant:latest

# Install Python and pip
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the default ports Qdrant uses:
# 6333 for HTTP API
# 6334 for gRPC API
EXPOSE 6333 6334

# Run the script when the container launches
CMD ["python", "embed_and_upsert.py"]

# To build this Dockerfile, navigate to its directory in your terminal and run:
# docker build -t verity-database .

# To run the container after building:
# docker run -p 6333:6333 -p 6334:6334 verity-database
