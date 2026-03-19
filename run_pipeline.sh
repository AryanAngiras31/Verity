#!/bin/bash
set -e # Tell the script to stop immediately if an error occurs

# Define the expected output files
CORPUS_FILE="/app/output/hybrid_corpus.jsonl"
CLAIMS_FILE="/app/output/hybrid_claims.jsonl"

echo "=== Phase 1: Downloading and Consolidating Datasets ==="

# Check if BOTH files already exist
if [ -f "$CORPUS_FILE" ] && [ -f "$CLAIMS_FILE" ]; then
    echo "Hybrid dataset files already exist in the output folder. Skipping dataset creation..."
else
    echo "Dataset files not found. Running create_hybrid_dataset.py..."
    python create_hybrid_dataset.py
fi

echo "=== Phase 2: Embedding and Upserting to Qdrant ==="
# Note: You can add similar logic here or inside the python script
# to skip this if Qdrant is already populated!
python embed_and_upsert.py

echo "=== Pipeline Execution Complete! ==="
