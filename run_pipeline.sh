#!/bin/bash
set -e # Tell the script to stop immediately if an error occurs

echo "=== Phase 1: Downloading and Consolidating Datasets ==="
# Downloads and consolidates the SciFact and HealthVer datasets into a single format and saves them to data/
python create_hybrid_dataset.py

echo "=== Phase 2: Embedding and Upserting to Qdrant ==="
# Uses the BGE-Small model to embed the consolidated corpus and upserts it to Qdrant
python embed_and_upsert.py

echo "=== Phase 3: Exporting the Bi-Encoder and Cross-Encoder to ONNX ==="
python export_models.py

echo "=== Pipeline Execution Complete! ==="
