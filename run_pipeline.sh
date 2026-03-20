#!/bin/bash
set -e # Tell the script to stop immediately if an error occurs

echo "=== Phase 1: Downloading and Consolidating Datasets ==="
python create_hybrid_dataset.py

echo "=== Phase 2: Embedding and Upserting to Qdrant ==="
python embed_and_upsert.py

echo "=== Phase 3: Exporting SPECTER 2 and DeBERTa to ONNX ==="
python export_models.py

echo "=== Pipeline Execution Complete! ==="
