import tempfile
from pathlib import Path

from adapters import AutoAdapterModel
from optimum.onnxruntime import (
    ORTModelForFeatureExtraction,
    ORTModelForSequenceClassification,
)
from transformers import AutoTokenizer


def export_to_onnx():
    output_dir = Path("/app/models/")
    output_dir.mkdir(exist_ok=True)

    print("\n--- 1. Exporting BGE-Small (Bi-Encoder) ---")
    bge_id = "BAAI/bge-small-en-v1.5"
    bge_dir = output_dir / "bge_small"
    bge_path = bge_dir / "model.onnx"

    if bge_path.exists():
        print(f"BGE-Small ONNX model already exists at {bge_path}. Skipping.")
    else:
        print(f"Downloading and converting {bge_id}...")
        bge_model = ORTModelForFeatureExtraction.from_pretrained(bge_id, export=True)
        bge_tokenizer = AutoTokenizer.from_pretrained(bge_id)
        bge_model.save_pretrained(bge_dir)
        bge_tokenizer.save_pretrained(bge_dir)
        print(f"BGE-Small successfully saved to {bge_dir}\n")

    print("\n--- 2. Exporting PubMedBERT (Cross-Encoder) ---")
    # --- SWAPPED TO PUBMEDBERT HERE ---
    pubmed_id = "pritamdeka/PubMedBERT-MNLI-MedNLI"
    pubmed_dir = output_dir / "pubmedbert"
    pubmed_path = pubmed_dir / "model.onnx"

    if pubmed_path.exists():
        print(
            f"PubMedBERT ONNX model already exists at {pubmed_path}. Skipping export."
        )
    else:
        print(f"Downloading and converting {pubmed_id}...")
        # ORTModelForSequenceClassification keeps the classification head
        pubmed_model = ORTModelForSequenceClassification.from_pretrained(
            pubmed_id, export=True
        )
        pubmed_tokenizer = AutoTokenizer.from_pretrained(pubmed_id)

        pubmed_model.save_pretrained(pubmed_dir)
        pubmed_tokenizer.save_pretrained(pubmed_dir)
        print(f"PubMedBERT successfully saved to {pubmed_dir}\n")


if __name__ == "__main__":
    export_to_onnx()
