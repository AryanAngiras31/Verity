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

    print("\n--- 2. Exporting DeBERTa-v3 (Cross-Encoder) ---")
    deberta_id = "cross-encoder/nli-deberta-v3-small"
    deberta_dir = output_dir / "deberta"
    deberta_path = deberta_dir / "model.onnx"  # Standard ONNX model filename

    if deberta_path.exists():
        print(
            f"DeBERTa-v3 ONNX model already exists at {deberta_path}. Skipping export."
        )
    else:
        # ORTModelForSequenceClassification keeps the classification head,
        # giving us the Support, Refute, and Neutral logits we need for our verdict.
        print(f"Downloading and converting {deberta_id}...")
        deberta_model = ORTModelForSequenceClassification.from_pretrained(
            deberta_id, export=True
        )
        deberta_tokenizer = AutoTokenizer.from_pretrained(deberta_id)

        deberta_model.save_pretrained(deberta_dir)
        deberta_tokenizer.save_pretrained(deberta_dir)
        print(f"DeBERTa-v3 successfully saved to {deberta_dir}\n")


if __name__ == "__main__":
    export_to_onnx()
