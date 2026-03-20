from pathlib import Path

from optimum.onnxruntime import (
    ORTModelForFeatureExtraction,
    ORTModelForSequenceClassification,
)
from transformers import AutoTokenizer


def export_to_onnx():
    # Define the output directory
    output_dir = Path("./models")
    output_dir.mkdir(exist_ok=True)

    print("\n--- 1. Exporting SPECTER 2 (Bi-Encoder) ---")
    specter_id = "allenai/specter2_base"
    specter_dir = output_dir / "specter2"
    specter_path = specter_dir / "model.onnx"  # Standard ONNX model filename

    if specter_path.exists():
        print(
            f"SPECTER 2 ONNX model already exists at {specter_path}. Skipping export."
        )
    else:
        # ORTModelForFeatureExtraction strips the final classification head,
        # leaving us with the raw 768-dimensional vector output we need for Qdrant.
        print(f"Downloading and converting {specter_id}...")
        specter_model = ORTModelForFeatureExtraction.from_pretrained(
            specter_id, export=True
        )
        specter_tokenizer = AutoTokenizer.from_pretrained(specter_id)

        specter_model.save_pretrained(specter_dir)
        specter_tokenizer.save_pretrained(specter_dir)
        print(f"SPECTER 2 successfully saved to {specter_dir}\n")

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
