import tempfile
from pathlib import Path

from adapters import AutoAdapterModel
from optimum.onnxruntime import (
    ORTModelForFeatureExtraction,
    ORTModelForSequenceClassification,
)
from transformers import AutoTokenizer


def export_to_onnx():
    # Define the output directory
    output_dir = Path("/app/models/")
    output_dir.mkdir(exist_ok=True)

    print("\n--- 1. Exporting SPECTER 2 (Bi-Encoder) ---")
    specter_id = "allenai/specter2_base"
    adapter_id = "allenai/specter2_adhoc_query"
    specter_dir = output_dir / "specter2"
    specter_path = specter_dir / "model.onnx"  # Standard ONNX model filename

    if specter_path.exists():
        print(
            f"SPECTER 2 ONNX model already exists at {specter_path}. Skipping export."
        )
    else:
        print(f"Downloading base model {specter_id}...")
        # 1. Load the base model using the adapters library
        model = AutoAdapterModel.from_pretrained(specter_id)
        tokenizer = AutoTokenizer.from_pretrained(specter_id)

        # 2. Load and activate the ad-hoc query adapter
        print(f"Loading adapter {adapter_id}...")
        model.load_adapter(adapter_id, set_active=True)

        # 3. FUSE the adapter weights into the base model permanently
        print("Merging adapter weights into base model...")
        model.merge_adapter(adapter_id)

        # 4. Save the merged PyTorch model temporarily so Optimum can export it
        print("Exporting fused model to ONNX...")
        with tempfile.TemporaryDirectory() as temp_dir:
            model.save_pretrained(temp_dir)
            tokenizer.save_pretrained(temp_dir)

            # 5. Load the merged model using Optimum for ONNX export
            ort_model = ORTModelForFeatureExtraction.from_pretrained(
                temp_dir, export=True
            )
            ort_model.save_pretrained(specter_dir)
            tokenizer.save_pretrained(specter_dir)

        print(f"SPECTER 2 (Ad-Hoc Query) successfully saved to {specter_dir}\n")

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
