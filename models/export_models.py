from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from pathlib import Path
import tempfile
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

    print("\n--- 2. Exporting Quantized PubMedBERT (Cross-Encoder) ---")
    pubmed_id = "pritamdeka/PubMedBERT-MNLI-MedNLI"
    pubmed_dir = output_dir / "pubmedbert"
    pubmed_path = pubmed_dir / "model_quantized.onnx"

    if pubmed_path.exists():
        print(
            f"Quantized PubMedBERT ONNX model already exists at {pubmed_path}. Skipping export."
        )
    else:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            print(f"Downloading and exporting f32 model to temporary path...")
            # Export the initial f32 model to the temp directory
            pubmed_model = ORTModelForSequenceClassification.from_pretrained(
                pubmed_id, export=True
            )
            pubmed_model.save_pretrained(temp_path)

            print("Applying INT8 Dynamic Quantization...")
            quantizer = ORTQuantizer.from_pretrained(temp_path)

            # Create the quantization configuration optimized for CPUs
            dqconfig = AutoQuantizationConfig.avx2(
                is_static=False, 
                per_channel=True
            )
    
            quantizer.quantize(
                save_dir=pubmed_dir, 
                quantization_config=dqconfig
            )
    
            pubmed_tokenizer = AutoTokenizer.from_pretrained(pubmed_id)
            pubmed_tokenizer.save_pretrained(pubmed_dir)
            print(f"Quantized PubMedBERT successfully saved to {pubmed_dir}\n")


if __name__ == "__main__":
    export_to_onnx()
