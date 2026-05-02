import os
import re
import sys
from collections import defaultdict

import jsonlines
from datasets import load_dataset

# Define the expected output files
CORPUS_FILE = "/app/output/hybrid_corpus.jsonl"
CLAIMS_FILE = "/app/output/hybrid_claims.jsonl"
CONSOLIDATED_CLAIMS_FILE = "/app/output/hybrid_claims_consolidated.jsonl"


def normalize_title(title):
    """Creates a standardized title string for deduplication."""
    if not title:
        return ""
    return re.sub(r"[^a-z0-9]", "", title.lower())


def consolidate_hf_datasets():
    # Check if both files already exist
    if (
        os.path.exists(CORPUS_FILE)
        and os.path.exists(CLAIMS_FILE)
        and os.path.exists(CONSOLIDATED_CLAIMS_FILE)
    ):
        print("Hybrid dataset files already exist in the output folder. Exiting...")
        sys.exit(0)

    master_corpus = {}
    master_claims = []
    title_to_doc_id = {}

    try:
        print("Downloading SciFact from Hugging Face...")
        # 'trust_remote_code=True' is often required for custom dataset scripts on HF
        scifact_corpus = load_dataset(
            "allenai/scifact", name="corpus", split="train", trust_remote_code=True
        )
        scifact_claims = load_dataset(
            "allenai/scifact", name="claims", split="train+validation+test", trust_remote_code=True
        )
    except Exception as e:
        print(f"Error downloading SciFact: {e}")
        return

    try:
        print("Downloading HealthVer from Hugging Face...")
        # 'dwadden/healthver_entailment' is pre-formatted with abstracts and verdicts
        healthver = load_dataset(
            "dwadden/healthver_entailment", split="train+validation+test", trust_remote_code=True
        )
    except Exception as e:
        print(f"Error downloading HealthVer: {e}")
        return

    try:
        print("Downloading LaySummary from Hugging Face...")
        lay_summ_corpus = load_dataset(
            "sulovexin/laysummary", split="train+validation+test", trust_remote_code=True
        )   
    except Exception as e:
        print(f"Error downloading LaySummary: {e}")
        return

    try:
        print("Downloading BioLaySumm2025-PLOS from Hugging Face...")
        bio_lay_summ_plos_corpus = load_dataset(
            "BioLaySumm/BioLaySumm2025-PLOS", split="train+validation+test", trust_remote_code=True
        )
    except Exception as e:
        print(f"Error downloading BioLaySumm2025-PLOS: {e}")
        return

    try:
        print("Downloading BioLaySumm2025-eLife from Hugging Face...")
        bio_lay_summ_elife_corpus = load_dataset(
            "BioLaySumm/BioLaySumm2025-eLife", split="train+validation+test", trust_remote_code=True
        )
    except Exception as e:
        print(f"Error downloading BioLaySumm2025-eLife: {e}")
        return

    # --- 1. Process SciFact ---
    print("Processing SciFact...")
    # Extract corpus data
    for doc in scifact_corpus:
        doc_id = str(doc["doc_id"])
        title_norm = normalize_title(doc.get("title", ""))

        master_corpus[doc_id] = {
            "doc_id": doc_id,
            "title": doc.get("title", ""),
            "abstract": " ".join(doc["abstract"]),
            "dataset_source": "scifact",
        }
        if title_norm:
            title_to_doc_id[title_norm] = doc_id

    for claim in scifact_claims:
        # SciFact HF structures evidence directly in the row
        unified_claim = {
            "id": claim["id"],
            "claim": claim["claim"],
            "evidence": {
                str(claim["evidence_doc_id"]): [
                    {
                        "sentences": claim["evidence_sentences"],
                        "label": claim[
                            "evidence_label"
                        ],  # Usually "SUPPORT" or "CONTRADICT"
                    }
                ]
            },
            "dataset_source": "scifact",
        }
        master_claims.append(unified_claim)

    # --- 2. Process HealthVer ---
    print("Processing HealthVer and Deduplicating...")
    for item in healthver:
        # Standardize labels to match SciFact
        raw_label = str(item["verdict"]).upper()
        label_map = {
            "SUPPORTS": "SUPPORT",
            "SUPPORT": "SUPPORT",
            "REFUTES": "CONTRADICT",
            "CONTRADICT": "CONTRADICT",
            "NEUTRAL": "NEUTRAL",
            "NOINFO": "NEUTRAL",
        }
        stance = label_map.get(raw_label, "NEUTRAL")

        raw_doc_id = str(item["abstract_id"])
        title = item.get("title", "")
        title_norm = normalize_title(title)

        # Deduplicate using title hash
        if title_norm and title_norm in title_to_doc_id:
            final_doc_id = title_to_doc_id[title_norm]
        else:
            final_doc_id = "hv_" + raw_doc_id
            if title_norm:
                title_to_doc_id[title_norm] = final_doc_id

            master_corpus[final_doc_id] = {
                "doc_id": final_doc_id,
                "title": title,
                "abstract": " ".join(item["abstract"]),
                "dataset_source": "healthver",
            }

        # HealthVer provides rationale sentences as raw strings.
        # We must find their integer index within the abstract to match SciFact's format.
        evidence_indices = []
        for ev_sentence in item["evidence"]:
            try:
                idx = item["abstract"].index(ev_sentence)
                evidence_indices.append(idx)
            except ValueError:
                continue  # Sentence not found exactly, skip

        unified_claim = {
            "id": item["claim_id"],
            "claim": item["claim"],
            "evidence": {
                final_doc_id: [{"sentences": evidence_indices, "label": stance}]
            },
            "dataset_source": "healthver",
        }
        master_claims.append(unified_claim)

    # --- 3. Process BioLaySumm ---
    print("Processing BioLaySumm...")
    bio_plos_counter = 0
    for doc in bio_lay_summ_plos_corpus:
        id = f"bio_plos_{bio_plos_counter}"
        master_corpus[str(id)] = {
            "doc_id": str(id),
            "title": doc.get("title", ""),
            "abstract": doc["summary"],
            "dataset_source": "bio_lay_summ_plos",
        }
        bio_plos_counter += 1

    bio_elife_counter = 0
    for doc in bio_lay_summ_elife_corpus:
        id = f"bio_elife_{bio_elife_counter}"
        master_corpus[str(id)] = {
            "doc_id": str(id),
            "title": doc.get("title", ""),
            "abstract": doc["summary"],
            "dataset_source": "bio_lay_summ_elife",
        }
        bio_elife_counter += 1

    # --- 4. Process LaySumm ---
    print("Processing LaySumm...")
    for doc in lay_summ_corpus:
        id = str(doc["id"])
        master_corpus[id] = {
            "doc_id": id,
            "title": doc.get("title", ""),
            "abstract": doc["summary"],
            "dataset_source": "lay_summ",
        }

    # Create consolidated claims
    print("Consolidating claims for benchmark (one label per claim)...")
    grouped_claims = defaultdict(list)

    # Group every evidence label associated with each unique claim string
    for item in master_claims:
        claim_text = item["claim"]
        # Extract the label from the evidence dict
        for doc_id, evidence_list in item["evidence"].items():
            for ev in evidence_list:
                label = ev.get("label", "NEUTRAL")
                if label:
                    grouped_claims[claim_text].append(label)

    consolidated_list = []
    for claim_text, labels in grouped_claims.items():
        # Numerical mapping for "Mean" calculation
        score = 0
        for label in labels:
            if label == "SUPPORT":
                score += 1
            elif label == "CONTRADICT":
                score -= 1
            # NEUTRAL contributes 0

        final_label = "NEUTRAL"
        if score > 0:
            final_label = "SUPPORT"
        elif score < 0:
            final_label = "CONTRADICT"

        consolidated_list.append(
            {"claim": claim_text, "label": final_label, "evidence_count": len(labels)}
        )

    print(f"Total Unique Documents: {len(master_corpus)}")
    print(f"Total Claims: {len(master_claims)}")
    print(f"Total Consolidated Claims for Benchmark: {len(consolidated_list)}")

    # --- 5. Export to JSONL ---
    print(
        "Writing to hybrid_corpus.jsonl and hybrid_claims.jsonl to the mounted volume..."
    )
    with jsonlines.open(CORPUS_FILE, mode="w") as writer:
        for doc in master_corpus.values():
            writer.write(doc)

    with jsonlines.open(CLAIMS_FILE, mode="w") as writer:
        for claim in master_claims:
            writer.write(claim)

    with jsonlines.open(CONSOLIDATED_CLAIMS_FILE, mode="w") as writer:
        for entry in consolidated_list:
            writer.write(entry)

    print("Consolidation Complete!")


if __name__ == "__main__":
    consolidate_hf_datasets()
