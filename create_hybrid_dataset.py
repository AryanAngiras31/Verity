import json
import os
import uuid


def load_scifact_corpus(filepath):
    """
    Loads and normalizes SciFact corpus data from a .jsonl file.
    Assumes each line is a JSON object representing a document.
    """
    scifact_documents = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line)
                # If 'abstract' field is missing, attempt to concatenate from 'sections'
                abstract_text = doc.get("abstract")
                if not abstract_text and "sections" in doc:
                    abstract_text = " ".join(
                        [s["text"] for s in doc["sections"] if "text" in s]
                    )

                normalized_doc = {
                    "original_doc_id": doc.get(
                        "doc_id"
                    ),  # Keep original for fallback key
                    "title": doc.get("title"),
                    "abstract": abstract_text,
                    "year": doc.get("year"),
                    # Construct URL from DOI if available
                    "url": f"https://doi.org/{doc['doi']}" if doc.get("doi") else None,
                    "pmid": str(doc["pmid"])
                    if doc.get("pmid")
                    else None,  # Ensure PMID is string
                    "doi": doc.get("doi"),
                    "dataset_source": "scifact",
                }
                scifact_documents.append(normalized_doc)
    except FileNotFoundError:
        print(f"Error: SciFact corpus file not found at {filepath}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from SciFact corpus file {filepath}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while loading SciFact corpus: {e}")
    return scifact_documents


def load_healthver_corpus(filepath):
    """
    Loads and normalizes HealthVer corpus data from a JSON file.
    Assumes the file contains a JSON array of document objects.
    """
    healthver_documents = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Assuming `data` is a list of documents. Adjust if it's nested (e.g., `data['documents']`).
            for doc in data:
                normalized_doc = {
                    "original_doc_id": doc.get(
                        "doc_id"
                    ),  # Keep original for fallback key
                    "title": doc.get("title"),
                    "abstract": doc.get("abstract"),
                    "year": doc.get(
                        "publish_year"
                    ),  # Assuming 'publish_year' for HealthVer
                    "url": doc.get("url"),
                    "pmid": str(doc["pmid"])
                    if doc.get("pmid")
                    else None,  # Ensure PMID is string
                    "doi": doc.get("doi"),
                    "dataset_source": "healthver",
                }
                healthver_documents.append(normalized_doc)
    except FileNotFoundError:
        print(f"Error: HealthVer corpus file not found at {filepath}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from HealthVer corpus file {filepath}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while loading HealthVer corpus: {e}")
    return healthver_documents


def consolidate_and_deduplicate(scifact_docs, healthver_docs):
    """
    Consolidates and deduplicates documents from both datasets based on DOI or PMID.
    Generates a new UUID for 'doc_id' for each unique document.
    """
    unified_corpus = []
    # Stores (identifier_type, identifier_value) tuples for deduplication
    unique_identifiers = set()

    def get_document_identifiers(doc):
        """Extracts primary identifiers (DOI, PMID) for a document."""
        identifiers = []
        if doc.get("doi"):
            identifiers.append(("doi", doc["doi"].lower()))
        if doc.get("pmid"):
            identifiers.append(("pmid", doc["pmid"].lower()))
        # Fallback for documents without DOI/PMID to ensure unique identification
        # within their source dataset if no other unique key is present.
        if not identifiers and doc.get("original_doc_id"):
            identifiers.append(
                ("fallback", f"{doc['dataset_source']}_{doc['original_doc_id']}")
            )
        return identifiers

    # Process SciFact documents
    for doc in scifact_docs:
        doc_identifiers = get_document_identifiers(doc)
        is_duplicate = False
        for id_type, id_value in doc_identifiers:
            if (id_type, id_value) in unique_identifiers:
                is_duplicate = True
                break

        if not is_duplicate:
            # Add all identifiers for this unique document to the set
            for id_type, id_value in doc_identifiers:
                unique_identifiers.add((id_type, id_value))

            # Create the final Qdrant payload structure
            qdrant_payload = {
                "doc_id": str(uuid.uuid4()),  # Generate a new UUID as required
                "title": doc.get("title"),
                "abstract": doc.get("abstract"),
                "year": doc.get("year"),
                "url": doc.get("url"),
                "dataset_source": doc.get("dataset_source"),
            }
            unified_corpus.append(qdrant_payload)
        else:
            print(
                f"Skipping duplicate SciFact document: {doc.get('doi') or doc.get('pmid') or doc.get('original_doc_id')}"
            )

    # Process HealthVer documents, checking for duplicates against the existing corpus
    for doc in healthver_docs:
        doc_identifiers = get_document_identifiers(doc)
        is_duplicate = False
        for id_type, id_value in doc_identifiers:
            if (id_type, id_value) in unique_identifiers:
                is_duplicate = True
                break

        if not is_duplicate:
            # Add all identifiers for this unique document to the set
            for id_type, id_value in doc_identifiers:
                unique_identifiers.add((id_type, id_value))

            qdrant_payload = {
                "doc_id": str(uuid.uuid4()),  # Generate a new UUID as required
                "title": doc.get("title"),
                "abstract": doc.get("abstract"),
                "year": doc.get("year"),
                "url": doc.get("url"),
                "dataset_source": doc.get("dataset_source"),
            }
            unified_corpus.append(qdrant_payload)
        else:
            print(
                f"Skipping duplicate HealthVer document: {doc.get('doi') or doc.get('pmid') or doc.get('original_doc_id')}"
            )

    return unified_corpus


if __name__ == "__main__":
    # Define directories relative to the project root
    project_root = "Verity"
    data_dir = os.path.join(project_root, "data")
    data_pipeline_dir = os.path.join(project_root, "data_pipeline")

    # Ensure output directory exists
    os.makedirs(data_pipeline_dir, exist_ok=True)

    # --- USER MUST UPDATE THESE FILE PATHS ---
    # These are placeholder paths. Make sure to download your datasets and
    # place them, then update these variables to their correct locations.
    # Example: If your SciFact corpus is in Verity/data/scifact_corpus.jsonl
    scifact_corpus_path = os.path.join(data_dir, "scifact_corpus.jsonl")
    healthver_corpus_path = os.path.join(data_dir, "healthver_corpus.json")
    # ----------------------------------------

    output_consolidated_path = os.path.join(
        data_pipeline_dir, "consolidated_corpus.json"
    )

    print(f"Attempting to load SciFact corpus from: {scifact_corpus_path}")
    scifact_docs = load_scifact_corpus(scifact_corpus_path)
    print(f"Loaded {len(scifact_docs)} SciFact documents (before consolidation).")

    print(f"Attempting to load HealthVer corpus from: {healthver_corpus_path}")
    healthver_docs = load_healthver_corpus(healthver_corpus_path)
    print(f"Loaded {len(healthver_docs)} HealthVer documents (before consolidation).")

    if scifact_docs or healthver_docs:
        print("Consolidating and deduplicating documents...")
        consolidated_data = consolidate_and_deduplicate(scifact_docs, healthver_docs)
        print(f"Consolidated corpus size: {len(consolidated_data)} unique documents.")

        # Save the consolidated data
        try:
            with open(output_consolidated_path, "w", encoding="utf-8") as f:
                json.dump(consolidated_data, f, indent=2)
            print(
                f"Consolidated data successfully saved to: {output_consolidated_path}"
            )
        except IOError as e:
            print(f"Error saving consolidated data to {output_consolidated_path}: {e}")
    else:
        print(
            "No documents loaded from either dataset. Skipping consolidation and saving."
        )
