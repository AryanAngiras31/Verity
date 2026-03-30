import os
import sys
import uuid

import jsonlines
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6334))
MODEL_NAME = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME = "verity_hybrid_corpus"


# Connects to the local Qdrant instance via gRPC and create collection.
def connect_to_qdrant():
    try:
        # 1. Connect to Qdrant via gRPC
        print("Connecting to local Qdrant instance via gRPC...")
        qdrant = QdrantClient(host=QDRANT_HOST, grpc_port=QDRANT_PORT, prefer_grpc=True)

        # 2. Create collection if it doesn't exist
        collection_name = COLLECTION_NAME
        if not qdrant.collection_exists(collection_name):
            qdrant.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
            print(f"Collection '{collection_name}' created.")
            return qdrant, False
        else:
            # Return None as qdrant client and True since collection exists
            return None, True

    except Exception as e:
        print(f"Failed to connect to Qdrant: {e}")
        return None, False


# Loads the SPECTER 2 model.
def get_model():
    try:
        print("Loading SPECTER 2 model...")
        # We use the base model which is highly optimized for semantic search
        model = SentenceTransformer(MODEL_NAME)
        return model
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None


# Embeds and upserts documents into the Qdrant collection.
def embed_and_upsert(qdrant, model, filename, batchsize=500):
    points = []
    total_processed = 0

    print(f"Reading documents from file {filename}...")
    with jsonlines.open(filename) as reader:
        for doc in reader:
            # 1. Join the abstract sentences into single string
            joined_abstract = " ".join(doc["abstract"])

            # 2. Format the joined abstract for embedding
            # Format for SPECTER 2: Title + [SEP] + Abstract
            formatted_abstract = (
                doc["title"] + model.tokenizer.sep_token + joined_abstract
            )

            # 3. Embed the formtted abstract
            vector = model.encode(formatted_abstract).tolist()

            # 4. Generate a deterministic UUID based on the document id
            point_id = uuid.uuid5(uuid.NAMESPACE_DNS, str(doc["doc_id"]))

            # 5. Build the Qdrant Point payload
            point = PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "doc_id": str(doc["doc_id"]),
                    "title": doc["title"],
                    "abstract": joined_abstract,
                    "dataset_source": doc.get("dataset_source", "unknown"),
                },
            )
            points.append(point)

            # 6. Upsert to qdrant in batches
            if len(points) >= batchsize:
                qdrant.upsert(
                    collection_name=COLLECTION_NAME,
                    points=points,
                )
                total_processed += len(points)
                print(
                    f"Progress: Upserted {total_processed} documents so far...",
                    flush=True,
                )
                points = []

    # Upsert any remaining points
    if points:
        qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
        )
        total_processed += len(points)
    print(f"Upserted {total_processed} documents in total.")


if __name__ == "__main__":
    model = get_model()
    qdrant, is_populated = connect_to_qdrant()
    if is_populated:
        print(f"Collection '{COLLECTION_NAME}' already exists. Exiting...")
        sys.exit(0)
    elif qdrant and model:
        embed_and_upsert(qdrant, model, "/app/output/hybrid_corpus.jsonl", 500)
    else:
        sys.exit(1)
