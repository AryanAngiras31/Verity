# System Architecture: Verity

This is the architectural documentation for **Verity**. This document provides a comprehensive overview of the system's design principles, constraints, data models, and execution flow.

Verity is engineered as a high-performance, hallucination-resistant **Retrieval-Augmented Natural Language Inference (NLI)** pipeline. It is designed to evaluate the logical truth of scientific and medical claims against a vector-embedded hybrid corpus of peer-reviewed scientific literature. 

It leverages Rust, Actix-Web, Tokio thread-offloading, and lock-free object pooling to optimize hardware utilization and concurrent throughput without requiring expensive GPU infrastructure. It uses an advanced aggregation algorithm utilizing max-pooling at the chunk-level and weighted thresholded sum pooling at the document-level. This was found to be the best approach, with the highest F1 score on the internal benchmarks after extensive experimentation with different aggregation strategies. 

## Table of Contents

1. [Functional Requirements](#1-functional-requirements)
2. [Non-Functional Requirements (NFRs)](#2-non-functional-requirements-nfrs)
3. [Core Entities](#3-core-entities)
4. [High Level Design](#4-high-level-design)
    * [4.1 Architecture Diagram](#41-architecture-diagram)
    * [4.2 Core Components](#42-core-components)
    * [4.3 Execution Flow (Online Inference)](#43-execution-flow-online-inference)
5. [Deep Dive](#5-deep-dive)
    * [5.1 Why Rust & Actix-Web for the Backend?](#51-why-rust--actix-web-for-the-backend)
    * [5.2 Concurrency Model: Object Pooling vs. Mutexes](#52-concurrency-model-object-pooling-vs-mutexes)
    * [5.3 NLI Heuristics & Stance Aggregation](#53-nli-heuristics--stance-aggregation)

---


# 1. Functional Requirements

1. **Claim Verification Input:** The system must provide a user-facing interface (React) and a REST API endpoint (`POST /api/verify`) that accepts a user-generated scientific or medical claim as a text string.

2. **Dynamic Configuration:** The API must accept an optional `qdrant_threshold` parameter to allow automated benchmarking scripts or advanced users to dynamically adjust the strictness of the vector search.

3. **Semantic Embedding:** The system must instantly tokenize and embed the incoming claim using a locally hosted Bi-Encoder model (via ONNX runtime) to generate a 768-dimensional dense vector representation.

4. **Context Retrieval (The Bouncer):** The system must query a Qdrant vector database (`verity_hybrid_corpus`) to retrieve the top 9 most semantically similar document abstracts.

5. **Out-of-Domain Rejection:** If no retrieved documents meet the predefined similarity threshold (e.g., `0.55`), the system must immediately abort the pipeline and return a `NEUTRAL` verdict to prevent the "Closed-World Fallacy."

6. **Dynamic Radius Retrieval**: To prevent low similarity documents from lowering the signal from the high similarity documents, dynamic radius retrieval was implemented where only documents whose similarity score is within the radius (eg., `0.05`) of the document with the top similarity score are retrieved.

7. **Text Sanitization & Chunking:** For retrieved documents, the system must split the abstract into individual, grammatically correct sentences (retaining terminal punctuation) to isolate facts and prevent attention dilution.

8. **Cross-Encoder NLI Inference:** The system must feed each sentence chunk alongside the original claim into a cross-encoder model to calculate SoftMax probabilities for three logical stances: `SUPPORT`, `REFUTE`, or `NEUTRAL`.

9. **Document-Level Stance Aggregation:** The system must determine a document's overall stance by taking the maximum `SUPPORT` or `REFUTE` probability across all its chunks, defaulting to `NEUTRAL` if no signal exceeds 50% confidence.

10. **Weighted Thresholded Sum Pooling (Verdict Calculation):** The system must aggregate the valid stances across all retrieved documents, averaging the confidences of strong signals (> 65%) to determine a mathematically sound `TRUE`, `FALSE`, or `NEUTRAL` final verdict.

11. **Evidence Formatting:** The system must return the final verdict, aggregate confidence score, and an array of evidence objects containing the source title, dataset origin, snippet (up to 200 characters), individual stance, and confidence score to the frontend.

# 2. Non-Functional Requirements (NFRs)

1. **High Concurrency & Throughput:**

    1. The backend must support multiple users simultaneously without request queuing or deadlocks.

    2. This is satisfied by utilizing an **Object Pooling architecture** combined with **Tokio Blocking Thread Offloading**, ensuring Actix-Web workers remain fully asynchronous and non-blocked during heavy ONNX matrix multiplications.

2. **Accuracy & Hallucination Resistance:**

    1. The system must provide highly deterministic logic, strictly avoiding Generative LLM hallucinations.

    2. It must resist "Lexical Bias" (matching antonyms blindly) and "Attention Dilution" (getting distracted by complex methodology) by enforcing windowed chunking.

3. **Cost-Efficiency & Hardware Agnosticism:**

    1. The inference engine must not require expensive, dedicated cloud GPUs.

    2. By leveraging Rust's memory safety and the ONNX CPU execution provider, the entire verification pipeline must be capable of running fast on standard, cheap commodity CPU servers.

4. **Deployability & Isolation:**

    1. The system must be fully containerized. The vector database, offline Python ingestion pipeline, and Rust backend must operate in isolated, networked Docker containers managed by `docker-compose`.

    2. The backend must not boot until the ingestion pipeline has successfully completed building the corpus.

5. **Maintainability & Modularity:**

    1. The architecture must decouple the offline heavy-lifting (downloading datasets, formatting JSONL, embedding the corpus, exporting ONNX files) from the online inference engine.

    2. The Rust backend must remain stateless and lightweight, holding only the neural network weights and Qdrant connection pool in memory (`AppState`).

# 3. Core Entities

## 3.1. VerifyRequest

- **Description:** The data structure representing a user's request to verify a specific claim against the scientific corpus.

- **Source:** User input through the React frontend or direct REST API call.

- **Key Attributes:**

    - `claim` (string): The text-based scientific or medical assertion submitted for verification.

    - `qdrant_threshold` (optional float): A dynamic hyperparameter allowing the user to override the default similarity score for the vector search.

## 3.2. AppState

- **Description:** The central shared state of the backend application, responsible for maintaining thread-safe access to persistent connections and neural models.

- **Source:** Initialized at runtime during server startup.

- **Key Attributes:**

    - `qdrant_client` (Qdrant): The persistent connection to the vector database.

    - `bi_encoder_model` The Bi-Encoder model used for embedding claims.

    - `cross_encoder_model` The Cross-Encoder model used for logical inference.

    - `bi_encoder_tokenizer` (Tokenizer): The tokenizer for the Bi-Encoder.

    - `cross_encoder_tokenizer` (Tokenizer): The tokenizer for the Cross-Encoder.

## 3.3. Document (Corpus Entry)

- **Description:** A single scientific evidence record retrieved from the database to validate or disprove a claim.

- **Source:** Consolidated from `scifact`,  `healthver`, `LaySumm` and `BioLaySumm` datasets and stored in Qdrant.

- **Key Attributes:**

    - `doc_id` (string): Unique identifier for the document.

    - `title` (string): The title of the scientific paper or medical article.

    - `abstract` (array/string): The body of the text, often split into sentence-level chunks for granular analysis.

    - `dataset_source` (string): Identifies if the record originated from `scifact` or `healthver`.

## 3.4. Evidence

- **Description:** A processed snippet of a Document that has been evaluated by the cross-encoder for logical entailment.

- **Source:** Derived from Document entities during the inference process.

- **Key Attributes:**

    - `title` (string): Inherited from the Document.

    - `source` (string): Inherited from the Document source.

    - `snippet` (string): A short excerpt of the text (up to 200 characters) shown to the user.

    - `stance` (string): The logical relationship to the claim: `SUPPORT`, `REFUTE`, or `NEUTRAL`.

    - `confidence` (float): The SoftMax probability score of the assigned stance.

## 3.5. VerifyResponse

- **Description:** The final aggregated result of the verification process returned to the user.

- **Source:** Aggregated from multiple Evidence entities using Weighted Thresholded Sum Pooling.

- **Key Attributes:**

    - `final_verdict` (string): The global conclusion: `TRUE`, `FALSE`, or `NEUTRAL`.

    - `aggregate_confidence` (float): The averaged confidence of all strongly supporting or refuting evidence.

    - `evidence` (array): A collection of individual evidence objects supporting the verdict.

# 4. High Level Design

## 4.1 Architecture Diagram

![HLDD](./assets/HLDD.png)

## 4.2. Core Components
### 4.2.1. Frontend Application (React/Vite)

- **Role:** The user-facing interface.

- **Responsibilities:** Captures user claims via a text input, visualizes the real-time processing steps (embedding, querying, inferencing), and renders the final `VerifyResponse` payload (Verdict, Confidence, and Evidence cards) in a readable manner.

### 4.2.2 Offline Ingestion Pipeline (Python)

- **Role:** The dataset, database and model initialization service.

- **Responsibilities:**

    - Executes `create_hybrid_dataset.py` to merge `SciFact`, `HealthVer`, `LaySumm` and `BioLaySumm` datasets to create the hybrid claims and hybrid corpus jsonl files.

    - Executes `embed_and_upsert.py` to batch-embed the corpus using the Bi-Encoder and push the vectors/payloads to Qdrant.

    - Executes `export_models.py` to convert the HuggingFace models into optimized `.onnx` files and save them to a shared Docker volume (`/models`).

- **Lifecycle:** Runs once on startup. The backend depends on this service completing successfully before it boots.

### 4.2.3. Vector Database (Qdrant)

- **Role:** The semantic search engine.

- **Responsibilities:** Stores the 768-dimensional document embeddings and their metadata (payloads). It rapidly computes Cosine Similarity or Dot Product scores against incoming user queries, filtering out documents that fall below the dynamic threshold to prevent out-of-domain hallucinations.

### 4.2.4. API Gateway / Orchestrator (Rust Actix-Web)

- **Role:** The concurrent, multi-threaded API Gateway.

- **Responsibilities:**

    - Exposes the `POST /api/verify` REST endpoint.

    - Maintains the `AppState` containing the Qdrant connection pool and the object pool for the ONNX `Session` models.

    - Handles data sanitization: cleaning raw abstracts and creating windows for the chunks.

    - Performs max pooling at the chunk level and weighted thresholded sum pooling at the document level

	- Uses Tokio's blocking pool (`web::block`) to offload heavy synchronous matrix math (ONNX `.run()`) to background threads, ensuring the Actix workers are never blocked and can continuously accept new HTTP requests

### 4.2.5. Inference Engine (ONNX Runtime)

- **Role:** Local machine learning execution.

- **Components:**

    - **Bi-Encoder (BGE-Small / SPECTER 2):** Converts the user's string claim into a dense semantic vector for database querying.

    - **Cross-Encoder (DeBERTa-v3):** Ingests two inputs simultaneously (Text A: The Chunk, Text B: The Claim) and computes cross-attention to determine if the sentence entails (Supports), contradicts (Refutes), or is unrelated to (Neutral) the claim.

## 4.3. Execution Flow (Online Inference)

1. **Request Initiation:** The React Frontend fires a JSON payload `{ "claim": "..."}` to the Rust Backend.

2. **Claim Tokenization & Embedding:** The Rust Backend tokenizes the claim and runs it through the Bi-Encoder's ONNX session.

3. **Vector Output:** The Bi-Encoder outputs a `[1, 768]` dimensional floating-point tensor.

4. **Semantic Search:** The Rust Backend builds a `QueryPointsBuilder` request using the generated vector and the provided `qdrant_threshold`. It queries the Qdrant container over gRPC.

5. **Context Retrieval:** Qdrant returns up to 9 document hits with a threshold of `0.55` for the similarity score. Dynamic Radius Retrieval is performed with a radius of `0.05` to remove low similarity documents.

6. **Chunking & Cross-Encoding:** For each retrieved document, the Rust Backend extracts the `abstract` payload, splits it by the `.` delimiter, restores the periods, and trims whitespace. To process the chunks, the backend asynchronously acquires a Cross-Encoder from the Object Pool and offloads the ONNX inference to a background blocking thread. This calculates the NLI logits for each (chunk, claim) pair without freezing the main web server. Once the document is processed, the Cross-Encoder is released back to the pool.

7. **Logical Scoring:** The Cross-Encoder returns raw logits for the 3 classes. The Rust Backend calculates the SoftMax probabilities to identify the highest confidence stance (Support, Refute, Neutral) for every chunk. A document's confidence and stance is the confidence and stance of the chunk with the highest confidence (Max Pooling).

8. **Aggregation & Response:** The Rust Backend iterates through the evidence list. It multiplies the confidence by the similarity score while filtering out weak signals (65% confidence threshold) and adds this to the `weighted_support_sum` if the stance of the document is `SUPPORT` and `weighted_refute_sum` if the stance of the document is `REFUTE`. The final stance is chosen by looking at the sums and the aggregated confidence is calculated. These along with the Evidence List is sent to the frontend.

# 5. Deep Dive
The architecture of Verity was shaped by three primary constraints: maximizing concurrent throughput, running efficiently on commodity CPU hardware (no expensive GPUs), and mathematically preventing LLM hallucinations. Below is the rationale behind the core engineering decisions.

### 5.1. Why Rust & Actix-Web for the Backend?
While Python is the standard for ML pipelines, using it for the API gateway would introduce the Global Interpreter Lock (GIL), severely bottlenecking concurrent requests.

1. **Zero-Cost Abstractions & Predictability:** Rust provides memory safety and concurrency without a garbage collector, ensuring latency remains stable and predictable under heavy load.
2. **C++ Interoperability:** The `ort` crate provides zero-copy bindings to the C++ ONNX Runtime.
3. **Asynchronous I/O:** Actix-Web, built on Tokio, allows a single cheap server to handle thousands of concurrent TCP connections while efficiently managing the lifecycle of the heavy ML models.

### 5.2. Concurrency Model: Object Pooling vs. Mutexes
A challenge in this architecture was preventing the synchronous, CPU-bound matrix multiplication of the ONNX models (`Session::run()`) from freezing the asynchronous web server.

1. **The Mutex Bottleneck:** Initially, models were shared across workers using a standard `std::sync::Mutex`. This caused high thread starvation at high loads due to Qdrant connection timeouts. If 10 requests arrived simultaneously, they formed a single-file line, resulting in massive queue times and timeout errors.
2. **The Solution (Object Pooling + Offloading):** I implemented a globally shared Object Pool (`deadpool`). When an Actix worker receives a request, it borrows a model from the pool and immediately offloads the inference to Tokio's background blocking thread pool (`web::block`). The web workers immediately return to listening for HTTP traffic, and the OS scheduler maps the background blocking threads perfectly to the available CPU cores, ensuring 100% hardware utilization.

### 5.3. NLI Heuristics & Stance Aggregation
To achieve deterministic fact-checking without risking hallucinations by Generative LLMs, a mathematical heuristic for evidence evaluation was engineered.

1. **Windowed Chunking:** Transformer cross-encoders (like DeBERTa) suffer from "Attention Dilution" when fed massive blocks of text. By chunking abstracts into sliding 2-sentence windows, we force the model's self-attention heads to focus strictly on the direct relationship between the specific claim and the isolated fact.
2. **Max Pooling at the Chunk Level** Most sentences in a scientific abstract are background information such as methodologies and objectives and will return a `NEUTRAL` stance. If we averaged the chunk scores, the `NEUTRAL` noise would mathematically drown out the one sentence that actually proves or disproves the claim. Max Pooling ensures that a single, high-confidence signal dictates the document's stance.
3. **Weighted Thresholded Sum Pooling for the Verdict** Not all retrieved documents are equally relevant. By multiplying the NLI confidence score by the Qdrant semantic similarity score, we heavily weight evidence that perfectly matches the topic of the claim. The 65% hard threshold acts as a low-pass filter, dropping uncertain cross-encoder predictions before they can influence the final mathematical verdict.
4. **Dynamic Radius Retrieval (DRR)?** Standard "Top-K" retrieval is inherently flawed for fact-checking because it forces the system to consider a fixed number of documents even if only the first one is truly relevant. This often introduces low-similarity noise that can dilute the final verdict. By implementing a dynamic radius of $0.05$, the system identifies the similarity score of the top-ranked document and discards any hits that fall below the threshold of $TopScore - 0.05$. This ensures that the evidence fed into the Cross-Encoder is of uniform semantic quality and prevents irrelevant documents from drowning the strongest signal.

-------
