mod types;
use types::{VerifyRequest, VerifyResponse, Evidence};

use std::sync::Mutex;
use std::env;

use actix_web::{post, web, App, HttpServer, Responder, HttpResponse};

use qdrant_client::Qdrant;
use qdrant_client::qdrant::QueryPointsBuilder;

use ort::session::Session;
use ort::value::Tensor;
use tokenizers::Tokenizer;

// This struct holds our shared application state
struct AppState {
    qdrant_client: Qdrant,
    bi_encoder_model: Mutex<Session>,
    bi_encoder_tokenizer: Tokenizer,
    cross_encoder_model: Mutex<Session>,
    cross_encoder_tokenizer: Tokenizer,
}

const COLLECTION_NAME: &str = "verity_hybrid_corpus";

#[post("/api/verify")]
async fn verify_claim(
    req_body: web::Json<VerifyRequest>,
    data: web::Data<AppState>,
) -> impl Responder {
    let claim: &str = &req_body.claim;

    println!("============================================================================================================");
    println!("Received claim: {}", claim);
    println!("============================================================================================================\n");

    // --- NEW FIX: BGE-Small strictly requires this exact prefix for search queries! ---
    let bi_encoder_query = format!("Represent this sentence for searching relevant passages: {}", claim);
    // Embed the claim using the BGE Small model
    let encoding = data.bi_encoder_tokenizer.encode(bi_encoder_query, true).expect("Failed to encode claim using the Bi-Encoder tokenizer");

    // BGE Small is build on a BERT style architecture and requires these three inputs
    // ONNX expects 64 bit integers for input ids and masks

    // Ids of the tokens in the sequence
    let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
    // Tells the model which tokens are padding and which are actual tokens. (1 for actual tokens, 0 for padding)
    let attention_mask: Vec<i64> = encoding.get_attention_mask().iter().map(|&x| x as i64).collect();
    // Tells the model which tokens belong to which sequence. (0 for the first sequence, 1 for the second, etc.)
    let token_type_ids: Vec<i64> = encoding.get_type_ids().iter().map(|&x| x as i64).collect();

    let batch_size: usize = 1;  // This is the number of sequences
    let seq_len: usize = input_ids.len();   // This is the number of tokens in the sequence
    let shape = vec![batch_size, seq_len];  // Pass a standard Rust Tuple (Shape, Data) directly to ONNX.

    // Use a scoping block to automatically drop the pointer from output to the Bi-Encoder and the Mutex lock after the embedding is extracted
    let embedding: Vec<f32> = {
        // Get the Mutex lock to get safe, mutable access to the model
        let mut bi_encoder = data.bi_encoder_model.lock().expect("Could not get Mutex lock for the Bi-Encoder model");

        // Run the BGE small model
        let outputs = bi_encoder.run(ort::inputs![
            "input_ids" => Tensor::from_array((shape.clone(), input_ids)).unwrap(),
            "attention_mask" => Tensor::from_array((shape.clone(), attention_mask)).unwrap(),
            "token_type_ids" => Tensor::from_array((shape.clone(), token_type_ids)).unwrap(),
        ]).expect("Failed to run the Bi-Encoder model");

        // Extract the embedding
        // BGE-Small uses the [CLS] token (the very first token at index 0) as the embedding for the whole sentence.
        let (_shape, tensor_data) = outputs["last_hidden_state"]    // The output is a tensor of shape [batch_size, sequence_length, hidden_dimension]
            .try_extract_tensor::<f32>()
            .expect("Failed to extract float tensor from ONNX output");

        // Because the tensor is flattened, the first token's 384 dimensions are simply the first 384 numbers in the array. Return them.
        tensor_data[0..384].to_vec()
    };

    // Extract the dynamic threshold from the request (default to 0.60)
    let threshold: f32 = req_body.qdrant_threshold.unwrap_or(0.60);
    // Query Qdrant for top 5 matches
    let query_request = QueryPointsBuilder::new(COLLECTION_NAME)
        .query(embedding)
        .limit(9)
        .score_threshold(threshold)
        .with_payload(true);

    let response_result = data.qdrant_client
        .query(query_request).await;

    let response = match response_result {
        Ok(response) => {
            response
        }
        Err(e) => {
            println!("Error querying Qdrant: {:?}", e);
            return HttpResponse::InternalServerError().body("Error querying Qdrant");
        }
    };

    // Perform dynamic radius retrieval
    let radius: f32 = 0.05;
    let top_result = response.result.first().map(|hit| hit.score).unwrap_or(0.0);
    let response = response.result.into_iter().filter(|hit| hit.score >= top_result - radius).collect::<Vec<_>>();

    // Cross Encoder Inference
    let mut evidence_list: Vec<Evidence> = Vec::new();

    // Get the Mutex lock for the Cross Encoder model
    let mut cross_encoder = data.cross_encoder_model.lock().expect("Could not get Mutex lock for the Cross Encoder model");

    // TRACKERS FOR DOCUMENT-LEVEL MAX POOLING
    let mut final_verdict = "NEUTRAL".to_string();
    let mut highest_weighted_confidence: f32 = 0.0;
    let mut raw_confidence_for_verdict: f32 = 0.0;

    for hit in response.iter() {
        let payload = &hit.payload;

        let get_string = |key: &str| {
            payload.get(key).and_then(|v| v.as_str()).map(|s| s.as_str()).unwrap_or("Unknown")
        };

        let title = get_string("title");
        let source = get_string("dataset_source");
        let abstract_text = get_string("abstract");

        // WE WILL USE THIS NOW
        let _score = hit.score;

        println!("----------------------------------------------------------------------------------------------------");
        println!("Score: {:.4} | Title: {} [{}]", _score, title, source);
        println!("----------------------------------------------------------------------------------------------------\n");

        // CHUNK-LEVEL MAX POOLING
        let mut best_support: f32 = 0.0;
        let mut best_refute: f32 = 0.0;
        let mut max_signal: f32 = 0.0;

        let sentences: Vec<String> = abstract_text
            .split(". ")
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .map(|s| format!("{}.", s))
            .collect();

        let window_size = 2;
        let mut chunks: Vec<String> = Vec::new();

        if sentences.len() <= window_size {
            chunks.push(sentences.join(" "));
        } else {
            for window in sentences.windows(window_size) {
                chunks.push(window.join(" "));
            }
        }

        for clean_chunk in chunks {
            let encoding = data.cross_encoder_tokenizer
                .encode((clean_chunk.as_str(), claim), true)
                .expect("Failed to tokenize chunk and claim");

            let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
            let attention_mask: Vec<i64> = encoding.get_attention_mask().iter().map(|&x| x as i64).collect();
            let token_type_ids: Vec<i64> = encoding.get_type_ids().iter().map(|&x| x as i64).collect();

            let shape = vec![1, input_ids.len() as i64];

            let cross_encoder_result = cross_encoder.run(ort::inputs![
                "input_ids" => Tensor::from_array((shape.clone(), input_ids)).unwrap(),
                "attention_mask" => Tensor::from_array((shape.clone(), attention_mask)).unwrap(),
                "token_type_ids" => Tensor::from_array((shape.clone(), token_type_ids)).unwrap(),
            ]);

            let outputs = match cross_encoder_result {
                Ok(outputs) => outputs,
                Err(e) => {
                    println!("Warning: Skipping chunk. ONNX inference failed (likely exceeded 512 tokens). Length: {}, Error: {:?}", seq_len, e);
                    continue;
                }
            };

            let (_shape, logits_data) = match outputs["logits"].try_extract_tensor::<f32>() {
                Ok((shape, logits_data)) => (shape, logits_data),
                Err(e) => {
                    println!("Warning: Failed to extract tensor, Error: {:?}", e);
                    continue;
                }
            };
            let logits = &logits_data[0..3];

            let max_logit: f32 = logits[0].max(logits[1].max(logits[2]));
            let exp_logits: Vec<f32> = logits.iter().map(|&x| (x-max_logit).exp()).collect();
            let sum_logits: f32 = exp_logits.iter().sum();
            let softmax_probs: Vec<f32> = exp_logits.iter().map(|&x| x / sum_logits).collect();

            let refute_prob = softmax_probs[0];
            let support_prob = softmax_probs[1];

            let chunk_signal = refute_prob.max(support_prob);

            // Keep only the highest signal chunk
            if chunk_signal > max_signal {
                max_signal = chunk_signal;
                best_support = support_prob;
                best_refute = refute_prob;
            }
        }

        let stance;
        let confidence;

        if best_support > best_refute && best_support > 0.50 {
            stance = "SUPPORT".to_string();
            confidence = best_support;
        } else if best_refute > best_support && best_refute > 0.50 {
            stance = "REFUTE".to_string();
            confidence = best_refute;
        } else {
            stance = "NEUTRAL".to_string();
            confidence = 1.0 - best_support - best_refute;
        }

        // DOCUMENT-LEVEL WEIGHTED MAX POOLING
        let weighted_confidence = confidence * _score;

        // Apply a strict 0.80 logic threshold so weak hallucinations don't win
        if stance != "NEUTRAL" && confidence > 0.80 {
            if weighted_confidence > highest_weighted_confidence {
                highest_weighted_confidence = weighted_confidence;
                final_verdict = stance.clone();
                raw_confidence_for_verdict = confidence; // Keep for the JSON response
            }
        }

        evidence_list.push(types::Evidence {
            title: title.to_string(),
            source: source.to_string(),
            snippet: format!("{}...", &abstract_text[..200.min(abstract_text.len())]),
            stance,
            confidence,
        });
    }

    drop(cross_encoder);

    // Map output to match the True/False expectation for your benchmark script
    if final_verdict == "SUPPORT" {
        final_verdict = "TRUE".to_string();
    } else if final_verdict == "REFUTE" {
        final_verdict = "FALSE".to_string();
    }

    // If all documents were filtered out or neutral, fallback to the highest neutral score
    let aggregate_confidence = if final_verdict != "NEUTRAL" {
        raw_confidence_for_verdict
    } else {
        evidence_list.iter().map(|e| e.confidence).fold(0.0, f32::max)
    };

    let response = VerifyResponse {
        final_verdict,
        aggregate_confidence,
        evidence: evidence_list,
    };

    HttpResponse::Ok().json(response)
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // 1. Initialize ONNX runtime engine
    ort::init().with_name("verity_inference_engine").commit();

    // 2. Initialize Qdrant Client (Connecting over the Docker network)
    let qdrant_url = env::var("QDRANT_URL").unwrap_or_else(|_| "http://qdrant:6334".to_string());

    println!("\nBackend connecting to Qdrant at {}...", qdrant_url);
    let client = Qdrant::from_url(&qdrant_url)
        .build();

    let client = match client {
        Ok(client) => {
            println!("Backend connected to Qdrant successfully.");
            client
        }
        Err(e) => {
            println!("Failed to connect to Qdrant: {:?}", e);
            return Err(std::io::Error::new(std::io::ErrorKind::Other, "Backend failed to connect to Qdrant"));
        }
    };

    println!("Started Verity Rust API on 0.0.0.0:8080...");

    // 5. Start the HTTP Server
    HttpServer::new(move || {
        // 4. Wrap client in Actix web::Data for thread-safe sharing
        let app_state = web::Data::new(AppState {
            qdrant_client: client.clone(),
            bi_encoder_model: Mutex::new(Session::builder().unwrap().commit_from_file("models/bge_small/model.onnx").expect("Failed to load BGE-Small model")),
            bi_encoder_tokenizer: Tokenizer::from_file("models/bge_small/tokenizer.json").expect("Failed to load BGE-Small tokenizer"),
            cross_encoder_model: Mutex::new(Session::builder().unwrap().commit_from_file("models/pubmedbert/model.onnx").expect("Failed to load the PubMedBERT model")),
            cross_encoder_tokenizer: Tokenizer::from_file("models/pubmedbert/tokenizer.json").expect("Failed to load the PubMedBERT tokenizer"),
        });

        App::new()
            .app_data(app_state.clone())
            .service(verify_claim)
    })
    .workers(2)     // Limit to 2 workers for development
    .bind(("0.0.0.0", 8080))?
    .run()
    .await
}
