mod types;
use types::{VerifyRequest, VerifyResponse, Evidence};

use std::env;

use actix_web::{post, web, App, HttpServer, Responder, HttpResponse};

use qdrant_client::Qdrant;
use qdrant_client::qdrant::QueryPointsBuilder;

use ort::session::Session;
use ort::value::Tensor;
use tokenizers::Tokenizer;

use deadpool::managed::{Manager, RecycleResult};


// Bi-Encoder pool manager
struct BiEncoderManager;

impl Manager for BiEncoderManager {
    type Type = Session;
    type Error = ort::Error;

    async fn create(&self) -> Result<Self::Type, Self::Error> {
        Session::builder()?
            .commit_from_file("models/bge_small/model.onnx")
    }

    async fn recycle(&self, _obj: &mut Self::Type, _: &deadpool::managed::Metrics) -> RecycleResult<Self::Error> {
        Ok(())
    }
}

// Cross-Encoder pool manager
struct CrossEncoderManager;

impl Manager for CrossEncoderManager {
    type Type = Session;
    type Error = ort::Error;

    async fn create(&self) -> Result<Self::Type, Self::Error> {
        Session::builder()?
            .commit_from_file("models/pubmedbert/model.onnx")
    }

    async fn recycle(&self, _obj: &mut Self::Type, _: &deadpool::managed::Metrics) -> RecycleResult<Self::Error> {
        Ok(())
    }
}

// Define the Pool types for convenience
type BiEncoderPool = deadpool::managed::Pool<BiEncoderManager>;
type CrossEncoderPool = deadpool::managed::Pool<CrossEncoderManager>;

// This struct holds our shared application state
struct AppState {
    qdrant_client: Qdrant,
    bi_encoder_pool: BiEncoderPool,
    bi_encoder_tokenizer: Tokenizer,
    cross_encoder_pool: CrossEncoderPool,
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

    // BGE-Small strictly requires this exact prefix for search queries
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

    // Get the Bi-Encoder from the pool asynchronously
    let mut bi_encoder = data.bi_encoder_pool.get().await.expect("Failed to get Bi-Encoder from pool");

    // Run the Bi-Encoder inference in a blocking thread
    let embedding: Vec<f32> = web::block(move || {
        let outputs = bi_encoder.run(ort::inputs![
            "input_ids" => Tensor::from_array((shape.clone(), input_ids)).unwrap(),
            "attention_mask" => Tensor::from_array((shape.clone(), attention_mask)).unwrap(),
            "token_type_ids" => Tensor::from_array((shape.clone(), token_type_ids)).unwrap(),
        ]).expect("Failed to run the Bi-Encoder model");

        let (_shape, tensor_data) = outputs["last_hidden_state"]
            .try_extract_tensor::<f32>()
            .expect("Failed to extract float tensor from ONNX output");

        tensor_data[0..384].to_vec()
    })
    .await
    .expect("Blocking task for Bi-Encoder failed");

    // Extract the dynamic threshold from the request (default to 0.55)
    let threshold: f32 = req_body.qdrant_threshold.unwrap_or(0.55);
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

    
    // Trackers for document-level thresholded weighted sum pooling
    let mut weighted_support_sum: f32 = 0.0;
    let mut weighted_refute_sum: f32 = 0.0;
    let mut highest_neutral_score: f32 = 0.0;
    
    let mut evidence_list: Vec<Evidence> = Vec::new();

    for hit in response.iter() {
        let payload = &hit.payload;

        // Helper to get fields from Qdrant response
        let get_string = |key: &str| {
            payload.get(key).and_then(|v| v.as_str()).map(|s| s.as_str()).unwrap_or("Unknown")
        };

        let title = get_string("title");
        let source = get_string("dataset_source");
        let abstract_text = get_string("abstract");

        let _score = hit.score;

        println!("----------------------------------------------------------------------------------------------------");
        println!("Score: {:.4} | Title: {} [{}]", _score, title, source);
        println!("----------------------------------------------------------------------------------------------------\n");

        // Variables for chunk-level max pooling
        let mut best_support: f32 = 0.0;
        let mut best_refute: f32 = 0.0;
        let mut max_signal: f32 = 0.0;

        // Clean and split the abstract text into sentences
        let sentences: Vec<String> = abstract_text
            .split(". ")
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .map(|s| format!("{}.", s))
            .collect();

        let window_size = 2;
        let mut chunks: Vec<String> = Vec::new();

        // If there are less than two sentences, combine them. If there are more, perform windowing
        if sentences.len() <= window_size {
            chunks.push(sentences.join(" "));
        } else {
            for window in sentences.windows(window_size) {
                chunks.push(window.join(" "));
            }
        }

        for clean_chunk in chunks {
            // Tokenize the chunk
            let encoding = data.cross_encoder_tokenizer
                .encode((clean_chunk.as_str(), claim), true)
                .expect("Failed to tokenize chunk and claim");

            let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
            let attention_mask: Vec<i64> = encoding.get_attention_mask().iter().map(|&x| x as i64).collect();
            let token_type_ids: Vec<i64> = encoding.get_type_ids().iter().map(|&x| x as i64).collect();

            let shape = vec![1, input_ids.len() as i64];

            // Get a cross-encoder model from the pool asynchronously
            let mut cross_encoder = data.cross_encoder_pool.get().await.expect("Failed to get Cross-Encoder from pool");

            // Offload the Cross-Encoder inference to a blocking thread
            let logits_result = web::block(move || -> Result<Vec<f32>, ort::Error> {
                let outputs = cross_encoder.run(ort::inputs![
                    "input_ids" => Tensor::from_array((shape.clone(), input_ids)).unwrap(),
                    "attention_mask" => Tensor::from_array((shape.clone(), attention_mask)).unwrap(),
                    "token_type_ids" => Tensor::from_array((shape.clone(), token_type_ids)).unwrap(),
                ])?;
                
                let (_shape, logits_data) = outputs["logits"].try_extract_tensor::<f32>()?;
                
                // Return an owned Vector of the 3 logits [refute, support, neutral]
                Ok(logits_data[0..3].to_vec())
            })
            .await
            .expect("Cross-Encoder blocking task panicked");

            let logits = match logits_result {
                Ok(l) => l, 
                Err(e) => {
                    println!("Warning: Skipping chunk. ONNX inference failed. Error: {:?}", e);
                    continue;
                }
            };

            let max_logit: f32 = logits[0].max(logits[1].max(logits[2]));
            let exp_logits: Vec<f32> = logits.iter().map(|&x| (x-max_logit).exp()).collect();
            let sum_logits: f32 = exp_logits.iter().sum();
            let softmax_probs: Vec<f32> = exp_logits.iter().map(|&x| x / sum_logits).collect();

            // We discard chunks with neutral probability for max pooling
            let refute_prob = softmax_probs[0];
            let support_prob = softmax_probs[1];

            let chunk_signal = refute_prob.max(support_prob);

            // Keep only the highest signal chunk for chunk-level max pooling
            if chunk_signal > max_signal {
                max_signal = chunk_signal;
                best_support = support_prob;
                best_refute = refute_prob;
            }
        }

        let stance;
        let confidence;

        // If confidence is lower than 0.50 for the document we assume stance as neutral since the cross-encoder uncertainty
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

        // Perform document-level thresholded weighted sum pooling
        let weighted_confidence = confidence * _score;

        // Only include documents with confidence > 0.80 for the weighted thresholded sum pooling
        if confidence > 0.80 {
            if stance == "SUPPORT" {
                weighted_support_sum += weighted_confidence;
            } else if stance == "REFUTE" {
                weighted_refute_sum += weighted_confidence;
            }
        }

        // Track the highest neutral score just in case all documents are filtered out
        if stance == "NEUTRAL" && confidence > highest_neutral_score {
            highest_neutral_score = confidence;
        }

        evidence_list.push(types::Evidence {
            title: title.to_string(),
            source: source.to_string(),
            snippet: format!("{}...", &abstract_text[..200.min(abstract_text.len())]),
            stance,
            confidence,
        });
    }

    // Map output to match the True/False expectation for your benchmark script
    let mut final_verdict = "NEUTRAL".to_string();
    let aggregate_confidence;

    // The stance with the most accumulated weighted evidence wins
    if weighted_support_sum > weighted_refute_sum && weighted_support_sum > 0.0 {
        final_verdict = "TRUE".to_string();

        // Normalize the confidence
        let total_sum = weighted_support_sum + weighted_refute_sum;
        aggregate_confidence = weighted_support_sum / total_sum;

    } else if weighted_refute_sum > weighted_support_sum && weighted_refute_sum > 0.0 {
        final_verdict = "FALSE".to_string();

        let total_sum = weighted_support_sum + weighted_refute_sum;
        aggregate_confidence = weighted_refute_sum / total_sum;

    } else {
        // Fallback if no documents passed the 0.80 threshold
        aggregate_confidence = highest_neutral_score;
    }

    let response = VerifyResponse {
        final_verdict,
        aggregate_confidence,
        evidence: evidence_list,
    };

    HttpResponse::Ok().json(response)
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // Initialize ONNX runtime engine
    ort::init().with_name("verity_inference_engine").commit();

    // Initialize Qdrant Client (Connecting over the Docker network)
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

    // Configure the model pools
    let pool_size = 2;

    let bi_encoder_pool = deadpool::managed::Pool::builder(BiEncoderManager)
        .max_size(pool_size)
        .build()
        .expect("Failed to create Bi-Encoder pool");

    let cross_encoder_pool = deadpool::managed::Pool::builder(CrossEncoderManager)
        .max_size(pool_size)
        .build()
        .expect("Failed to create Cross-Encoder pool");

    // Create app state 
    let app_state = web::Data::new(AppState {
        qdrant_client: client,
        bi_encoder_pool,
        bi_encoder_tokenizer: Tokenizer::from_file("models/bge_small/tokenizer.json").unwrap(),
        cross_encoder_pool,
        cross_encoder_tokenizer: Tokenizer::from_file("models/pubmedbert/tokenizer.json").unwrap(),
    });

    println!("Started Verity Rust API on 0.0.0.0:8080...");

    // Start the HTTP Server
    HttpServer::new(move || {
        App::new()
            .app_data(app_state.clone())
            .service(verify_claim)
    })
    .workers(2)     // Limit to 2 workers for development
    .bind(("0.0.0.0", 8080))?
    .run()
    .await
}
