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
            .commit_from_file("models/pubmedbert/model_quantized.onnx")
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

    // 250 equates to 60-70 tokens and prevents Premise Length Dilution in the Cross-Encoder.
    if claim.chars().count() > 250 {
        return HttpResponse::BadRequest().body("Claim is too long. Please limit your claim to 250 characters.");
    }

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

    // Get a cross-encoder model from the pool asynchronously
    let mut cross_encoder = data.cross_encoder_pool.get().await.expect("Failed to get Cross-Encoder from pool");

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

        let batch_size = chunks.len();
        if batch_size == 0 {
            continue;
        }

        // Preapre the inputs for the tokenizer. A list of tuples (Chunk, Claim)
        let encoding_inputs: Vec<(String, String)> = chunks
            .iter()
            .map(|chunk| (chunk.clone(), claim.to_string()))
            .collect();
        
        // Encode the entire batch at once. The tokenizer will pad them automatically
        let encodings = data.cross_encoder_tokenizer
            .encode_batch(encoding_inputs, true)
            .unwrap();

        // Since they are padded, we can get the sequence length from the first encoding
        let seq_len = encodings[0].get_ids().len();

        // Flatten the 2D batch into 1D vectors for ONNX
        let mut input_ids: Vec<i64> = Vec::with_capacity(batch_size * seq_len);
        let mut attention_mask: Vec<i64> = Vec::with_capacity(batch_size * seq_len);
        let mut token_type_ids: Vec<i64> = Vec::with_capacity(batch_size * seq_len);

        for encoding in encodings {
            input_ids.extend(encoding.get_ids().iter().map(|&x| x as i64));
            attention_mask.extend(encoding.get_attention_mask().iter().map(|&x| x as i64));
            token_type_ids.extend(encoding.get_type_ids().iter().map(|&x| x as i64));
        }

        let shape = vec![batch_size as i64, seq_len as i64];

        // Offload the batched inference to a blocking thread
        let (logits_result, returned_cross_encoder) = web::block(move || {
            let logits_res = {
                let result = cross_encoder.run(ort::inputs![
                    "input_ids" => Tensor::from_array((shape.clone(), input_ids)).unwrap(),
                    "attention_mask" => Tensor::from_array((shape.clone(), attention_mask)).unwrap(),
                    "token_type_ids" => Tensor::from_array((shape.clone(), token_type_ids)).unwrap(),
                ]);
                
                match result {
                    Ok(outputs) => {
                        match outputs["logits"].try_extract_tensor::<f32>() {
                            Ok((_shape, logits_data)) => Ok(logits_data.to_vec()),
                            Err(e) => Err(ort::Error::from(e))
                        }
                    }
                    Err(e) => Err(e)
                }
            };
            (logits_res, cross_encoder)
        })
        .await
        .expect("Cross-Encoder blocking failed");

        // Get cross-encoder back for next iteration
        cross_encoder = returned_cross_encoder;

        let flat_logits = match logits_result {
            Ok(logits) => logits,
            Err(e) => {
                println!("Warning: Skipping document. ONNX batched inference failed. Error: {:?}", e);
                continue;
            }
        };

        // Process the flattened logits array. 
        // Every 3 items in the array correspond to [Refute, Support, Neutral] for one chunk.\
        for logits_chunk in flat_logits.chunks(3) {
            // Convert logits to SoftMax probabilities 
            let max_logit: f32 = logits_chunk[0].max(logits_chunk[1].max(logits_chunk[2]));
            let exp_logits: Vec<f32> = logits_chunk.iter().map(|&x| (x - max_logit).exp()).collect();
            let sum_logits: f32 = exp_logits.iter().sum();
            let softmax_probs: Vec<f32> = exp_logits.iter().map(|&x| x / sum_logits).collect();
            
            // Extract probabilities for each class
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

        // Find the byte index of the 200th character, or the end of the string
        let end_index = abstract_text
            .char_indices()
            .map(|(i, _)| i)
            .nth(200) 
            .unwrap_or(abstract_text.len());
        
        let snippet = if abstract_text.len() > end_index {
            format!("{}...", &abstract_text[..end_index])
        } else {
            abstract_text.to_string()
        };

        evidence_list.push(types::Evidence {
            title: title.to_string(),
            source: source.to_string(),
            snippet,
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

    // Initialize Cross-Encoder Tokenizer with padding enabled
    let mut cross_encoder_tokenizer = Tokenizer::from_file("models/pubmedbert/tokenizer.json").unwrap();

    // Enable automatic padding to the longest sequence in the batch
    cross_encoder_tokenizer.with_padding(Some(tokenizers::PaddingParams {
        strategy: tokenizers::PaddingStrategy::BatchLongest,
        ..Default::default()
    }));

    // Create app state 
    let app_state = web::Data::new(AppState {
        qdrant_client: client,
        bi_encoder_pool,
        bi_encoder_tokenizer: Tokenizer::from_file("models/bge_small/tokenizer.json").unwrap(),
        cross_encoder_pool,
        cross_encoder_tokenizer,    // Use configured tokenizer with padding enabled
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
