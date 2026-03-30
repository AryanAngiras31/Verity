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
    specter_model: Mutex<Session>,
    specter_tokenizer: Tokenizer,
    deberta_model: Mutex<Session>,
    deberta_tokenizer: Tokenizer,
}

const COLLECTION_NAME: &str = "verity_hybrid_corpus";

#[post("/api/verify")]
async fn verify_claim(
    req_body: web::Json<VerifyRequest>,
    data: web::Data<AppState>,
) -> impl Responder {
    let claim: &str = &req_body.claim;

    println!("\nReceived claim: {}", claim);
    println!("--------------------------------------------------");

    // Embed the claim using the Specter 2
    let encoding = data.specter_tokenizer.encode(claim, true).expect("Failed to encode claim using SPECTER 2");

    // SPECTER 2 is build on a BERT style architecture and requires these three inputs
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

    // Use a scoping block to automatically drop the pointer from output to specter and the Mutex lock after the embedding is extracted
    let embedding: Vec<f32> = {
        // Get the Mutex lock to get safe, mutable access to the model
        let mut specter = data.specter_model.lock().expect("Could not get Mutex lock for SPECTER 2 model");

        // Run the SPECTER 2 model
        let outputs = specter.run(ort::inputs![
            "input_ids" => Tensor::from_array((shape.clone(), input_ids)).unwrap(),
            "attention_mask" => Tensor::from_array((shape.clone(), attention_mask)).unwrap(),
            "token_type_ids" => Tensor::from_array((shape.clone(), token_type_ids)).unwrap(),
        ]).expect("Failed to run SPECTER 2 model");

        // Extract the embedding
        // SPECTER uses the [CLS] token (the very first token at index 0) as the embedding for the whole sentence.
        let (_shape, tensor_data) = outputs["last_hidden_state"]    // The output is a tensor of shape [batch_size, sequence_length, hidden_dimension]
            .try_extract_tensor::<f32>()
            .expect("Failed to extract float tensor from ONNX output");

        // Because the tensor is flattened, the first token's 768 dimensions are simply the first 768 numbers in the array. Return them.
        tensor_data[0..768].to_vec()
    };

    // Extract the dynamic threshold from the request (default to 0.80)
    let threshold: f32 = req_body.qdrant_threshold.unwrap_or(0.80);
    // Query Qdrant for top 5 matches
    let query_request = QueryPointsBuilder::new(COLLECTION_NAME)
        .query(embedding)
        .limit(5)
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

    // DeBERTa Cross Encoder Inference
    let mut evidence_list: Vec<Evidence> = Vec::new();

    // Get the Mutex lock for the DeBERTa model
    let mut deberta = data.deberta_model.lock().expect("Could not get Mutex lock for DeBERTa model");

    // TRACKERS FOR THRESHOLDED MEAN POOLING
    let mut valid_support_sum = 0.0;
    let mut valid_refute_sum = 0.0;
    let mut support_count = 0.0;
    let mut refute_count = 0.0;

    for hit in response.result.iter() {
        let payload = &hit.payload;

        // Helper to extract strings from the Qdrant Value type safely
        let get_string = |key: &str| {
            payload.get(key)
                .and_then(|v| v.as_str())
                .map(|s| s.as_str())
                .unwrap_or("Unknown")
        };

        let title = get_string("title");
        let source = get_string("dataset_source");

        // Extract the abstract
        let abstract_text = get_string("abstract");

        let _score = hit.score;

        let sentences: Vec<String> = abstract_text
            .split('.')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .map(|s| format!("{}.", s))
            .collect();

        /*println!("--------------------------------------------------");
        println!("Score: {:.4} | Title: {} [{}]", _score, title, source);
        println!("--------------------------------------------------");*/

        // We take the max of the confidences from each sentence to represent the document's overall support/refute/neutral score
        let mut doc_max_support: f32 = 0.0;
        let mut doc_max_refute: f32 = 0.0;

        for clean_chunk in sentences {
            // Tokenize the claim and abstract text. The DeBERTa tokenizer automatically inserts the [SEP] token between them.
            // DeBERTa is trained to read Text A (The Premise) and decide if it supports or refutes Text B (The Hypothesis).
            let encoding = data.deberta_tokenizer
                .encode((clean_chunk.as_str(), claim), true)
                .expect("Failed to tokenize chunk and claim");

            // DeBERTa does not require token_type_ids, so we do not include it in the inputs.
            let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
            let attention_mask: Vec<i64> = encoding.get_attention_mask().iter().map(|&x| x as i64).collect();

            let shape = vec![1, input_ids.len() as i64];

            let outputs = deberta.run(ort::inputs![
                "input_ids" => Tensor::from_array((shape.clone(), input_ids)).unwrap(),
                "attention_mask" => Tensor::from_array((shape.clone(), attention_mask)).unwrap(),
            ]).expect("DeBERTa inference failed");

            // DeBERTa's SequenceClassification exports the final layer as "logits"
            let (_shape, logits_data) = outputs["logits"].try_extract_tensor::<f32>().unwrap();
            let logits = &logits_data[0..3];

            // Calculate SoftMax for logits. Maps logits to a range of 0 to 1 making sure they add up to 1.
            let max_logit: f32 = logits[0].max(logits[1].max(logits[2]));
            let exp_logits: Vec<f32> = logits.iter().map(|&x| (x-max_logit).exp()).collect();
            let sum_logits: f32 = exp_logits.iter().sum();
            let softmax_probs: Vec<f32> = exp_logits.iter().map(|&x| x / sum_logits).collect();

            // HuggingFace DeBERTa-v3-small NLI Mapping:
            // 0 -> Contradiction (Refute), 1 -> Entailment (Support), 2 -> Neutral
            let refute_prob = softmax_probs[0];
            let support_prob = softmax_probs[1];
            let _neutral_prob = softmax_probs[2];

            /*println!("Chunk: {}, \nrefute_prob: {:.4}, support_prob: {:.4}, _neutral_prob: {:.4}\n", clean_chunk, refute_prob, support_prob, _neutral_prob);*/

            // Track the highest scores found across ALL sentences in this document
            doc_max_refute = doc_max_refute.max(refute_prob);
            doc_max_support = doc_max_support.max(support_prob);
        }

        /*println!("title: {}, doc_max_support: {}, doc_max_refute: {}", title, doc_max_support, doc_max_refute);*/

        // Calculate the stance and confidence of the evidence document
        let stance;
        let confidence;

        if doc_max_support > doc_max_refute && doc_max_support > 0.50 {
            stance = "SUPPORT".to_string();
            confidence = doc_max_support;
        } else if doc_max_refute > doc_max_support && doc_max_refute > 0.50 {
            stance = "REFUTE".to_string();
            confidence = doc_max_refute;
        } else {
            // If neither signal was strong enough, it defaults to Neutral
            stance = "NEUTRAL".to_string();
            confidence = 1.0 - doc_max_support - doc_max_refute; // Rough estimate of neutral weight
        }

        // Apply Threshold Filtering: Only count highly confident logical stances
        if stance == "SUPPORT" && confidence > 0.55 {
            valid_support_sum += confidence;
            support_count += 1.0;
        } else if stance == "REFUTE" && confidence > 0.55 {
            valid_refute_sum += confidence;
            refute_count += 1.0;
        }

        evidence_list.push(types::Evidence {
            title: title.to_string(),
            source: source.to_string(),
            snippet: format!("{}...", &abstract_text[..200.min(abstract_text.len())]),
            stance,
            confidence,
        });
    }

    drop(deberta);

    // Verdict aggregation (Thresholded Mean Pooling)
    let avg_support = if support_count > 0.0 { valid_support_sum / support_count } else { 0.0 };
    let avg_refute = if refute_count > 0.0 { valid_refute_sum / refute_count } else { 0.0 };

    let mut final_verdict = "NEUTRAL".to_string();
    let mut aggregate_confidence = 0.0;

    // The mathematically sound verdict: Whichever average probability is strictly the highest wins.
    if valid_support_sum > valid_refute_sum {
        final_verdict = "TRUE".to_string();
        aggregate_confidence = avg_support;
    } else if valid_refute_sum > valid_support_sum {
        final_verdict = "FALSE".to_string();
        aggregate_confidence = avg_refute;
    } else if evidence_list.iter().any(|e| e.stance == "NEUTRAL") {
        // If everything was filtered out, default to the highest Neutral score
        aggregate_confidence = evidence_list.iter().map(|e| e.confidence).fold(0.0, f32::max);
    }

    println!("\nNumber of strongly supporting documents: {} (Support Confidence: {:.2}%)", support_count, avg_support * 100.0);
    println!("Number of strongly refuting documents: {} (Refute Confidence: {:.2}%)", refute_count, avg_refute * 100.0);

    let response = VerifyResponse {
        final_verdict,
        aggregate_confidence,
        evidence: evidence_list,
    };

    println!("Verdict: {} (Confidence: {:.2}%)", response.final_verdict, response.aggregate_confidence * 100.0);

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
            specter_model: Mutex::new(Session::builder().unwrap().commit_from_file("models/specter2/model.onnx").expect("Failed to load Specter model")),
            specter_tokenizer: Tokenizer::from_file("models/specter2/tokenizer.json").expect("Failed to load Specter tokenizer"),
            deberta_model: Mutex::new(Session::builder().unwrap().commit_from_file("models/deberta/model.onnx").expect("Failed to load DeBERTa model")),
            deberta_tokenizer: Tokenizer::from_file("models/deberta/tokenizer.json").expect("Failed to load DeBERTa tokenizer"),
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
