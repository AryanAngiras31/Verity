mod types;
use types::{VerifyRequest, VerifyResponse};

use std::env;

use actix_web::{post, web, App, HttpServer, Responder, HttpResponse};

use qdrant_client::Qdrant;
use qdrant_client::qdrant::Value;

use ort::session::Session;
use ort::value::Tensor;
use tokenizers::Tokenizer;
use ndarray::Array2;

// This struct holds our shared application state
struct AppState {
    qdrant_client: Qdrant,
    specter_model: Session,
    specter_tokenizer: Tokenizer,
    deberta_model: Session,
    deberta_tokenizer: Tokenizer,
}

#[post("/api/verify")]
async fn verify_claim(
    req_body: web::Json<VerifyRequest>,
    data: web::Data<AppState>,
) -> impl Responder {
    let claim = &req_body.claim;

    println!("Received claim: {}", claim);

    // Embed the claim using the Specter 2
    let encoding = data.specter_tokenizer.encode(claim.as_str(), true).expect("Failed to encode claim using SPECTER 2");

    // ONNX expects 64 bit integers for input ids and masks
    let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
    let attention_mask: Vec<i64> = encoding.get_attention_mask().iter().map(|&x| x as i64).collect();

    // This is the number of sequences
    let batch_size = 1;
    // This is the number of tokens in the sequence
    let seq_len = input_ids.len();

    // Convert the raw vectors into 2D ndarray tensors
    let input_ids_array = ndarray::Array2::from_shape_vec((batch_size, seq_len), input_ids).unwrap();
    let attention_mask_array = ndarray::Array2::from_shape_vec((batch_size, seq_len), attention_mask).unwrap();

    // Convert ndarray tensors to ONNX tensors
    let input_ids_tensor = Tensor::from_array(input_ids_array).unwrap();
    let attention_mask_tensor = Tensor::from_array(attention_mask_array).unwrap();

    // Run the SPECTER 2 model
    let outputs = data.specter_model.run(ort::inputs![
        "input_ids" => input_ids_tensor,
        "attention_mask" => attention_mask_tensor,
    ]).expect("Failed to run SPECTER 2 model");

    println!("{:?}", outputs);

    // Extract the embedding
    // SPECTER uses the [CLS] token (the very first token at index 0) as the embedding for the whole sentence.
    let last_hidden_state = outputs["last_hidden_state"]    // The output is a tensor of shape [batch_size, sequence_length, hidden_dimension]
        .try_extract_tensor::<f32>()
        .expect("Failed to extract float tensor from ONNX output");

    // Slice out the first token's 768-dimensional vector
    let embedding: Vec<f32> = last_hidden_state
        .view()
        .slice(ndarray::s![0, 0, ..]) // [Batch 0, Token 0, All 768 dimensions]
        .to_vec();

    println!("Successfully generated embedding of size: {}", embedding.len());

    // Dummy response to ensure the routing works before adding ML logic
    let dummy_response = VerifyResponse {
        final_verdict: "NEUTRAL".to_string(),
        aggregate_confidence: 0.0,
        evidence: vec![],
    };

    HttpResponse::Ok().json(dummy_response)
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // 1. Initialize ONNX runtime engine
    ort::init().with_name("verity_inference_engine").commit();

    // 2. Initialize Qdrant Client (Connecting over the Docker network)
    let qdrant_url = env::var("QDRANT_URL").unwrap_or_else(|_| "http://qdrant:6334".to_string());

    println!("Connecting to Qdrant at {}...", qdrant_url);
    let client = Qdrant::from_url(&qdrant_url)
        .build()
        .expect("Failed to build Qdrant Client");

    // 4. Wrap client in Actix web::Data for thread-safe sharing
    let app_state = web::Data::new(AppState {
        qdrant_client: client,
        specter_model: Session::new("models/specter2/model.onnx").expect("Failed to load Specter model"),
        specter_tokenizer: Tokenizer::from_file("models/specter2/tokenizer.json").expect("Failed to load Specter tokenizer"),
        deberta_model: Session::new("models/deberta/model.onnx").expect("Failed to load DeBERTa model"),
        deberta_tokenizer: Tokenizer::from_file("models/deberta/tokenizer.json").expect("Failed to load DeBERTa tokenizer"),
    });

    println!("Starting Verity Rust API on 0.0.0.0:8080...");

    // 5. Start the HTTP Server
    HttpServer::new(move || {
        App::new()
            .app_data(app_state.clone())
            .service(verify_claim)
    })
    .bind(("0.0.0.0", 8080))?
    .run()
    .await
}
