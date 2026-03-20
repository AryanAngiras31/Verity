mod types;

use actix_web::{post, web, App, HttpServer, Responder, HttpResponse};
use qdrant_client::Qdrant;
use qdrant_client::qdrant::Value;
use types::{VerifyRequest, VerifyResponse};
use std::env;
use ort::{GraphOptimizationLevel, Session};
use tokenizers::Tokenizer;

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
    ort::init().with_name("verity_inference_engine").commit().expect("Failed to initialize ONNX runtime engine");

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
