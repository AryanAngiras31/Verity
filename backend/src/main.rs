mod types;

use actix_web::{post, web, App, HttpServer, Responder, HttpResponse};
use qdrant_client::Qdrant;
use qdrant_client::qdrant::Value;
use types::{VerifyRequest, VerifyResponse};
use std::env;

// This struct holds our shared application state
struct AppState {
    qdrant_client: Qdrant,
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
    // 1. Initialize Qdrant Client (Connecting over the Docker network)
    let qdrant_url = env::var("QDRANT_URL").unwrap_or_else(|_| "http://qdrant:6334".to_string());

    println!("Connecting to Qdrant at {}...", qdrant_url);
    let client = Qdrant::from_url(&qdrant_url)
        .build()
        .expect("Failed to build Qdrant Client");

    // 2. Wrap client in Actix web::Data for thread-safe sharing
    let app_state = web::Data::new(AppState {
        qdrant_client: client,
    });

    println!("Starting Verity Rust API on 0.0.0.0:8080...");

    // 3. Start the HTTP Server
    HttpServer::new(move || {
        App::new()
            .app_data(app_state.clone())
            .service(verify_claim)
    })
    .bind(("0.0.0.0", 8080))?
    .run()
    .await
}
