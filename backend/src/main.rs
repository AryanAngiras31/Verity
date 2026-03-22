mod types;
use types::{VerifyRequest, VerifyResponse};

use std::sync::Mutex;
use std::env;

use actix_web::{post, web, App, HttpServer, Responder, HttpResponse};

use qdrant_client::Qdrant;

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

#[post("/api/verify")]
async fn verify_claim(
    req_body: web::Json<VerifyRequest>,
    data: web::Data<AppState>,
) -> impl Responder {
    let claim = &req_body.claim;

    println!("Received claim: {}", claim);

    // Embed the claim using the Specter 2
    let encoding = data.specter_tokenizer.encode(claim.as_str(), true).expect("Failed to encode claim using SPECTER 2");

    // SPECTER 2 is build on a BERT style architecture and requires these three inputs
    // ONNX expects 64 bit integers for input ids and masks
    // Ids of the tokens in the sequence
    let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
    // Tells the model which tokens are padding and which are actual tokens. (1 for actual tokens, 0 for padding)
    let attention_mask: Vec<i64> = encoding.get_attention_mask().iter().map(|&x| x as i64).collect();
    // Tells the model which tokens belong to which sequence. (0 for the first sequence, 1 for the second, etc.)
    let token_type_ids: Vec<i64> = encoding.get_type_ids().iter().map(|&x| x as i64).collect();

    // This is the number of sequences
    let batch_size: usize = 1;
    // This is the number of tokens in the sequence
    let seq_len: usize = input_ids.len();
    // Pass a standard Rust Tuple (Shape, Data) directly to ONNX.
    let shape = vec![batch_size, seq_len];

    // Convert ndarray tensors to ONNX tensors
    let input_ids_tensor = Tensor::from_array((shape.clone(), input_ids)).unwrap();
    let attention_mask_tensor = Tensor::from_array((shape.clone(), attention_mask)).unwrap();
    let token_type_ids_tensor = Tensor::from_array((shape.clone(), token_type_ids)).unwrap();

    // Get the Mutex lock to get safe, mutable access to the model
    let mut specter = data.specter_model.lock().expect("Could not get Mutex lock for SPECTER 2 model");

    // Run the SPECTER 2 model
    let outputs = specter.run(ort::inputs![
        "input_ids" => input_ids_tensor,
        "attention_mask" => attention_mask_tensor,
        "token_type_ids" => token_type_ids_tensor,
    ]).expect("Failed to run SPECTER 2 model");

    // Extract the embedding
    // SPECTER uses the [CLS] token (the very first token at index 0) as the embedding for the whole sentence.
    let (_shape, tensor_data) = outputs["last_hidden_state"]    // The output is a tensor of shape [batch_size, sequence_length, hidden_dimension]
        .try_extract_tensor::<f32>()
        .expect("Failed to extract float tensor from ONNX output");

    // Drop the mutex lock as soon as we're done with the embedding
    drop(data.specter_model.lock());

    // Because the tensor is flattened, the first token's 768 dimensions
    // are simply the first 768 numbers in the array!
    let embedding: Vec<f32> = tensor_data[0..768].to_vec();

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

    println!("Backend connecting to Qdrant at {}...", qdrant_url);
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

    // 4. Wrap client in Actix web::Data for thread-safe sharing
    let app_state = web::Data::new(AppState {
        qdrant_client: client,
        specter_model: Mutex::new(Session::builder().unwrap().commit_from_file("models/specter2/model.onnx").expect("Failed to load Specter model")),
        specter_tokenizer: Tokenizer::from_file("models/specter2/tokenizer.json").expect("Failed to load Specter tokenizer"),
        deberta_model: Mutex::new(Session::builder().unwrap().commit_from_file("models/deberta/model.onnx").expect("Failed to load DeBERTa model")),
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
