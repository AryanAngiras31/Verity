use serde::{Deserialize, Serialize};

// The incoming request payload
#[derive(Deserialize)]
pub struct VerifyRequest {
    pub claim: String,
}

// The outgoing response payload
#[derive(Serialize)]
pub struct VerifyResponse {
    pub final_verdict: String,
    pub aggregate_confidence: f32,
    pub evidence: Vec<Evidence>,
}

// The individual evidence cards for the React frontend
#[derive(Serialize)]
pub struct Evidence {
    pub title: String,
    pub source: String,
    pub snippet: String,
    pub stance: String,
    pub confidence: f32,
}
