# docker run --rm -it --network host -v "$(pwd):/app" -w /app python:3.10-slim bash -c "pip install requests && API_URL=http://localhost:8080/api/verify python benchmark.py"
import json
import os
import time

import requests

# We use an environment variable so Docker can map the network properly
API_URL = os.getenv("API_URL", "http://localhost:8080/api/verify")


class label_proportions:
    def __init__(self):
        self.support = 0
        self.contradict = 0
        self.neutral = 0


def parse_ground_truth(label):
    label_map = {
        "SUPPORT": "TRUE",
        "CONTRADICT": "FALSE",
        "NEUTRAL": "NEUTRAL",
        "": "NEUTRAL",
    }

    return label_map.get(label, "NEUTRAL")


def test_claim_against_api(claim, threshold):
    """Fires a single claim to the Rust backend."""
    payload = {"claim": claim, "qdrant_threshold": threshold}
    try:
        res = requests.post(API_URL, json=payload, timeout=30)
        if res.status_code == 200:
            return res.json().get("final_verdict", "ERROR")
        return "ERROR"
    except requests.exceptions.Timeout:
        print("\n[Warning] Request timed out!")
        return "TIMEOUT"
    except Exception as e:
        print(f"\n[Error] {e}")
        return "CONNECTION_FAILED"


def run_benchmark(dataset_path, threshold, limit=50):
    """Evaluates the pipeline against the dataset using a specific threshold."""
    print(f"\n--- Running Benchmark (Threshold: {threshold:.2f}) ---")
    correct = 0
    total = 0

    model_label_proportions = label_proportions()
    true_label_proportions = label_proportions()

    with open(dataset_path, "r") as f:
        for line in f:
            if total >= limit:
                break

            data = json.loads(line)
            claim = data["claim"]
            ground_truth = parse_ground_truth(data["label"])

            prediction = test_claim_against_api(claim, threshold)

            if prediction == "CONNECTION_FAILED":
                print("Could not connect to Rust API. Is it running?")
                return 0

            match prediction:
                case "TRUE":
                    model_label_proportions.support += 1
                case "FALSE":
                    model_label_proportions.contradict += 1
                case "NEUTRAL":
                    model_label_proportions.neutral += 1
            match ground_truth:
                case "TRUE":
                    true_label_proportions.support += 1
                case "FALSE":
                    true_label_proportions.contradict += 1
                case "NEUTRAL":
                    true_label_proportions.neutral += 1

            if prediction == ground_truth:
                correct += 1

            total += 1
            if total % 10 == 0:
                print(f"  Processed {total}/{limit} claims...")

    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"Pass Complete! Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(
        f"Model Label Proportions: Support={model_label_proportions.support / total:.2f}, Contradict={model_label_proportions.contradict / total:.2f}, Neutral={model_label_proportions.neutral / total:.2f}"
    )
    print(
        f"True Label Proportions: Support={true_label_proportions.support / total:.2f}, Contradict={true_label_proportions.contradict / total:.2f}, Neutral={true_label_proportions.neutral / total:.2f}"
    )
    return accuracy


def hyperparameter_tuning(dataset_path, thresholds_to_test, limit=50):
    """Sweeps multiple thresholds to find the mathematical 'sweet spot'."""
    print(
        f"\nStarting Hyperparameter Tuning on {len(thresholds_to_test)} thresholds..."
    )
    results = {}

    for t in thresholds_to_test:
        acc = run_benchmark(dataset_path, threshold=t, limit=limit)
        results[t] = acc
        time.sleep(1)  # Let the Rust server catch its breath

    print("\n=====================================")
    print("HYPERPARAMETER TUNING RESULTS:")
    print("=====================================")
    best_t = None
    best_acc = -1

    for t, acc in results.items():
        print(f"Threshold {t:.2f} -> {acc:.2f}% Accuracy")
        if acc > best_acc:
            best_acc = acc
            best_t = t

    print(f"\nOPTIMAL QDRANT THRESHOLD: {best_t:.2f} ({best_acc:.2f}% Accuracy)")


if __name__ == "__main__":
    DATASET_FILE = "data/hybrid_claims_consolidated.jsonl"

    # We will test 5 different strictness levels for the Qdrant Bouncer
    thresholds = [0.60, 0.65, 0.70, 0.75]

    # I set the limit to 50 so your first test finishes in ~10 seconds.
    # Once you confirm it works, change limit to 500 to evaluate the whole corpus!
    hyperparameter_tuning(DATASET_FILE, thresholds, limit=50)
