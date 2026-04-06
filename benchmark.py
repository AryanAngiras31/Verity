# docker run --rm -it --network host -v "$(pwd):/app" -w /app python:3.10-slim bash -c "pip install requests && API_URL=http://localhost:8080/api/verify python benchmark.py"
import json
import os
import time
from ftplib import FTP_PORT

import requests

# We use an environment variable so Docker can map the network properly
API_URL = os.getenv("API_URL", "http://localhost:8080/api/verify")


class label_proportions:
    def __init__(self):
        self.support = 0
        self.contradict = 0
        self.neutral = 0


def parse_ground_truth(label):
    """
    Parses the ground truth label into a format the API expects.
    Args:
        label (str): The ground truth label from the dataset.
    Returns:
        str: The parsed label ("TRUE", "FALSE", or "NEUTRAL").
    """
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
        res = requests.post(API_URL, json=payload, timeout=60)
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
    print(f"\n====== Running Benchmark (Threshold: {threshold:.2f}) ======")
    correct = 0
    total = 0

    # Initialize dictionaries for multi-class F1 tracking
    classes = ["TRUE", "FALSE", "NEUTRAL"]
    TP = {c: 0 for c in classes}
    FP = {c: 0 for c in classes}
    FN = {c: 0 for c in classes}

    # Initialize label proportions to zero
    model_label_proportions = label_proportions()
    true_label_proportions = label_proportions()

    with open(dataset_path, "r") as f:
        for line in f:
            if total >= limit:
                break
            data = json.loads(line)
            claim = data["claim"]
            ground_truth = parse_ground_truth(data["label"])

            try:
                prediction = test_claim_against_api(claim, threshold)
            except Exception as e:
                prediction = "CONNECTION_FAILED"

            if prediction == "CONNECTION_FAILED":
                print("Could not connect to Rust API. Is it running?")
                # Return 0 to abort early if the server is completely down
                return 0

            # Update predictions tracker
            match prediction:
                case "TRUE":
                    model_label_proportions.support += 1
                case "FALSE":
                    model_label_proportions.contradict += 1
                case "NEUTRAL":
                    model_label_proportions.neutral += 1

            # Update ground truth tracker
            match ground_truth:
                case "TRUE":
                    true_label_proportions.support += 1
                case "FALSE":
                    true_label_proportions.contradict += 1
                case "NEUTRAL":
                    true_label_proportions.neutral += 1

            # Update multi-class confusion metrics
            if prediction == ground_truth:
                correct += 1
                if ground_truth in classes:
                    TP[ground_truth] += 1
            else:
                if prediction in classes:
                    FP[prediction] += 1  # The model guessed this class incorrectly
                if ground_truth in classes:
                    FN[ground_truth] += 1  # The model missed the actual class

            total += 1
            if total % 10 == 0:
                print(f"  Processed {total}/{limit} claims...")

    # Display results
    print(f"\n---- Benchmark Results for Threshold {threshold} ----")

    # Calculate Precision, Recall, and F1 for each class
    f1_scores = {}
    for c in classes:
        precision = TP[c] / (TP[c] + FP[c]) if (TP[c] + FP[c]) > 0 else 0
        recall = TP[c] / (TP[c] + FN[c]) if (TP[c] + FN[c]) > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        f1_scores[c] = f1
        print(
            f"Class '{c}': Precision: {precision:.2f} | Recall: {recall:.2f} | F1: {f1:.2f}"
        )

    # Display accuracy
    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"\nAccuracy: {accuracy:.2f}% ({correct}/{total})")

    # Display weighted F1
    weighted_f1 = (
        f1_scores["TRUE"] * true_label_proportions.support
        + f1_scores["FALSE"] * true_label_proportions.contradict
        + f1_scores["NEUTRAL"] * true_label_proportions.neutral
    ) / total
    print(f"Weighted F1: {weighted_f1:.4f}")

    print(
        f"\nModel Label Proportions: Support={model_label_proportions.support / total:.2f}, Contradict={model_label_proportions.contradict / total:.2f}, Neutral={model_label_proportions.neutral / total:.2f}"
    )
    print(
        f"True Label Proportions: Support={true_label_proportions.support / total:.2f}, Contradict={true_label_proportions.contradict / total:.2f}, Neutral={true_label_proportions.neutral / total:.2f}"
    )
    return weighted_f1


def hyperparameter_tuning(dataset_path, thresholds_to_test, limit=50):
    """Sweeps multiple thresholds to find the mathematical 'sweet spot'."""
    print(
        f"\nStarting Hyperparameter Tuning on {len(thresholds_to_test)} thresholds..."
    )
    results = {}

    for t in thresholds_to_test:
        f1 = run_benchmark(dataset_path, threshold=t, limit=limit)
        results[t] = f1
        time.sleep(1)  # Let the Rust server catch its breath

    print("\n=====================================")
    print("HYPERPARAMETER TUNING RESULTS:")
    print("=====================================")
    best_t = None
    best_f1 = -1

    for t, f1 in results.items():
        print(f"Threshold {t:.2f} -> {f1:.2f} F1 Score")
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    print(f"\nOPTIMAL QDRANT THRESHOLD: {best_t:.2f} ({best_f1:.2f} F1 Score)")


if __name__ == "__main__":
    DATASET_FILE = "data/hybrid_claims_consolidated.jsonl"
    # We will test 5 different strictness levels for the Qdrant Bouncer
    thresholds = [0.60, 0.65, 0.70, 0.75]

    # I set the limit to 50 so your first test finishes in ~10 seconds.
    # Once you confirm it works, change limit to 500 to evaluate the whole corpus!
    hyperparameter_tuning(DATASET_FILE, thresholds, limit=50)
