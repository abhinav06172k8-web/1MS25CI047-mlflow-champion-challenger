"""Shared settings for the Champion-Challenger mini project (USN 1MS25CI047)."""
import os

STUDENT_USN = "1MS25CI047"
STUDENT_NAME = "GOLLA ABHINAV"

# Tracking server (start it with: python -m mlflow server --host 127.0.0.1 --port 5000)
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = "hf-sentiment-champion-challenger"
REGISTERED_MODEL_NAME = "sentiment-model"

TASK = "sentiment"
DATASET_NAME = "nyu-mll/glue sst2 (validation split)"
EVAL_SAMPLES = 200
SEED = 42
BATCH_SIZE = 8
MAX_LENGTH = 128

# Model A = current Champion, Model B = Challenger
MODELS = {
    "A": {
        "hf_model_id": "distilbert-base-uncased-finetuned-sst-2-english",
        "short_name": "distilbert-sst2",
        "role": "champion",
    },
    "B": {
        "hf_model_id": "siebert/sentiment-roberta-large-english",
        "short_name": "roberta-large",
        "role": "challenger",
    },
}

# Champion rule (written BEFORE the final run - see README section 3)
PRIMARY_METRIC = "macro_f1"          # higher is better
MARGIN = 0.01                        # Challenger must beat Champion by >= 1 percentage point
MAX_LATENCY_RATIO = 10.0             # guardrail: reject Challenger if avg latency > 10x Champion

LABEL_MAP = {0: "negative", 1: "positive"}

DATA_DIR = "data"
OUTPUT_DIR = "outputs"
