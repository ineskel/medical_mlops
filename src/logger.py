"""
src/logger.py
Inference request logger — writes JSONL to logs/inference_log.jsonl
Covers: request logging for monitoring (MLOps Part 2, slide 19)
"""
import json
import time
import uuid
from datetime import datetime
from pathlib import Path

ROOT     = Path(__file__).parent.parent
LOG_DIR  = ROOT / "logs"
LOG_FILE = LOG_DIR / "inference_log.jsonl"
LOG_DIR.mkdir(exist_ok=True)


def log_request(
    model_name: str,
    image_name: str,
    predictions: dict,
    positives: list,
    latency_ms: float,
    threshold_mode: str,
    source: str = "api",          # "api" or "dashboard"
    hospital_id: str | None = None,
):
    """Append one inference record to the JSONL log."""
    probs = [v["probability"] for v in predictions.values()]
    entry = {
        "request_id":     str(uuid.uuid4())[:8],
        "timestamp":      datetime.utcnow().isoformat() + "Z",
        "source":         source,
        "hospital_id":    hospital_id,
        "model":          model_name,
        "image":          image_name,
        "latency_ms":     round(latency_ms, 2),
        "threshold_mode": threshold_mode,
        "n_positives":    len(positives),
        "positives":      positives,
        "max_prob":       round(max(probs), 4),
        "mean_prob":      round(sum(probs) / len(probs), 4),
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return entry