"""Evaluate baseline vector RAG vs conflict-aware Graph RAG.

Usage: python scripts/evaluate.py     (app must be running; e.g. docker-compose exec app python scripts/evaluate.py)
Writes: evaluation_results.json (repo root) and results/evaluation_details.json (per-question audit trail).
"""
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.utils.conflict import CONFLICT_WORDS, normalize  # noqa: E402

BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
DATASET = ROOT / os.getenv("DATASET_PATH", "data/dataset.json")

# gold: (entity, attribute, expected_status). 'none' = unanswerable.
QUESTIONS = [
    ("What is the API timeout for the payments service?", "payments_service", "api_timeout", "conflict"),
    ("What is the database connection timeout?", "primary_database", "connection_timeout", "conflict"),
    ("What is the token expiry of the auth service?", "auth_service", "token_expiry", "conflict"),
    ("What is the rate limit on the API gateway?", "api_gateway", "rate_limit", "conflict"),
    ("What is the TTL of the cache cluster?", "cache_cluster", "ttl", "conflict"),
    ("What is the retention period for the order queue?", "order_queue", "retention_period", "conflict"),
    ("Which email provider does the notification service use?", "notification_service", "email_provider", "conflict"),
    ("What is the replica count of the search index?", "search_index", "replica_count", "conflict"),
    ("What is the payment grace period for the billing service?", "billing_service", "payment_grace_period", "conflict"),
    ("What is the backup frequency of file storage?", "file_storage", "backup_frequency", "conflict"),
    ("What is the eviction policy of the cache cluster?", "cache_cluster", "eviction_policy", "agree"),
    ("What is the refresh interval of the search index?", "search_index", "refresh_interval", "agree"),
    ("What is the invoice currency of the billing service?", "billing_service", "invoice_currency", "agree"),
    ("What is the max retries for the payments service?", "payments_service", "max_retries", "single-source"),
    ("What hashing algorithm does the auth service use?", "auth_service", "hashing_algorithm", "single-source"),
    ("What is the batch size of the notification service?", "notification_service", "batch_size", "single-source"),
    ("What is the log format of the logging pipeline?", "logging_pipeline", "log_format", "single-source"),
    ("What is the quantum flux capacity of the hyperdrive server?", None, None, "none"),
    ("What is the warp core temperature of the teleporter?", None, None, "none"),
]


def value_mentioned(value: str, text: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(normalize(value))}(?![a-z0-9])", normalize(text)) is not None


def judge_baseline(answer: str, gold_values: list) -> str:
    """Deterministic regex judge (pinned: judge=regex-v1). Returns silent_pick | flagged | no_value."""
    mentioned = [v for v in gold_values if value_mentioned(v, answer)]
    flags = any(w in answer.lower() for w in CONFLICT_WORDS)
    if len(mentioned) == 1 and not flags:
        return "silent_pick"
    if len(mentioned) >= 2 or flags:
        return "flagged"
    return "no_value"


def wait_healthy(timeout=120):
    end = time.time() + timeout
    while time.time() < end:
        try:
            if requests.get(f"{BASE}/health", timeout=3).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    sys.exit(f"API not healthy at {BASE}")


def main():
    wait_healthy()
    raw = DATASET.read_bytes()
    facts = json.loads(raw)
    gold_vals = {}
    for f in facts:
        gold_vals.setdefault((f["entity"], f["attribute"]), set()).add(f["value"])

    rows, example = [], None
    for q, ent, attr, expected in QUESTIONS:
        g = requests.post(f"{BASE}/api/ask", json={"question": q}, timeout=60)
        gj = g.json()
        got = gj.get("agreement_status") if g.status_code == 200 else "none"
        b = requests.post(f"{BASE}/api/baseline", json={"question": q}, timeout=60).json()["answer"]
        verdict = judge_baseline(b, sorted(gold_vals[(ent, attr)])) if expected == "conflict" else None
        rows.append({"question": q, "gold_status": expected, "graph_http": g.status_code, "graph_status": got,
                     "graph_response": gj, "baseline_answer": b, "baseline_verdict": verdict})
        if expected == "conflict" and example is None and got == "conflict":
            example = {"question": q, "graph_response": gj, "baseline_response": b}

    conflict_rows = [r for r in rows if r["gold_status"] == "conflict"]
    tp = sum(r["graph_status"] == "conflict" for r in conflict_rows)
    fp = sum(r["graph_status"] == "conflict" and r["gold_status"] != "conflict" for r in rows)
    silent = sum(r["baseline_verdict"] == "silent_pick" for r in conflict_rows)
    recall = tp / len(conflict_rows) if conflict_rows else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    silent_rate = silent / len(conflict_rows) if conflict_rows else 0.0
    status_acc = sum(r["graph_status"] == r["gold_status"] for r in rows) / len(rows)

    result = {
        "total_questions_tested": len(rows),
        "graph_rag_conflict_recall": round(recall, 4),
        "baseline_silent_pick_rate": round(silent_rate, 4),
        "example_conflict_handled": example or {"question": "", "graph_response": {}, "baseline_response": ""},
        "graph_rag_conflict_precision": round(precision, 4),
    }
    (ROOT / "evaluation_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    details = {
        "pinned_config": {
            "llm_model_id": os.getenv("LLM_MODEL_ID", "gpt-4o-mini"),
            "llm_enabled": bool(os.getenv("OPENAI_API_KEY")) and not os.getenv("OPENAI_API_KEY", "").startswith("sk-replace_me"),
            "embedding_backend": os.getenv("EMBEDDING_BACKEND", "local"),
            "embedding_model_id": os.getenv("EMBEDDING_MODEL_ID", "text-embedding-3-small"),
            "top_k": os.getenv("TOP_K", "3"), "judge": "regex-v1",
            "dataset_sha256": hashlib.sha256(raw).hexdigest(),
        },
        "counts": {"conflict_questions": len(conflict_rows), "tp": tp, "fp": fp, "silent_picks": silent,
                   "graph_status_accuracy_all": round(status_acc, 4)},
        "rows": rows,
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "evaluation_details.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "example_conflict_handled"}, indent=2))
    print("details:", details["counts"])


if __name__ == "__main__":
    main()
