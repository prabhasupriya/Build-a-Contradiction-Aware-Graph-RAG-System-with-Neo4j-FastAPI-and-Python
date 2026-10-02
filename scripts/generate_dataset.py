"""Generate the synthetic, provenance-tagged fact corpus.

Run:  python scripts/generate_dataset.py
Out:  data/dataset.json (canonical) and ./dataset.json (copy for submission)

Composition (deterministic, no randomness):
  * 12 entities, 40 facts
  * 12 CONFLICT groups  (same entity+attribute, different value/source/timestamp)
  *  4 AGREE groups     (same entity+attribute, same value, different sources)
  *  8 SINGLE-source facts
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent


class ProvenanceFact(BaseModel):
    entity: str
    attribute: str
    value: str
    source: str
    timestamp: str
    confidence: float = Field(ge=0.0, le=1.0)
    context_text: str


def _text(entity: str, attribute: str, value: str) -> str:
    return f"The {attribute.replace('_', ' ')} of the {entity.replace('_', ' ')} is set to {value}."


# (entity, attribute, value, source, timestamp, confidence, optional custom text)
CONFLICTS = [
    (("payments_service", "api_timeout", "30s", "architecture-doc-v2.pdf", "2023-01", 0.8,
      "The API timeout for the payments service is strictly set to 30s."),
     ("payments_service", "api_timeout", "60s", "runbook-2024.md", "2024-06", 0.9,
      "Per the latest runbook, the payments service API timeout was raised to 60s.")),
    (("primary_database", "connection_timeout", "30s", "runbook-2022.md", "2022-03", 0.7,
      "The database connection timeout is 30s."),
     ("primary_database", "connection_timeout", "60s", "adr-014-db-timeouts.md", "2024-02", 0.9,
      "ADR-014 updates the database connection timeout to 60s.")),
    (("auth_service", "token_expiry", "15m", "security-policy-2022.pdf", "2022-09", 0.75, None),
     ("auth_service", "token_expiry", "60m", "auth-redesign-adr-2024.md", "2024-04", 0.9, None)),
    (("api_gateway", "rate_limit", "100 req/min", "gateway-handbook-2022.pdf", "2022-11", 0.7, None),
     ("api_gateway", "rate_limit", "500 req/min", "gateway-config-2024.yaml", "2024-08", 0.95, None)),
    (("cache_cluster", "ttl", "300s", "cache-guide-2023.md", "2023-05", 0.8, None),
     ("cache_cluster", "ttl", "600s", "platform-wiki-2024.md", "2024-03", 0.85, None)),
    (("order_queue", "retention_period", "7 days", "queue-spec-v1.pdf", "2022-06", 0.7, None),
     ("order_queue", "retention_period", "14 days", "queue-spec-v2.pdf", "2024-01", 0.9, None)),
    (("notification_service", "email_provider", "SendGrid", "vendor-list-2022.xlsx", "2022-08", 0.7, None),
     ("notification_service", "email_provider", "Amazon SES", "vendor-migration-adr-2024.md", "2024-05", 0.9, None)),
    (("search_index", "replica_count", "2", "search-runbook-2023.md", "2023-02", 0.8, None),
     ("search_index", "replica_count", "3", "search-capacity-plan-2024.pdf", "2024-07", 0.85, None)),
    (("billing_service", "payment_grace_period", "7 days", "billing-policy-2022.pdf", "2022-12", 0.8, None),
     ("billing_service", "payment_grace_period", "14 days", "billing-policy-2024.pdf", "2024-09", 0.9, None)),
    (("file_storage", "backup_frequency", "daily", "storage-ops-2023.md", "2023-07", 0.8, None),
     ("file_storage", "backup_frequency", "hourly", "dr-plan-2024.pdf", "2024-04", 0.9, None)),
    (("logging_pipeline", "log_retention", "30 days", "observability-guide-2022.md", "2022-10", 0.75, None),
     ("logging_pipeline", "log_retention", "90 days", "compliance-update-2024.pdf", "2024-02", 0.95, None)),
    (("analytics_warehouse", "refresh_schedule", "nightly", "data-platform-doc-2023.md", "2023-03", 0.8, None),
     ("analytics_warehouse", "refresh_schedule", "every 6 hours", "data-platform-doc-2024.md", "2024-06", 0.85, None)),
]

AGREES = [
    (("cache_cluster", "eviction_policy", "LRU", "cache-guide-2023.md", "2023-05", 0.8, None),
     ("cache_cluster", "eviction_policy", "LRU", "platform-wiki-2024.md", "2024-03", 0.85, None)),
    (("search_index", "refresh_interval", "1s", "search-runbook-2023.md", "2023-02", 0.8, None),
     ("search_index", "refresh_interval", "1s", "search-capacity-plan-2024.pdf", "2024-07", 0.85, None)),
    (("billing_service", "invoice_currency", "USD", "billing-policy-2022.pdf", "2022-12", 0.8, None),
     ("billing_service", "invoice_currency", "USD", "billing-policy-2024.pdf", "2024-09", 0.9, None)),
    (("analytics_warehouse", "query_timeout", "300s", "data-platform-doc-2023.md", "2023-03", 0.8, None),
     ("analytics_warehouse", "query_timeout", "300s", "data-platform-doc-2024.md", "2024-06", 0.85, None)),
]

SINGLES = [
    ("payments_service", "max_retries", "3", "architecture-doc-v2.pdf", "2023-01", 0.8, None),
    ("primary_database", "engine", "PostgreSQL 15", "adr-014-db-timeouts.md", "2024-02", 0.9, None),
    ("auth_service", "hashing_algorithm", "bcrypt", "security-policy-2022.pdf", "2022-09", 0.85, None),
    ("api_gateway", "max_payload_size", "10MB", "gateway-config-2024.yaml", "2024-08", 0.95, None),
    ("order_queue", "visibility_timeout", "30s", "queue-spec-v2.pdf", "2024-01", 0.9, None),
    ("notification_service", "batch_size", "500", "vendor-migration-adr-2024.md", "2024-05", 0.85, None),
    ("file_storage", "encryption_standard", "AES-256", "storage-ops-2023.md", "2023-07", 0.9, None),
    ("logging_pipeline", "log_format", "JSON", "observability-guide-2022.md", "2022-10", 0.8, None),
]


def _mk(row) -> ProvenanceFact:
    e, a, v, s, t, c, txt = row
    return ProvenanceFact(entity=e, attribute=a, value=v, source=s, timestamp=t,
                          confidence=c, context_text=txt or _text(e, a, v))


def generate_synthetic_corpus() -> List[Dict[str, Any]]:
    rows: List[tuple] = []
    for pair in CONFLICTS + AGREES:
        rows.extend(pair)
    rows.extend(SINGLES)
    facts = [_mk(r).model_dump() for r in rows]
    keys = {(f["entity"], f["attribute"], f["value"], f["source"], f["timestamp"]) for f in facts}
    assert len(keys) == len(facts), "duplicate facts would collapse in the graph"
    return facts


def summarize(facts: List[Dict[str, Any]]) -> Dict[str, int]:
    groups: Dict[tuple, set] = {}
    for f in facts:
        groups.setdefault((f["entity"], f["attribute"]), set()).add(f["value"].strip().lower())
    return {
        "facts": len(facts),
        "entities": len({f["entity"] for f in facts}),
        "conflict_groups": sum(1 for v in groups.values() if len(v) > 1),
    }


def write_dataset(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    facts = generate_synthetic_corpus()
    for out in ([path] if path else [ROOT / "data" / "dataset.json", ROOT / "dataset.json"]):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(facts, indent=2), encoding="utf-8")
    return facts


if __name__ == "__main__":
    data = write_dataset()
    print("Wrote data/dataset.json and dataset.json:", summarize(data))
    sys.exit(0)
