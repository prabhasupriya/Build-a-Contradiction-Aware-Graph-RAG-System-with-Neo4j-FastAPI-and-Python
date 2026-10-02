import os

import pytest
from fastapi.testclient import TestClient

from app.api.graph_rag import conflict_aware_query
from app.utils import conflict as C
from app.utils.resolver import resolve_target


def F(v, s="a.md", t="2024-01", c=0.5):
    return {"value": v, "source": s, "timestamp": t, "confidence": c}


def test_status_none_single_agree_conflict():
    assert C.compute_status([]) is None
    assert C.compute_status([F("30s")]) == "single-source"
    assert C.compute_status([F("30s"), F("30s", "b.md")]) == "agree"
    assert C.compute_status([F("30s"), F("60s", "b.md")]) == "conflict"


def test_status_normalizes_case_and_whitespace():
    assert C.compute_status([F("LRU"), F(" lru ", "b.md")]) == "agree"


def test_heuristic_newest_then_confidence():
    facts = [F("30s", t="2023-01", c=0.99), F("60s", "b.md", "2024-06", 0.1)]
    assert C.preferred_fact(facts)["value"] == "60s"
    tie = [F("a", t="2024-01", c=0.2), F("b", "b.md", "2024-01", 0.9)]
    assert C.preferred_fact(tie)["value"] == "b"


def test_conflict_answer_has_conflict_and_heuristic_language():
    facts = [F("30s", "x.pdf", "2023-01", 0.8), F("60s", "y.md", "2024-06", 0.9)]
    ans = C.build_answer("conflict", facts).lower()
    assert "disagree" in ans and "recent" in ans and "timestamp" in ans
    assert "30s" in ans and "60s" in ans
    assert C.answer_is_compliant("conflict", ans, facts)
    assert not C.answer_is_compliant("conflict", "The timeout is 60s.", facts)


def test_dataset_contract(facts):
    assert len(facts) >= 30 and len({f["entity"] for f in facts}) >= 10
    groups = {}
    for f in facts:
        groups.setdefault((f["entity"], f["attribute"]), set()).add(f["value"])
    assert sum(len(v) > 1 for v in groups.values()) >= 8
    assert set(facts[0]) == {"entity", "attribute", "value", "source", "timestamp", "confidence", "context_text"}


def test_resolver(graph):
    cat = graph.catalog()
    assert resolve_target("What is the API timeout for the payments service?", cat) == ("payments_service", "api_timeout")
    assert resolve_target("What is the database connection timeout?", cat) == ("primary_database", "connection_timeout")
    assert resolve_target("What is the quantum flux capacity of the hyperdrive server?", cat) is None


def test_graph_pipeline_conflict_single_agree_missing(graph, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    r = conflict_aware_query("What is the API timeout for the payments service?", graph)
    assert r.agreement_status == "conflict" and len(r.sources) == 2
    assert [s.value for s in r.sources] == ["30s", "60s"]
    r = conflict_aware_query("What is the max retries for the payments service?", graph)
    assert r.agreement_status == "single-source" and len(r.sources) == 1
    r = conflict_aware_query("What is the eviction policy of the cache cluster?", graph)
    assert r.agreement_status == "agree"
    assert conflict_aware_query("What is the quantum flux capacity of the hyperdrive server?", graph) is None


def test_http_contract(graph, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    from app import main
    from app.ingestion.vector_builder import build_from_dataset
    from app.config import get_settings
    main.state["graph"] = graph
    main.state["store"] = build_from_dataset(get_settings().dataset_path)
    c = TestClient(main.app)  # no lifespan: neo4j not needed
    assert c.get("/health").json() == {"status": "ok"}
    r = c.post("/api/ask", json={"question": "What is the API timeout for the payments service?"})
    assert r.status_code == 200 and r.json()["agreement_status"] == "conflict"
    r = c.post("/api/ask", json={"question": "What is the quantum flux capacity of the hyperdrive server?"})
    assert r.status_code == 404
    r = c.post("/api/baseline", json={"question": "What is the database connection timeout?"})
    assert r.status_code == 200 and isinstance(r.json()["answer"], str)
