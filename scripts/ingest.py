"""Load data/dataset.json into Neo4j (idempotent).
Usage: docker-compose exec app python scripts/ingest.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.ingestion.graph_builder import KnowledgeGraphBuilder  # noqa: E402

if __name__ == "__main__":
    s = get_settings()
    facts = json.loads(s.dataset_path.read_text(encoding="utf-8"))
    g = KnowledgeGraphBuilder(s.neo4j_uri, s.neo4j_user, s.neo4j_password)
    g.wait_until_ready()
    n = g.ingest_facts(facts)
    print(f"Ingested. Fact nodes in graph: {n} (dataset has {len(facts)})")
    assert n == len(facts)
    g.close()
