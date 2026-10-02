import hashlib
import logging
import time
from typing import Any, Dict, List, Tuple

from neo4j import GraphDatabase

from app.utils import cypher_queries as Q

log = logging.getLogger(__name__)


def fact_id(f: Dict[str, Any]) -> str:
    raw = "|".join(str(f[k]) for k in ("entity", "attribute", "value", "source", "timestamp"))
    return hashlib.sha1(raw.encode()).hexdigest()


class KnowledgeGraphBuilder:
    """Schema: (Entity)-[:HAS_ATTRIBUTE]->(Attribute)-[:STATED_BY]->(Fact{value,source,timestamp,confidence})."""

    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def wait_until_ready(self, timeout: int = 90) -> None:
        deadline = time.time() + timeout
        while True:
            try:
                self.driver.verify_connectivity()
                return
            except Exception as exc:  # noqa: BLE001
                if time.time() > deadline:
                    raise
                log.info("waiting for neo4j: %s", exc)
                time.sleep(2)

    def ingest_facts(self, facts: List[dict], reset: bool = True) -> int:
        with self.driver.session() as s:
            for c in Q.CONSTRAINTS:
                s.run(c)
            if reset:
                s.run(Q.CLEAR_GRAPH)
            for f in facts:  # MERGE keyed on deterministic fact_id => idempotent re-runs
                s.run(
                    Q.INGEST_FACT,
                    entity=f["entity"], attribute=f["attribute"],
                    attr_key=f"{f['entity']}::{f['attribute']}", fact_id=fact_id(f),
                    value=f["value"], source=f["source"], timestamp=f["timestamp"],
                    confidence=float(f["confidence"]), context_text=f["context_text"],
                )
        return self.count_facts()

    def count_facts(self) -> int:
        with self.driver.session() as s:
            return s.run(Q.COUNT_FACTS).single()["fact_count"]

    def fetch_facts(self, entity: str, attribute: str) -> List[Dict[str, Any]]:
        with self.driver.session() as s:
            return [dict(r) for r in s.run(Q.FETCH_FACTS, entity=entity, attribute=attribute)]

    def catalog(self) -> List[Tuple[str, str]]:
        with self.driver.session() as s:
            return [(r["entity"], r["attribute"]) for r in s.run(Q.CATALOG)]

    def close(self):
        self.driver.close()
