import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


class FakeGraph:
    """In-memory stand-in for KnowledgeGraphBuilder (same fetch_facts/catalog contract)."""
    def __init__(self, facts):
        self.facts = facts

    def catalog(self):
        return sorted({(f["entity"], f["attribute"]) for f in self.facts})

    def fetch_facts(self, entity, attribute):
        return [dict(value=f["value"], source=f["source"], timestamp=f["timestamp"], confidence=f["confidence"])
                for f in self.facts if f["entity"] == entity and f["attribute"] == attribute]


@pytest.fixture(scope="session")
def facts():
    from generate_dataset import generate_synthetic_corpus
    return generate_synthetic_corpus()


@pytest.fixture()
def graph(facts):
    return FakeGraph(facts)
