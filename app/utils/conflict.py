"""Deterministic conflict detection + resolution heuristic. NO LLM involved here.

agreement_status is the absolute structural state of the graph; the resolution heuristic is a
separate, secondary signal that never hides the conflict.
"""
from typing import Any, Dict, List, Optional

Fact = Dict[str, Any]


def normalize(value: Any) -> str:
    return " ".join(str(value).strip().lower().split())


def compute_status(facts: List[Fact]) -> Optional[str]:
    """None = no facts. Otherwise 'conflict' | 'single-source' | 'agree'."""
    if not facts:
        return None
    if len({normalize(f["value"]) for f in facts}) > 1:
        return "conflict"
    if len(facts) == 1:
        return "single-source"
    return "agree"


def rank_facts(facts: List[Fact]) -> List[Fact]:
    """Resolution heuristic ordering: newest timestamp first, confidence breaks ties."""
    return sorted(facts, key=lambda f: (str(f["timestamp"]), float(f.get("confidence") or 0)), reverse=True)


def preferred_fact(facts: List[Fact]) -> Fact:
    return rank_facts(facts)[0]


def chronological(facts: List[Fact]) -> List[Fact]:
    return sorted(facts, key=lambda f: (str(f["timestamp"]), str(f["source"])))


def build_answer(status: str, facts: List[Fact]) -> str:
    """Deterministic answer text (also used as validated fallback for the LLM phrasing)."""
    facts = chronological(facts)
    if status == "single-source":
        f = facts[0]
        return f"Only one source covers this: {f['source']} ({f['timestamp']}) states {f['value']}."
    if status == "agree":
        srcs = ", ".join(f"{f['source']} ({f['timestamp']})" for f in facts)
        return f"All {len(facts)} sources agree: the value is {facts[0]['value']} (per {srcs})."
    best = preferred_fact(facts)
    parts = "; ".join(f"{f['source']} ({f['timestamp']}) states {f['value']}" for f in facts)
    return (
        f"Sources disagree: {parts}. "
        f"By the resolution heuristic (most recent timestamp, confidence as tie-break), the most recent "
        f"source {best['source']} ({best['timestamp']}, confidence {best['confidence']}) suggests "
        f"{best['value']}, but this should be verified."
    )


CONFLICT_WORDS = ("disagree", "conflict", "differ", "contradict")
HEURISTIC_WORDS = ("newer", "recent", "timestamp", "confidence", "prioritiz")


def answer_is_compliant(status: str, answer: str, facts: List[Fact]) -> bool:
    """Guard around LLM phrasing: accept only if it states the conflict + heuristic and mentions every value."""
    a = answer.lower()
    if status != "conflict":
        return bool(answer.strip())
    return (
        any(w in a for w in CONFLICT_WORDS)
        and any(w in a for w in HEURISTIC_WORDS)
        and all(normalize(f["value"]) in a for f in facts)
    )
