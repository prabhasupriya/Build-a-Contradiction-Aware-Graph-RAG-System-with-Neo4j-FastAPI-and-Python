"""Deterministic (entity, attribute) resolution against the graph's catalog.

Scores every (entity, attribute) pair by the fraction of its name tokens present in the question.
Requires both sides to match >= 50%, picks the unique best pair, and otherwise returns None
(=> graceful 404; we never guess).
"""
from typing import List, Optional, Sequence, Tuple

from app.utils.text import content_tokens

Pair = Tuple[str, str]


def _frac(name: str, q: set) -> float:
    toks = set(content_tokens(name.replace("_", " ")))
    return len(toks & q) / len(toks) if toks else 0.0


def resolve_target(question: str, catalog: Sequence[Pair]) -> Optional[Pair]:
    q = set(content_tokens(question))
    scored: List[Tuple[float, int, Pair]] = []
    for entity, attribute in catalog:
        e, a = _frac(entity, q), _frac(attribute, q)
        if e >= 0.5 and a >= 0.5:
            matched = len(set(content_tokens(entity.replace("_", " "))) & q) + \
                      len(set(content_tokens(attribute.replace("_", " "))) & q)
            scored.append((e + a, matched, (entity, attribute)))
    if not scored:
        return None
    scored.sort(reverse=True)
    if len(scored) > 1 and scored[0][:2] == scored[1][:2]:
        return None  # ambiguous
    return scored[0][2]
