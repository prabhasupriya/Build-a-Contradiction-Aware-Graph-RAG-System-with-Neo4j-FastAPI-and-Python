import logging
from typing import Optional, Protocol, Sequence, Tuple, List, Dict, Any

from app.models.schemas import GraphRagResponse, SourceFact
from app.utils import conflict as C
from app.utils import llm
from app.utils.resolver import resolve_target

log = logging.getLogger(__name__)


class GraphLike(Protocol):
    def catalog(self) -> Sequence[Tuple[str, str]]: ...
    def fetch_facts(self, entity: str, attribute: str) -> List[Dict[str, Any]]: ...


SYSTEM = (
    "You phrase answers about knowledge-base facts. The agreement_status and sources are computed by code and "
    "are final. If status is 'conflict' the answer MUST say the sources disagree, cite each source/value/timestamp, "
    "and then state which value the resolution heuristic prefers (newest timestamp, confidence as tie-break) "
    "while saying it should be verified. Never hide the conflict."
)


def _resolve(question: str, graph: GraphLike) -> Optional[Tuple[str, str]]:
    catalog = list(graph.catalog())
    target = resolve_target(question, catalog)
    if target or not llm.llm_enabled():
        return target
    try:  # LLM fallback, constrained to catalog entries (code validates the result)
        from pydantic import BaseModel

        class Target(BaseModel):
            entity: str
            attribute: str

        listing = "\n".join(f"{e} / {a}" for e, a in catalog)
        t = llm.structured(Target, "Pick the single best (entity, attribute) from the list, or the closest "
                           "match. Reply with exact names.", f"Question: {question}\nCatalog:\n{listing}")
        return (t.entity, t.attribute) if (t.entity, t.attribute) in set(catalog) else None
    except Exception as exc:  # noqa: BLE001
        log.warning("LLM target extraction failed: %s", exc)
        return None


def conflict_aware_query(question: str, graph: GraphLike) -> Optional[GraphRagResponse]:
    """Returns None when no entity/attribute/facts can be found (caller maps to 404)."""
    target = _resolve(question, graph)
    if not target:
        return None
    entity, attribute = target
    facts = graph.fetch_facts(entity, attribute)          # ALL facts, not top-1
    status = C.compute_status(facts)                       # deterministic detector
    log.info("entity=%s attribute=%s n=%d status=%s", entity, attribute, len(facts), status)
    if status is None:
        return None
    ordered = C.chronological(facts)
    sources = [SourceFact(value=f["value"], source=f["source"], timestamp=str(f["timestamp"])) for f in ordered]
    answer = C.build_answer(status, facts)                 # deterministic fallback text
    if llm.llm_enabled():
        try:
            user = (f"Question: {question}\nentity={entity} attribute={attribute}\nstatus={status}\n"
                    f"Facts sorted by resolution heuristic (newest first): {C.rank_facts(facts)}")
            out = llm.structured(GraphRagResponse, SYSTEM, user)
            if C.answer_is_compliant(status, out.answer, facts):
                answer = out.answer
            else:
                log.warning("LLM answer failed compliance check; using deterministic text")
        except Exception as exc:  # noqa: BLE001
            log.warning("LLM phrasing failed (%s); using deterministic text", exc)
    # status + sources always come from code, never from the LLM
    return GraphRagResponse(answer=answer, agreement_status=status, sources=sources)
