import logging

from app.config import get_settings
from app.ingestion.vector_builder import VectorStore
from app.utils import llm

log = logging.getLogger(__name__)

PROMPT = "Answer the following question based on the context.\nContext: {ctx}\nQuestion: {q}"


def baseline_rag_query(question: str, store: VectorStore) -> dict:
    """Plain top-K vector RAG. No provenance, no conflict awareness (this is the control group)."""
    s = get_settings()
    hits = store.search(question, s.top_k)
    log.info("baseline retrieved: %s", hits)
    if llm.llm_enabled():
        ctx = "\n".join(f"- {t}" for t, _ in hits)
        return {"answer": llm.chat(PROMPT.format(ctx=ctx, q=question))}
    # OFFLINE generator: extractive, answers from the best-ranked chunk - mimics the
    # "whichever chunk ranks higher wins" failure mode without needing an API key.
    return {"answer": hits[0][0]}
