import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.api.baseline_rag import baseline_rag_query
from app.api.graph_rag import conflict_aware_query
from app.config import get_settings
from app.ingestion.graph_builder import KnowledgeGraphBuilder
from app.ingestion.vector_builder import build_from_dataset
from app.models.schemas import BaselineResponse, GraphRagResponse, NotFoundResponse, QuestionRequest

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("app")
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    state["store"] = build_from_dataset(s.dataset_path)
    graph = KnowledgeGraphBuilder(s.neo4j_uri, s.neo4j_user, s.neo4j_password)
    try:
        graph.wait_until_ready()
        if s.auto_ingest and graph.count_facts() == 0:
            facts = json.loads(s.dataset_path.read_text(encoding="utf-8"))
            log.info("graph empty -> ingested %d facts", graph.ingest_facts(facts))
    except Exception as exc:  # noqa: BLE001  (keep /health alive; endpoints report errors)
        log.error("neo4j init failed: %s", exc)
    state["graph"] = graph
    yield
    graph.close()


app = FastAPI(title="Conflict-Aware Graph RAG", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/baseline", response_model=BaselineResponse)
def baseline(req: QuestionRequest):
    return baseline_rag_query(req.question, state["store"])


@app.post("/api/ask", response_model=GraphRagResponse,
          responses={404: {"model": NotFoundResponse}})
def ask(req: QuestionRequest):
    result = conflict_aware_query(req.question, state["graph"])
    if result is None:
        msg = "I couldn't find any facts for that entity/attribute in the knowledge graph."
        return JSONResponse(status_code=404, content=NotFoundResponse(
            message=msg, answer=msg).model_dump())
    return result
