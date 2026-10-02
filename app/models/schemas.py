from typing import List, Literal

from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1)


class BaselineResponse(BaseModel):
    answer: str


class SourceFact(BaseModel):
    value: str
    source: str
    timestamp: str


class GraphRagResponse(BaseModel):
    answer: str = Field(
        ...,
        description="Human-readable explanation. MUST explicitly say sources disagree when status is 'conflict', "
        "and name which value the resolution heuristic (newest timestamp) prefers.",
    )
    agreement_status: Literal["agree", "conflict", "single-source"]
    sources: List[SourceFact]


class NotFoundResponse(BaseModel):
    error: str = "no_facts_found"
    message: str
    answer: str
    sources: List[SourceFact] = []
