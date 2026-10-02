"""Thin OpenAI/instructor wrapper. When no real key is configured the stack runs OFFLINE (deterministic)."""
from typing import List, Optional, Type

from pydantic import BaseModel

from app.config import get_settings


def llm_enabled() -> bool:
    return get_settings().llm_enabled


def _openai():
    from openai import OpenAI
    s = get_settings()
    kw = {"api_key": s.openai_api_key}
    if s.openai_base_url:
        kw["base_url"] = s.openai_base_url
    return OpenAI(**kw)


def chat(prompt: str) -> str:
    s = get_settings()
    r = _openai().chat.completions.create(
        model=s.llm_model_id, temperature=0, messages=[{"role": "user", "content": prompt}])
    return r.choices[0].message.content or ""


def structured(model: Type[BaseModel], system: str, user: str) -> BaseModel:
    import instructor
    s = get_settings()
    client = instructor.from_openai(_openai())
    return client.chat.completions.create(
        model=s.llm_model_id, temperature=0, response_model=model, max_retries=2,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])


def embed_openai(texts: List[str]) -> List[List[float]]:
    s = get_settings()
    r = _openai().embeddings.create(model=s.embedding_model_id, input=texts)
    return [d.embedding for d in r.data]
