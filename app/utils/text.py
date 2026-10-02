"""Tiny shared tokenizer (stemming + stopwords) used by the resolver and the local embedder."""
import re
from typing import List

STOPWORDS = {
    "what", "is", "the", "a", "an", "of", "for", "to", "in", "on", "and", "or", "does", "do", "are",
    "how", "which", "at", "by", "with", "our", "s", "current", "currently", "configured", "set", "value",
    "tell", "me", "us", "use", "uses", "used", "its", "it", "this", "that", "be", "there", "long",
}
SYNONYMS = {"db": "database", "logging": "log", "logs": "log", "retries": "retry", "authentication": "auth"}


def stem(tok: str) -> str:
    if tok in SYNONYMS:
        return SYNONYMS[tok]
    if tok.endswith("ies") and len(tok) > 4:
        return tok[:-3] + "y"
    if tok.endswith("s") and not tok.endswith("ss") and len(tok) > 3:
        return tok[:-1]
    return tok


def content_tokens(text: str) -> List[str]:
    toks = re.findall(r"[a-z0-9]+", text.lower())
    return [stem(t) for t in toks if t not in STOPWORDS]
