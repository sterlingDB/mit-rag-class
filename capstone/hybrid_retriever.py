from __future__ import annotations

import re
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from rank_bm25 import BM25Okapi
from chroma_helpers import API_KEY, OPENROUTER_BASE_URL, get_db

if TYPE_CHECKING:
    from langchain_chroma import Chroma

LLM_MODEL = "openai/gpt-5.4-mini"
NUM_RETRIEVED = 4
CANDIDATE_POOL = 10
WEIGHT_BM25 = 0.5
WEIGHT_VECTOR = 0.5

SYSTEM_PROMPT = """You are a helpful assistant. Answer the question using ONLY the
provided documents, and quote from them where you can. If the documents do not
contain the answer, say so rather than guessing."""


_STOPWORDS = {
    "a", "an", "the", "and", "but", "or", "nor", "so", "yet", "for",
    "in", "on", "at", "to", "of", "by", "with", "from", "into", "onto", "upon",
    "about", "above", "below", "between", "through", "during", "before", "after",
    "under", "over", "around", "along", "across", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "i", "we", "you", "he", "she", "it", "they", "me", "us", "him", "her", "them",
    "my", "our", "your", "his", "its", "their", "this", "that", "these", "those",
    "as", "if", "up", "out", "not", "no",
}


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS]


def _normalize(scores: list[float], invert: bool = False) -> list[float]:
    """Scale scores to [0, 1] so BM25 and vector results can be combined."""
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [0.5] * len(scores)
    norm = [(s - lo) / (hi - lo) for s in scores]
    return [1.0 - n for n in norm] if invert else norm


def bm25_topk(
    bm25: BM25Okapi,
    paths: list[str],
    contents: list[str],
    query: str,
    k: int,
) -> list[tuple[str, str, float]]:
    scores = bm25.get_scores(tokenize(query))
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [(paths[i], contents[i], float(scores[i])) for i in top]


def vector_topk(db: Chroma, query: str, k: int) -> list[tuple[str, str, float]]:
    results = db.similarity_search_with_score(query, k=k)
    return [(d.metadata.get("source", "unknown"), d.page_content, s) for d, s in results]


def load_documents_from_db(db: Chroma) -> tuple[list[str], list[str]]:
    records = db.get(include=["documents", "metadatas"])
    ids = records.get("ids") or []
    documents = records.get("documents") or []
    metadatas = records.get("metadatas") or []

    paths = []
    contents = []
    for index, content in enumerate(documents):
        metadata = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
        fallback_id = ids[index] if index < len(ids) else f"doc{index + 1}"
        paths.append(metadata.get("source", fallback_id))
        contents.append(content)
    return paths, contents


def retrieval_source(bm25_score: float, vector_score: float) -> str:
    if bm25_score > 0 and vector_score > 0:
        return "bm25+vector"
    if bm25_score > 0:
        return "bm25"
    return "vector"


class HybridRetriever:
    def __init__(self, db: Chroma | None = None, num_retrieved: int = NUM_RETRIEVED):
        if db is None:
            db = get_db()

        self._llm = ChatOpenAI(
            model=LLM_MODEL,
            api_key=API_KEY,
            base_url=OPENROUTER_BASE_URL,
        )
        self._num_retrieved = num_retrieved
        self._db = db
        self._paths, self._contents = load_documents_from_db(db)
        self._bm25 = BM25Okapi([tokenize(doc) for doc in self._contents])
        print(f"Hybrid retriever ready over {len(self._paths)} documents.")

    def _build_user_message(self, query: str, context: str) -> str:
        return f"Context (documents):\n{context}\n\nQuestion: {query}"

    def getTopK(self, query: str, k: int) -> list[tuple[str, str, float, str]]:
        """Fuse BM25 keyword results with Chroma vector results."""
        bm = bm25_topk(self._bm25, self._paths, self._contents, query, CANDIDATE_POOL)
        vec = vector_topk(self._db, query, CANDIDATE_POOL)

        content_by_name: dict[str, str] = {}
        bm_norm: dict[str, float] = {}
        vec_norm: dict[str, float] = {}

        if bm:
            for (name, content, _), val in zip(bm, _normalize([s for _, _, s in bm])):
                content_by_name[name] = content
                bm_norm[name] = val
        if vec:
            for (name, content, _), val in zip(vec, _normalize([d for _, _, d in vec], invert=True)):
                content_by_name[name] = content
                vec_norm[name] = val

        fused = [
            (
                name,
                content,
                WEIGHT_BM25 * bm_norm.get(name, 0.0)
                + WEIGHT_VECTOR * vec_norm.get(name, 0.0),
                retrieval_source(bm_norm.get(name, 0.0), vec_norm.get(name, 0.0)),
            )
            for name, content in content_by_name.items()
        ]
        fused.sort(key=lambda t: t[2], reverse=True)
        return fused[:k]

    def retrievedContext(self, query: str) -> str:
        results = self.getTopK(query, self._num_retrieved)
        return "\n\n---\n\n".join(f"[{name}]\n{content}" for name, content, _, _ in results)

    def query(self, question: str) -> str:
        context = self.retrievedContext(question)
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=self._build_user_message(question, context)),
        ]
        response = self._llm.invoke(messages)
        return response.content if hasattr(response, "content") else str(response)
