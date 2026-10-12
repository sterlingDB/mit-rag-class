"""A small persistent answer cache using the project's existing Chroma dependency."""
import hashlib
import json
import time

import chromadb


def cache_namespace(documents: dict[str, str], configuration: dict) -> str:
    """Changing article content or prompt/model settings invalidates old answers."""
    digest = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode())
    for article_id, content in sorted(documents.items()):
        digest.update(json.dumps([article_id, content]).encode())
    return digest.hexdigest()[:24]


class SemanticAnswerCache:
    def __init__(self, path, embedder, namespace, max_size=400, ttl_seconds=86400,
                 similarity_threshold=0.85):
        if max_size < 1 or ttl_seconds <= 0:
            raise ValueError("Cache size and lifetime must be positive")
        self.client = chromadb.PersistentClient(path=str(path))
        self.collection = self.client.get_or_create_collection(
            "answers_" + namespace, metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )
        self.embedder = embedder
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.threshold = similarity_threshold
        self.embedding_calls = 0
        self.embedding_seconds = 0.0

    @staticmethod
    def key(question):
        normalized = " ".join(question.split()).casefold()
        return hashlib.sha256(normalized.encode()).hexdigest()

    def prune(self):
        records = self.collection.get(include=["metadatas"])
        expired = [key for key, meta in zip(records["ids"], records["metadatas"])
                   if time.time() - meta["timestamp"] >= self.ttl_seconds]
        if expired:
            self.collection.delete(ids=expired)

    def embed(self, question):
        self.embedding_calls += 1
        started = time.perf_counter()
        try:
            return self.embedder.embed_query(question)
        finally:
            self.embedding_seconds += time.perf_counter() - started

    def lookup(self, question):
        self.prune()
        exact = self.collection.get(ids=[self.key(question)], include=["metadatas"])
        if exact["ids"]:
            return json.loads(exact["metadatas"][0]["payload"]), "exact"
        if not self.collection.count():
            return None
        nearest = self.collection.query(query_embeddings=[self.embed(question)], n_results=1,
                                        include=["metadatas", "distances"])
        if nearest["ids"][0] and 1 - nearest["distances"][0][0] >= self.threshold:
            return json.loads(nearest["metadatas"][0][0]["payload"]), "semantic"
        return None

    def store(self, question, result):
        self.prune()
        payload = {key: result[key] for key in ("answer", "sources", "evidence")}
        payload["question"] = question
        vector = self.embed(question)
        self.collection.upsert(ids=[self.key(question)], embeddings=[vector], metadatas=[{
            "timestamp": time.time(), "payload": json.dumps(payload),
        }])
        records = self.collection.get(include=["metadatas"])
        oldest = sorted(zip(records["ids"], records["metadatas"]), key=lambda item: item[1]["timestamp"])
        excess = len(oldest) - self.max_size
        if excess > 0:
            self.collection.delete(ids=[key for key, _ in oldest[:excess]])
