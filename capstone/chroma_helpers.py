import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
CHROMA_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "sample_docs"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
EMBEDDING_MODEL = "openai/text-embedding-3-small"


def get_embeddings() -> OpenAIEmbeddings:
    """Create the embedding model Chroma uses for vector search."""
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )


def load_db() -> Chroma:
    """Load the existing Chroma DB."""
    print(f"Loading existing vector DB from {CHROMA_DIR}/")
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=get_embeddings(),
    )


def build_db_from_docs() -> Chroma:
    """Build a Chroma DB from SAMPLE_DOCS-style dictionaries."""
    from sample_docs import SAMPLE_DOCS

    embeddings = get_embeddings()

    docs = [
        Document(page_content=doc["text"], metadata={"source": doc["id"]})
        for doc in SAMPLE_DOCS
    ]
    ids = [doc["id"] for doc in SAMPLE_DOCS]

    print("Building vector DB from SAMPLE_DOCS (first run - embedding the documents)...")
    db = Chroma.from_documents(
        docs,
        embeddings,
        collection_name=COLLECTION_NAME,
        ids=ids,
        persist_directory=str(CHROMA_DIR),
    )
    print(f"  Indexed {len(docs)} sample documents into {CHROMA_DIR}/")
    return db


def get_db() -> Chroma:
    return load_db()
