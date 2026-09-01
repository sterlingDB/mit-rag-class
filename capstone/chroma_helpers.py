import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from wiki_helpers import html_to_text


def get_embeddings(api_key: str, embedding_model: str, base_url: str) -> OpenAIEmbeddings:
    """Create the embedding model Chroma uses for vector search."""
    return OpenAIEmbeddings(
        model=embedding_model,
        api_key=api_key,
        base_url=base_url,
    )


def build_or_load_db(
    docs_dir: str | Path,
    chroma_dir: str,
    api_key: str,
    embedding_model: str,
    base_url: str,
) -> Chroma:
    """Load an existing Chroma DB, or build one from local .txt/.html files."""
    if os.path.isdir(chroma_dir) and os.listdir(chroma_dir):
        print(f"Loading existing vector DB from {chroma_dir}/")
        return Chroma(
            persist_directory=chroma_dir,
            embedding_function=get_embeddings(api_key, embedding_model, base_url),
        )

    print("Building vector DB (first run - embedding the documents)...")
    docs = []
    docs_dir = Path(docs_dir)

    for path in sorted(docs_dir.iterdir()):
        if path.suffix not in {".txt", ".html"}:
            continue

        text = path.read_text(encoding="utf-8", errors="replace")
        if path.suffix == ".html":
            text = html_to_text(text)

        docs.append(Document(page_content=text, metadata={"source": path.name}))

    db = Chroma.from_documents(
        docs,
        get_embeddings(api_key, embedding_model, base_url),
        persist_directory=chroma_dir,
    )
    print(f"  Indexed {len(docs)} documents into {chroma_dir}/")
    return db
