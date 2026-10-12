"""One-hop retrieval over links in saved Wikipedia HTML.

Reuse in a checkpoint:
    graph = GraphRetriever(retriever)
    hits = graph.getTopK(question, 3)

Graph construction and neighbor scoring are local. Initial hybrid search uses
the existing retriever's embedding API. Article text always comes from its index.
"""

from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import unquote, urlsplit

import networkx as nx

from wiki_helpers import WIKIPEDIA_DIR, article_html_only

if TYPE_CHECKING:
    from hybrid_retriever import HybridRetriever


def article_id_from_link(href: str) -> str | None:
    """Convert an internal article URL to the source ID used in Chroma."""
    url = urlsplit(href)
    if url.scheme not in {"", "http", "https"}:
        return None
    if url.netloc and url.netloc.lower() != "en.wikipedia.org":
        return None
    path = url.path
    if path.startswith("/wiki/"):
        title = path[len("/wiki/"):]
    elif not url.netloc and path.startswith("./"):
        title = path[2:]
    elif not url.netloc and not path.startswith("/") and path.endswith(".html"):
        title = path
    else:
        return None
    title = unquote(title).removesuffix(".html").replace(" ", "_")
    if not title or ":" in title:
        return None
    return title


class ArticleLinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.article_ids: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                article_id = article_id_from_link(href)
                if article_id:
                    self.article_ids.add(article_id)


def build_graph(html_dir: Path, doc_ids: set[str]) -> nx.DiGraph:
    """Keep only links between indexed articles; never fetch missing pages."""
    if not html_dir.is_dir():
        raise FileNotFoundError(f"Wikipedia HTML directory not found: {html_dir}")
    graph = nx.DiGraph()
    graph.add_nodes_from(sorted(doc_ids), node_type="article")
    for path in sorted(html_dir.glob("*.html")):
        if path.stem not in doc_ids:
            continue
        parser = ArticleLinkParser()
        parser.feed(article_html_only(path.read_text(encoding="utf-8", errors="replace")))
        for target in sorted(parser.article_ids & doc_ids):
            if target != path.stem:
                graph.add_edge(path.stem, target, edge_type="links_to")
    return graph


class GraphRetriever:
    def __init__(self, retriever: HybridRetriever, html_dir: Path = WIKIPEDIA_DIR):
        self.retriever = retriever
        self.documents = retriever.get_documents()
        self.graph = build_graph(Path(html_dir), set(self.documents))

    def getTopK(
        self,
        query: str,
        k: int = 3,
        seed_hits: list[tuple[str, str, float, str]] | None = None,
    ) -> list[tuple[str, str, float, str]]:
        """Keep up to two primary hits, then add one-hop context within k.

        Pass baseline or decomposition hits to reuse an existing search. Scores
        on context hits are raw BM25, not comparable to primary hybrid scores.
        """
        if k <= 0:
            return []
        if seed_hits is None:
            seed_hits = self.retriever.getTopK(query, k)
        unique = {}
        for hit in seed_hits:
            if hit[0] in self.documents:
                unique.setdefault(hit[0], hit)
        seeds = list(unique.values())[:min(2, k)]
        hits = [(doc_id, text, score, f"primary: {method}") for doc_id, text, score, method in seeds]
        selected = {hit[0] for hit in hits}
        linked_from: dict[str, list[str]] = {}
        for seed, _, _, _ in seeds:
            for target in self.graph.successors(seed):
                if target not in selected:
                    linked_from.setdefault(target, []).append(seed)

        scores = self.retriever.score_documents(query, list(linked_from)) if linked_from else {}
        ranked = sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))
        for doc_id in ranked:
            if len(hits) >= k:
                break
            if scores[doc_id] <= 0:
                continue
            sources = ", ".join(linked_from[doc_id])
            hits.append((doc_id, self.documents[doc_id], scores[doc_id], f"context: linked from {sources}; BM25"))
            selected.add(doc_id)

        # A disconnected seed should still be usable as ordinary hybrid search.
        for doc_id, text, score, method in unique.values():
            if len(hits) >= k:
                break
            if doc_id not in selected:
                hits.append((doc_id, text, score, f"primary: {method}"))
                selected.add(doc_id)
        return hits
