r"""Lab 4.2 (SOLUTION) — graph-augmented retrieval over the email database.

Run:  python lab_4_2_graph_retrieval_solution.py [annotatedEmails.json]

Some answers depend on RELATIONSHIPS between e-mails — the rest of a conversation
thread, or other e-mails on the same topic, that a keyword or vector search
won't surface on their own. This lab builds a knowledge graph of the e-mails
(NetworkX) and uses it to enrich retrieval:

  hybrid search finds 5 seed e-mails
    -> for each seed, follow graph edges to add its THREAD neighbors
       and the earliest e-mails on each of its TOPICS
    -> deduplicate, then give the LLM a context that clearly separates the
       main retrieved e-mails from the thread/topic context.

The graph is built from annotatedEmails.json, which the faculty provides. Each
record already carries sender/recipients/mentions/threadID/topics (extracting
those with an LLM over thousands of e-mails is slow and is done for you), plus
the raw e-mail text — so this whole lab runs from that one file.

Setup
-----
1. Create the environment (one-time). Either use conda:
       conda env create -f environment.yml
       conda activate ragcourse
   or a plain virtual environment + pip:
       python -m venv .venv
       #  Windows:      .venv\Scripts\activate
       #  macOS/Linux:  source .venv/bin/activate
       pip install python-dotenv langchain-openai langchain-core langchain-chroma rank-bm25 networkx
2. Add the OpenRouter API key provided for this program. Create a file
   named ".env" in this folder containing a single line:
       OPENROUTER_API_KEY=sk-or-your-key-here
   (or set it in your shell —  Windows:  setx OPENROUTER_API_KEY sk-or-...
    macOS/Linux:  export OPENROUTER_API_KEY=sk-or-...)
3. Data: annotatedEmails.json (provided by the faculty) in this folder, or pass
   its path as the first argument. The vector DB is persisted to ./chroma_db
   (delete it to rebuild); the graph is rebuilt from the JSON each run (it's fast).

Sample questions to try (over the email corpus)
-----------------------------------------------
Questions that need a whole conversation or the history of a topic:
    "How did the discussion about the government project unfold?"
        -> the seed e-mail plus its thread neighbors reconstruct the conversation.
    "Trace how the budget concerns developed over time."
        -> topic context pulls the earliest e-mails on that topic.
Compare against plain hybrid retrieval (Lab 2.2) on the SAME questions and a
comparable number of e-mails: graph context helps when relationships matter.
"""
import json
import os
import re
import sys
from abc import ABC, abstractmethod

import networkx as nx
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from rank_bm25 import BM25Okapi

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "openai/gpt-oss-120b"
EMBEDDING_MODEL = "openai/text-embedding-3-small"
CHROMA_DIR = "chroma_db"
CANDIDATE_POOL = 10
WEIGHT_BM25 = 0.5
WEIGHT_VECTOR = 0.5
HYBRID_FETCH_K = 5    # seed e-mails from hybrid search
THREAD_NEIGHBORS = 5  # closest e-mails in a seed's thread
TOPIC_EARLIEST = 5    # earliest e-mails per topic

SYSTEM_PROMPT = """You are a helpful assistant for Precision Paperclip Inc. \
You answer questions by drawing information exclusively from the company e-mails \
provided to you as context in each message.

Rules:
- If the answer can be found in the provided e-mails, answer clearly and concisely.
- If the provided e-mails do not contain enough information to answer the question, \
say so explicitly and do not speculate or use outside knowledge.
- Do not answer questions that are unrelated to the content of the provided e-mails."""

GRAPH_SYSTEM_PROMPT = SYSTEM_PROMPT + """

The context is organized into three kinds of e-mails:
- RETRIEVED E-MAIL: the main e-mails matched to the question.
- THREAD CONTEXT: other e-mails from the same conversation, for continuity.
- TOPIC CONTEXT: earlier e-mails on the same topic, to show how it developed.
Use the thread and topic context to understand relationships, but base your
answer on what the e-mails actually say."""


def require_api_key() -> None:
    """Exit early with a clear message if OPENROUTER_API_KEY is not set, instead of
    failing later with a KeyError when the model client is created."""
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit(
            "\n[setup] OPENROUTER_API_KEY is not set.\n"
            "  1. Use the OpenRouter API key provided for this program.\n"
            "  2. Create a file named '.env' in this folder with one line:\n"
            "         OPENROUTER_API_KEY=sk-or-your-key-here\n"
            "     or set it in your shell  (Windows: setx OPENROUTER_API_KEY sk-or-... ;\n"
            "     macOS/Linux: export OPENROUTER_API_KEY=sk-or-...).\n"
        )


def chat_loop(response):
    print("Chat over the email DB. Type your question; 'exit'/'quit' to stop.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break
        if not user_input:
            continue
        try:
            result = response(user_input)
        except Exception as e:
            print(f"Error: {e}")
            continue
        print(f"\nAssistant: {result}\n")


class BaseRetriever(ABC):
    def __init__(self, llm_model: str = LLM_MODEL, system_prompt: str = SYSTEM_PROMPT):
        self._llm = ChatOpenAI(
            model=llm_model,
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url=OPENROUTER_BASE_URL,
        )
        self._system_prompt = system_prompt
        self._history = [SystemMessage(content=system_prompt)]

    @abstractmethod
    def retrievedContext(self, query: str) -> str: ...

    def _build_user_message(self, query: str, context: str) -> str:
        return f"Context (e-mails):\n{context}\n\nQuestion: {query}"

    def queryWHistory(self, question: str) -> str:
        context = self.retrievedContext(question)
        self._history.append(HumanMessage(content=self._build_user_message(question, context)))
        try:
            response = self._llm.invoke(self._history)
            answer = response.content if hasattr(response, "content") else str(response)
        except Exception:
            self._history.pop()
            raise
        self._history.append(AIMessage(content=answer))
        return answer

    def chat(self) -> None:
        chat_loop(self.queryWHistory)


# ─── Data loading + hybrid stack (built from annotatedEmails.json) ───
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


def _seq_num(filename: str) -> int:
    """Trailing number in a filename like mail_01_01_14_225.txt — used to order
    e-mails within a thread and within a topic."""
    m = re.search(r"_(\d+)\.txt$", filename)
    return int(m.group(1)) if m else 0


def load_annotated(path: str) -> tuple[list[dict], dict[str, str]]:
    """Load the faculty annotatedEmails.json. Returns (emails, body_by_filename)."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    emails = data["emails"]
    body_by_filename = {e["filename"]: e.get("rawText", "") for e in emails}
    return emails, body_by_filename


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=OPENROUTER_BASE_URL,
    )


def build_or_load_db(emails: list[dict], chroma_dir: str = CHROMA_DIR) -> Chroma:
    if os.path.isdir(chroma_dir) and os.listdir(chroma_dir):
        print(f"Loading existing vector DB from {chroma_dir}/")
        return Chroma(persist_directory=chroma_dir, embedding_function=get_embeddings())
    print(f"Building vector DB (first run — embedding {len(emails)} emails)...")
    docs = [
        Document(page_content=e.get("rawText", ""), metadata={"source": e["filename"]})
        for e in emails
    ]
    db = Chroma.from_documents(docs, get_embeddings(), persist_directory=chroma_dir)
    print(f"  Indexed {len(docs)} emails into {chroma_dir}/")
    return db


def _normalize(scores: list[float], invert: bool = False) -> list[float]:
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [0.5] * len(scores)
    norm = [(s - lo) / (hi - lo) for s in scores]
    return [1.0 - n for n in norm] if invert else norm


class HybridRetriever:
    """Score-fusion of BM25 + vector, returning (filename, content, score)."""

    def __init__(self, emails: list[dict], db: Chroma):
        self._paths = [e["filename"] for e in emails]
        self._contents = [e.get("rawText", "") for e in emails]
        self._bm25 = BM25Okapi([tokenize(c) for c in self._contents])
        self._db = db

    def getTopK(self, query: str, k: int) -> list[tuple[str, str, float]]:
        scores = self._bm25.get_scores(tokenize(query))
        bm_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:CANDIDATE_POOL]
        bm = [(self._paths[i], self._contents[i], scores[i]) for i in bm_idx]
        vec = [
            (d.metadata.get("source", "unknown"), d.page_content, s)
            for d, s in self._db.similarity_search_with_score(query, k=CANDIDATE_POOL)
        ]
        content_by_name, bm_norm, vec_norm = {}, {}, {}
        for (n, c, _), v in zip(bm, _normalize([s for _, _, s in bm])):
            content_by_name[n] = c
            bm_norm[n] = v
        for (n, c, _), v in zip(vec, _normalize([d for _, _, d in vec], invert=True)):
            content_by_name[n] = c
            vec_norm[n] = v
        fused = [
            (n, c, WEIGHT_BM25 * bm_norm.get(n, 0.0) + WEIGHT_VECTOR * vec_norm.get(n, 0.0))
            for n, c in content_by_name.items()
        ]
        fused.sort(key=lambda t: t[2], reverse=True)
        return fused[:k]


# ─── Build the knowledge graph (provided infrastructure) ─────────────
def build_graph(emails: list[dict]) -> nx.DiGraph:
    """Build a directed graph with email / thread / person / topic nodes and the
    five edge types: belongs_to (email->thread), sent (person->email),
    received_by (email->person), mentions (email->person), relates_to (email->topic)."""
    G = nx.DiGraph()
    for e in emails:
        fname = e["filename"]
        email_node = f"email:{fname}"
        G.add_node(email_node, node_type="email", filename=fname, seq_num=_seq_num(fname))

        thread_id = e.get("threadID")
        if thread_id is not None:
            thread_node = f"thread:{thread_id}"
            if not G.has_node(thread_node):
                G.add_node(thread_node, node_type="thread", thread_id=thread_id)
            G.add_edge(email_node, thread_node, edge_type="belongs_to")

        sender = e.get("sender")
        if sender and sender.get("email"):
            p = f"person:{sender['email'].lower()}"
            if not G.has_node(p):
                G.add_node(p, node_type="person", email=sender["email"], name=sender.get("name"))
            G.add_edge(p, email_node, edge_type="sent")

        for person in (e.get("toLst") or []) + (e.get("ccLst") or []):
            if person and person.get("email"):
                p = f"person:{person['email'].lower()}"
                if not G.has_node(p):
                    G.add_node(p, node_type="person", email=person["email"], name=person.get("name"))
                G.add_edge(email_node, p, edge_type="received_by")

        for mentioned in e.get("mentions") or []:
            addr = mentioned.get("email") if isinstance(mentioned, dict) else mentioned
            if addr:
                p = f"person:{addr.lower()}"
                if not G.has_node(p):
                    G.add_node(p, node_type="person", email=addr,
                               name=mentioned.get("name") if isinstance(mentioned, dict) else None)
                G.add_edge(email_node, p, edge_type="mentions")

        for topic in e.get("topics") or []:
            topic_node = f"topic:{topic}"
            if not G.has_node(topic_node):
                G.add_node(topic_node, node_type="topic", topic=topic)
            G.add_edge(email_node, topic_node, edge_type="relates_to")
    return G


# ─── Graph-augmented retriever ───────────────────────────────────────
class GraphRetriever(BaseRetriever):
    def __init__(self, graph: nx.DiGraph, hybrid: HybridRetriever, body_by_filename: dict[str, str], **kwargs):
        super().__init__(system_prompt=GRAPH_SYSTEM_PROMPT, **kwargs)
        self._G = graph
        self._hybrid = hybrid
        self._bodies = body_by_filename
        # Pre-index thread members and topic members, each sorted by seq_num.
        self._thread_emails: dict[str, list[str]] = {}
        self._topic_emails: dict[str, list[str]] = {}
        for node, data in graph.nodes(data=True):
            if data.get("node_type") != "email":
                continue
            for _, target, edata in graph.out_edges(node, data=True):
                if edata.get("edge_type") == "belongs_to":
                    self._thread_emails.setdefault(target, []).append(node)
                elif edata.get("edge_type") == "relates_to":
                    self._topic_emails.setdefault(target, []).append(node)
        for members in self._thread_emails.values():
            members.sort(key=lambda n: graph.nodes[n].get("seq_num", 0))
        for members in self._topic_emails.values():
            members.sort(key=lambda n: graph.nodes[n].get("seq_num", 0))

    # — graph traversal helpers (provided) —
    def _filename(self, email_node: str) -> str:
        return self._G.nodes[email_node].get("filename", email_node.removeprefix("email:"))

    def _thread_for_email(self, email_node: str):
        for _, target, edata in self._G.out_edges(email_node, data=True):
            if edata.get("edge_type") == "belongs_to":
                return target
        return None

    def _closest_thread_neighbors(self, email_node: str) -> list[str]:
        """Up to THREAD_NEIGHBORS thread members closest to this e-mail (by seq_num),
        expanding outward from its position and excluding the e-mail itself."""
        thread_node = self._thread_for_email(email_node)
        if thread_node is None:
            return []
        members = self._thread_emails.get(thread_node, [])
        if email_node not in members:
            return [self._filename(n) for n in members[:THREAD_NEIGHBORS]]
        idx = members.index(email_node)
        out, lo, hi = [], idx - 1, idx + 1
        while len(out) < THREAD_NEIGHBORS and (lo >= 0 or hi < len(members)):
            if lo >= 0:
                out.append(self._filename(members[lo])); lo -= 1
            if len(out) < THREAD_NEIGHBORS and hi < len(members):
                out.append(self._filename(members[hi])); hi += 1
        return out

    def _topics_for_email(self, email_node: str) -> list[str]:
        return [t for _, t, ed in self._G.out_edges(email_node, data=True) if ed.get("edge_type") == "relates_to"]

    def _earliest_topic_emails(self, topic_node: str) -> list[str]:
        return [self._filename(n) for n in self._topic_emails.get(topic_node, [])[:TOPIC_EARLIEST]]

    def _body(self, filename: str) -> str:
        return self._bodies.get(filename, f"[content not found: {filename}]")

    def _assemble_context(self, seeds, thread_neighbors, topic_context) -> str:
        sections = []
        for fname in seeds:
            sections.append(f"=== RETRIEVED E-MAIL: {fname} ===\n{self._body(fname)}")
            for nb in thread_neighbors.get(fname, []):
                sections.append(f"--- THREAD CONTEXT for {fname}: {nb} ---\n{self._body(nb)}")
            for topic, files in topic_context.get(fname, {}).items():
                for ef in files:
                    sections.append(f'--- TOPIC CONTEXT "{topic}" (via {fname}): {ef} ---\n{self._body(ef)}')
        return ("\n\n" + "=" * 70 + "\n\n").join(sections)

    def retrievedContext(self, query: str) -> str:
        """Graph-augmented retrieval: hybrid seeds + thread + topic neighbours, deduped.

        Get HYBRID_FETCH_K seed e-mails, then for each seed, walk the graph to add its
        thread neighbors and the earliest e-mails on its topics. A `union` dict keyed
        by filename keeps each e-mail's single source label at its highest priority
        (seed > thread > topic), and we drop neighbors that are themselves seeds."""
        seeds = [name for name, _, _ in self._hybrid.getTopK(query, HYBRID_FETCH_K)]
        union: dict[str, str] = {}
        thread_neighbors: dict[str, list[str]] = {}
        topic_context: dict[str, dict[str, list[str]]] = {}

        for fname in seeds:
            union[fname] = "seed"  # seeds always win the label
            email_node = f"email:{fname}"
            if not self._G.has_node(email_node):
                continue
            nbrs = [n for n in self._closest_thread_neighbors(email_node) if n != fname]
            thread_neighbors[fname] = nbrs
            for nb in nbrs:
                union.setdefault(nb, "thread")
            tctx: dict[str, list[str]] = {}
            for topic_node in self._topics_for_email(email_node):
                earliest = [ef for ef in self._earliest_topic_emails(topic_node) if ef != fname]
                if earliest:
                    tctx[topic_node.removeprefix("topic:")] = earliest
                    for ef in earliest:
                        union.setdefault(ef, "topic")
            topic_context[fname] = tctx

        # Drop thread/topic neighbors that are themselves seeds (avoid redundancy).
        for fname in seeds:
            thread_neighbors[fname] = [n for n in thread_neighbors.get(fname, []) if union.get(n) != "seed"]
            topic_context[fname] = {
                t: [ef for ef in files if union.get(ef) != "seed"]
                for t, files in topic_context.get(fname, {}).items()
            }
        return self._assemble_context(seeds, thread_neighbors, topic_context)


def main():
    require_api_key()
    annotated_path = sys.argv[1] if len(sys.argv) > 1 else "annotatedEmails.json"
    print(f"Loading {annotated_path} ...")
    emails, body_by_filename = load_annotated(annotated_path)
    print(f"  {len(emails)} emails loaded.")
    db = build_or_load_db(emails, CHROMA_DIR)
    hybrid = HybridRetriever(emails, db)
    print("Building knowledge graph ...")
    graph = build_graph(emails)
    print(f"  graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges.")
    GraphRetriever(graph=graph, hybrid=hybrid, body_by_filename=body_by_filename).chat()


if __name__ == "__main__":
    main()
