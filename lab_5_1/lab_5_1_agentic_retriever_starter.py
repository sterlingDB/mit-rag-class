r"""Lab 5.1 (STARTER) — an agentic retriever that decides when to retrieve more.

Run:  python lab_5_1_agentic_retriever_starter.py <emails_dir>

Until now,  retrieval was a fixed pipeline: one query in, top-k e-mails out. Here, the
retriever becomes *agentic*. It retrieves, looks at what it has, and decides whether
that's enough or whether to issue more queries, looping until it's satisfied (or hits
a cap). The control flow is built with LangGraph: a `retrieve` node and an `analyze`
node with a conditional edge that loops back or stops.

    retrieve ──▶ analyze ──(need more?)──▶ retrieve
                    │
                    └──(enough / max iters)──▶ done

Your job: Implement the analyze_node() function, which decides whether the agent has enough
information to answer or should issue additional retrieval queries. The retrieve node, the 
graph wiring, the loop guard, and the hybrid retriever are all provided. The lesson is implementing 
the agent's decision to keep retrieving or stop. The rest of the retrieval system and graph workflow are provided.

Setup
-----
1. Create the environment (one-time). Either use conda:
       conda env create -f environment.yml
       conda activate ragcourse
   or a plain virtual environment + pip:
       python -m venv .venv
       #  Windows:      .venv\Scripts\activate
       #  macOS/Linux:  source .venv/bin/activate
       python -m pip install --upgrade pip
       pip install -r requirements.txt   # pinned versions — avoids dependency-drift errors
2. Add the OpenRouter API key provided for this program. Create a file
   named ".env" in this folder containing a single line:
       OPENROUTER_API_KEY=sk-or-your-key-here
   (or set it in your shell —  Windows:  setx OPENROUTER_API_KEY sk-or-...
    macOS/Linux:  export OPENROUTER_API_KEY=sk-or-...)
3. Email data: Place the company email files (one .txt per email) in a folder
   named 'detailedEmails' in this directory, or pass a folder path as the first
   argument. The folder must contain the .txt files. The vector DB is persisted to
   ./chroma_db (delete it to rebuild).

Sample questions to try (over the email corpus)
-----------------------------------------------
Broad/multifaceted questions show the agent issuing follow-up queries:
    "What were the main problems with the government project and how were they resolved?"
    "Give me the full picture of the budget concerns raised across the company."
With debug output on, you'll see each retrieval round and the agent's decision.
"""
import json
import os
import re
import sys
from abc import ABC, abstractmethod
from typing import TypedDict

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, StateGraph
from rank_bm25 import BM25Okapi

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "openai/gpt-5.4-mini"  # Latest small OpenAI model, fast; covered by program credits
EMBEDDING_MODEL = "openai/text-embedding-3-small"
CHROMA_DIR = "chroma_db"
NUM_RETRIEVED = 5
CANDIDATE_POOL = 10
WEIGHT_BM25 = 0.5
WEIGHT_VECTOR = 0.5
MAX_ITERATIONS = 3   # Safety cap on agentic retrieval rounds
DEBUG = True

SYSTEM_PROMPT = """You are a helpful assistant for Precision Paperclip Inc. \
You answer questions by drawing information exclusively from the company e-mails \
provided to you as context in each message.

Rules:
- If the answer can be found in the provided e-mails, answer clearly and concisely.
- If the provided e-mails do not contain enough information to answer the question, \
say so explicitly and do not speculate or use outside knowledge.
- Do not answer questions that are unrelated to the content of the provided e-mails."""

ANALYZE_SYSTEM = """You are an agentic retrieval assistant for a company e-mail database (Precision Paperclip Inc.).

Your job is to decide whether the e-mails retrieved so far are sufficient to answer the
user's question and, if not, to generate additional retrieval queries.

You are given the original question, the queries already executed, and the e-mails
retrieved so far. Respond with a JSON object and nothing else:
{
  "done": true,
  "new_queries": [],
  "reasoning": "brief explanation"
}

Rules:
- Set "done": True when the retrieved e-mails contain enough information to answer.
- Set "done": False and provide 1-3 focused "new_queries" targeting missing information
  not already covered by prior queries.
- Do not repeat queries already executed."""


def require_api_key() -> None:
    """Exit early with a clear message if OPENROUTER_API_KEY is not set, instead of
    failing later with a KeyError when the model client is created."""
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit(
            "\n[setup] OPENROUTER_API_KEY is not set.\n"
            "  1. Use the OpenRouter API key provided for this program."
            "  2. Create a file named '.env' in this folder with one line:\n"
            "         OPENROUTER_API_KEY=sk-or-your-key-here\n"
            "     or set it in your shell  (Windows: setx OPENROUTER_API_KEY sk-or-... ;\n"
            "     macOS/Linux: export OPENROUTER_API_KEY=sk-or-...).\n"
        )


# ─── Provided: chat loop + base class ────────────────────────────────
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
    def __init__(self, llm_model: str = LLM_MODEL):
        self._llm = ChatOpenAI(
            model=llm_model,
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url=OPENROUTER_BASE_URL,
        )
        self._history = [SystemMessage(content=SYSTEM_PROMPT)]

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


# ─── Provided: hybrid retrieval stack (from Module 2) ────────────────
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


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=OPENROUTER_BASE_URL,
        check_embedding_ctx_length=False,  # OpenRouter needs raw text, not pre-tokenized input
    )


def build_or_load_db(emails_dir: str, chroma_dir: str = CHROMA_DIR) -> Chroma:
    if os.path.isdir(chroma_dir) and os.listdir(chroma_dir):
        print(f"Loading existing vector DB from {chroma_dir}/")
        return Chroma(persist_directory=chroma_dir, embedding_function=get_embeddings())
    print("Building vector DB (first run — embedding the emails)...")
    docs = []
    for fname in sorted(os.listdir(emails_dir)):
        if not fname.endswith(".txt"):
            continue
        with open(os.path.join(emails_dir, fname), encoding="utf-8", errors="replace") as fh:
            docs.append(Document(page_content=fh.read(), metadata={"source": fname}))
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


def _load_emails(emails_dir: str) -> tuple[list[str], list[str]]:
    paths = sorted(os.path.join(emails_dir, f) for f in os.listdir(emails_dir) if f.endswith(".txt"))
    contents = []
    for p in paths:
        with open(p, encoding="utf-8", errors="replace") as fh:
            contents.append(fh.read())
    return [os.path.basename(p) for p in paths], contents


class HybridRetriever:
    """Score-fusion of BM25 + vector, returning (filename, content, score)."""

    def __init__(self, emails_dir: str, db: Chroma):
        self._paths, self._contents = _load_emails(emails_dir)
        self._bm25 = BM25Okapi([tokenize(doc) for doc in self._contents])
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


# ─── Agentic retriever (LangGraph: retrieve -> analyze -> loop) ──────
class _AgentState(TypedDict):
    original_query: str
    pending_queries: list[str]
    executed_queries: list[str]
    email_bodies: dict[str, str]
    iterations: int
    done: bool


class AgenticRetriever(BaseRetriever):
    def __init__(self, hybrid: HybridRetriever, top_k: int = NUM_RETRIEVED,
                 max_iterations: int = MAX_ITERATIONS, debug: bool = DEBUG, **kwargs):
        super().__init__(**kwargs)
        self._hybrid = hybrid
        self._top_k = top_k
        self._max_iterations = max_iterations
        self._debug = debug
        self._graph = self._build_graph()

    def _build_graph(self):
        # Provided: Runs the hybrid retriever for each pending query and
        # accumulates the e-mails into the state.
        def retrieve_node(state: _AgentState) -> dict:
            email_bodies = dict(state["email_bodies"])
            executed = list(state["executed_queries"])
            for query in state["pending_queries"]:
                if self._debug:
                    print(f"[agentic] retrieving for: {query!r}")
                for name, content, _ in self._hybrid.getTopK(query, self._top_k):
                    email_bodies[name] = content
                executed.append(query)
            return {
                "email_bodies": email_bodies,
                "executed_queries": executed,
                "pending_queries": [],
                "iterations": state["iterations"] + 1,
            }

        def analyze_node(state: _AgentState) -> dict:
            """Decide whether to stop or retrieve more.

            TODO (Step 1): this is the agentic decision.
              1. If state["iterations"] >= self._max_iterations, stop now:
                     return {"done": True, "pending_queries": []}
              2. Otherwise build a context string from state["original_query"],
                 state["executed_queries"], and state["email_bodies"], and ask the LLM
                 with the provided ANALYZE_SYSTEM prompt:
                     resp = self._llm.invoke([SystemMessage(content=ANALYZE_SYSTEM),
                                              HumanMessage(content=user_content)])
                 - Cap the e-mails you include to ~20. The set grows every round, so this
                   keeps the decision prompt small, fast, and within the context window;
                   the full set is still returned to the chatbot at the end.
                 - Include the executed queries so the LLM targets the missing information
                   and does not repeat searches it has already run.
              3. Parse the reply as JSON. LLMs often wrap JSON in a ```json ... ``` code
                 fence, so strip the leading fence line and the trailing ``` before
                 json.loads. Read "done" (bool) and, when not done, "new_queries" (list).
              4. If parsing fails, default to done=True with no new queries. Stopping on a
                 parse failure (together with the max_iterations cap) is what guarantees
                 the agent never loops forever.
              5. return {"done": done, "pending_queries": new_queries}

            Delete the raise NotImplementedError line once your code works.
            """
            raise NotImplementedError("Implement analyze_node — see the TODO above.")

        def should_continue(state: _AgentState) -> str:
            if state["done"] or state["iterations"] >= self._max_iterations:
                return END
            return "retrieve"

        graph = StateGraph(_AgentState)
        graph.add_node("retrieve", retrieve_node)
        graph.add_node("analyze", analyze_node)
        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "analyze")
        graph.add_conditional_edges("analyze", should_continue)
        return graph.compile()

    def retrievedContext(self, query: str) -> str:
        initial: _AgentState = {
            "original_query": query,
            "pending_queries": [query],
            "executed_queries": [],
            "email_bodies": {},
            "iterations": 0,
            "done": False,
        }
        final = self._graph.invoke(initial)
        bodies = final["email_bodies"]
        return "\n\n---\n\n".join(f"[{name}]\n{content}" for name, content in bodies.items())


if __name__ == "__main__":
    require_api_key()
    emails_dir = sys.argv[1] if len(sys.argv) > 1 else "detailedEmails"
    db = build_or_load_db(emails_dir, CHROMA_DIR)
    hybrid = HybridRetriever(emails_dir=emails_dir, db=db)
    AgenticRetriever(hybrid=hybrid).chat()
