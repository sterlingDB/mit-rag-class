r"""Lab 5.2 (SOLUTION) — a tool-using agent that plans which action to take.

Run:  python lab_5_2_tool_using_agent_solution.py <emails_dir>

In Lab 5.1 the agent's only action was "retrieve more". Here the agent is put fully
in charge of the conversation and can choose among several actions on each step:

    plan ──▶ retrieve        (semantic search of the e-mail DB)
         ──▶ retrieve_by_date (pull every e-mail on a date / short date range)
         ──▶ clarify          (ask the user a question, then re-plan)
         ──▶ answer           (respond to the user)  ──▶ done

The PLAN node is the brain: it looks at the conversation, what's been retrieved, and
any clarifications, then emits a JSON action. This is built with LangGraph; the agent
owns the chat loop (so it can ask clarifying questions), and a batch mode (query())
skips clarification for automated testing.

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
2. Add your OpenRouter API key (free at https://openrouter.ai/keys). Create a file
   named ".env" in this folder containing a single line:
       OPENROUTER_API_KEY=sk-or-your-key-here
   (or set it in your shell —  Windows:  setx OPENROUTER_API_KEY sk-or-...
    macOS/Linux:  export OPENROUTER_API_KEY=sk-or-...)
3. Email data: place the company email files (one .txt per email) in a folder
   named 'detailedEmails' in this directory, or pass a folder path as the first
   argument. Filenames must follow mail_MM_DD_YY_<id>.txt so the date tool works.
   The folder MUST contain .txt files. The vector DB is persisted to ./chroma_db.

Sample questions to try (over the email corpus)
-----------------------------------------------
    "What e-mails were sent on 01/01/2014?"        -> agent chooses the by_date tool.
    "What was decided about the government project?" -> agent chooses semantic retrieve.
    "Tell me about the project."                    -> agent may ask a clarifying question.
With debug output ON you'll see the planned action and reasoning at each step.
"""
import glob
import json
import os
import re
import sys
from datetime import date, timedelta
from typing import Optional, TypedDict

from dateutil import parser as dateutil_parser
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, StateGraph
from rank_bm25 import BM25Okapi

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "openai/gpt-5.4-mini"  # latest small OpenAI model, fast; covered by course credits
EMBEDDING_MODEL = "openai/text-embedding-3-small"
CHROMA_DIR = "chroma_db"
NUM_RETRIEVED = 5
CANDIDATE_POOL = 10
WEIGHT_BM25 = 0.5
WEIGHT_VECTOR = 0.5
MAX_ITERATIONS = 5
EMAILS_DIR_DEFAULT = "detailedEmails"
MAX_DATE_SPAN_DAYS = 7   # cap by_date ranges so we don't dump the whole corpus

_FILENAME_RE = re.compile(r"^mail_(\d{2})_(\d{2})_(\d{2})_\d+\.txt$")

ANSWER_SYSTEM = """You are a research assistant for Precision Paperclip Inc. \
Answer questions exclusively from the company e-mails provided as context. Consider \
any clarifications, which may add important details; if the original message is not a \
full question, answer the last question asked in the clarifications. If the retrieved \
e-mails do not contain enough information, say so explicitly. Do not speculate or use \
outside knowledge. If the user is simply asking to exit, answer exactly: Exiting"""

PLAN_SYSTEM = """You are a research agent for Precision Paperclip Inc. You help the user \
find information about the company using a semantic search database of company e-mails.

You are given the user's question (and prior conversation), the queries already executed,
the e-mails retrieved so far, and any clarifications from the user. Decide what to do next
and respond with a JSON object ONLY:
{
  "action": "retrieve" | "by_date" | "clarify" | "answer",
  "queries": ["query1", "query2"],   // 1-3 NEW queries; only for action=="retrieve"; don't repeat executed ones
  "start_date": "MM/DD/YYYY",          // only for action=="by_date"
  "end_date": "MM/DD/YYYY",            // optional; only for action=="by_date" (range <= 7 days)
  "clarification": "question text",    // only for action=="clarify"
  "reasoning": "brief explanation"
}

Guidelines:
- "retrieve": you need more information via semantic search.
- "by_date": the question references specific dates or a short range ("start of the month",
  "this week", "between X and Y"). Retrieves ALL e-mails on that day/range (<= 7 days).
- "clarify": the message isn't really a question and you must ask for more input. Use ONLY
  as a last resort, and only AFTER attempting to query the database.
- "answer": you have enough information, or the user asked to exit.
Products are sometimes referred to by multiple names — searching alternate names can help."""


def require_api_key() -> None:
    """Exit early with a clear message if OPENROUTER_API_KEY is not set, instead of
    failing later with a KeyError when the model client is created."""
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit(
            "\n[setup] OPENROUTER_API_KEY is not set.\n"
            "  1. Get a free key at https://openrouter.ai/keys\n"
            "  2. Create a file named '.env' in this folder with one line:\n"
            "         OPENROUTER_API_KEY=sk-or-your-key-here\n"
            "     or set it in your shell  (Windows: setx OPENROUTER_API_KEY sk-or-... ;\n"
            "     macOS/Linux: export OPENROUTER_API_KEY=sk-or-...).\n"
        )


# ─── Hybrid retrieval stack (from Module 2) ──────────────────────────
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


# ─── Tool-using agent (LangGraph: plan -> retrieve/by_date/clarify/answer) ──
class _AgentState(TypedDict):
    conversation_history: list[dict]
    clarification_history: list[str]
    pending_queries: list[str]
    pending_date_range: dict
    executed_queries: list[str]
    email_bodies: dict[str, str]
    iterations: int
    mode: str
    next_action: str
    clarification_question: str
    answer: str
    done: bool


class ToolUsingAgent:
    def __init__(self, hybrid: HybridRetriever, emails_dir: str, top_k: int = NUM_RETRIEVED,
                 max_iterations: int = MAX_ITERATIONS, debug: bool = True):
        self._llm = ChatOpenAI(model=LLM_MODEL, api_key=os.environ["OPENROUTER_API_KEY"], base_url=OPENROUTER_BASE_URL)
        self._hybrid = hybrid
        self._emails_dir = emails_dir
        self._top_k = top_k
        self._max_iterations = max_iterations
        self._debug = debug
        self._graph = self._build_graph()

    # — tool: retrieve all e-mails on a date / short range (filename-encoded dates) —
    def retrieve_by_date(self, start_date: str, end_date: Optional[str] = None) -> list[tuple[str, str]]:
        start = dateutil_parser.parse(start_date).date()
        end = dateutil_parser.parse(end_date).date() if end_date else start
        if (end - start).days > MAX_DATE_SPAN_DAYS:
            end = start + timedelta(days=MAX_DATE_SPAN_DAYS)  # clamp to the cap, don't dump the corpus
        results = []
        for path in glob.glob(os.path.join(self._emails_dir, "mail_*.txt")):
            m = _FILENAME_RE.match(os.path.basename(path))
            if not m:
                continue
            email_date = date(2000 + int(m.group(3)), int(m.group(1)), int(m.group(2)))
            if start <= email_date <= end:
                try:
                    with open(path, encoding="utf-8", errors="replace") as f:
                        results.append((os.path.basename(path), f.read()))
                except OSError:
                    pass
        if self._debug:
            print(f"[agent] by_date {start}..{end}: {len(results)} e-mail(s)")
        return results

    def _build_graph(self):
        def retrieve_node(state: _AgentState) -> dict:
            email_bodies = dict(state["email_bodies"])
            executed = list(state["executed_queries"])
            for query in state["pending_queries"]:
                if self._debug:
                    print(f"[agent] retrieving for: {query!r}")
                for name, content, _ in self._hybrid.getTopK(query, self._top_k):
                    email_bodies[name] = content
                executed.append(query)
            return {"email_bodies": email_bodies, "executed_queries": executed,
                    "pending_queries": [], "iterations": state["iterations"] + 1}

        def retrieve_by_date_node(state: _AgentState) -> dict:
            dr = state.get("pending_date_range", {})
            start, end = dr.get("start", ""), (dr.get("end", "") or None)
            if not start:
                return {"pending_date_range": {}, "iterations": state["iterations"] + 1}
            email_bodies = dict(state["email_bodies"])
            for name, content in self.retrieve_by_date(start, end):
                email_bodies[name] = content
            tag = f"DATE:{start}" + (f"..{end}" if end else "")
            return {"email_bodies": email_bodies,
                    "executed_queries": list(state["executed_queries"]) + [tag],
                    "pending_date_range": {}, "iterations": state["iterations"] + 1}

        def plan_node(state: _AgentState) -> dict:
            """The agent's brain: pick the next action as a JSON object."""
            if state["iterations"] >= self._max_iterations:
                if self._debug:
                    print("[agent] max iterations — forcing answer.")
                return {"next_action": "answer", "pending_queries": [], "pending_date_range": {}, "clarification_question": ""}

            email_summary = "\n\n---\n\n".join(
                f"[{name}]\n{content}" for name, content in list(state["email_bodies"].items())[:20]
            )
            executed_str = "\n".join(f"- {q}" for q in state["executed_queries"]) or "(none)"
            clarif_str = "\n".join(state["clarification_history"]) or "(none)"
            conv_str = "\n".join(f"{m['role'].capitalize()}: {m['content']}"
                                 for m in state["conversation_history"][-6:]) or "(none)"
            user_content = (
                f"Prior conversation:\n{conv_str}\n\n"
                f"Interactions so far:\n{clarif_str}\n\n"
                f"Queries already executed:\n{executed_str}\n\n"
                f"E-mails retrieved ({len(state['email_bodies'])} total):\n\n{email_summary}"
            )
            response = self._llm.invoke([SystemMessage(content=PLAN_SYSTEM), HumanMessage(content=user_content)])
            raw = (response.content if hasattr(response, "content") else str(response)).strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
            try:
                result = json.loads(raw)
                action = result.get("action", "answer")
            except (json.JSONDecodeError, ValueError):
                action, result = "answer", {}
            if self._debug:
                print(f"[agent] action={action} reasoning={result.get('reasoning', '')!r}")
            if action == "retrieve":
                return {"next_action": "retrieve", "pending_queries": result.get("queries", []),
                        "pending_date_range": {}, "clarification_question": ""}
            if action == "by_date":
                return {"next_action": "by_date", "pending_queries": [],
                        "pending_date_range": {"start": result.get("start_date", ""), "end": result.get("end_date", "")},
                        "clarification_question": ""}
            if action == "clarify":
                return {"next_action": "clarify", "pending_queries": [], "pending_date_range": {},
                        "clarification_question": result.get("clarification", "Could you clarify your question?")}
            return {"next_action": "answer", "pending_queries": [], "pending_date_range": {}, "clarification_question": ""}

        def clarify_node(state: _AgentState) -> dict:
            question = state["clarification_question"]
            history = list(state["clarification_history"])
            if state["mode"] == "chat":
                print(f"\nAssistant: {question}")
                history.append(f"Q: {question}\nA: {input('You: ').strip()}")
            else:  # batch mode never blocks on input
                history.append(f"Q: {question}\n[no clarification available in batch mode]")
            return {"clarification_history": history, "clarification_question": ""}

        def answer_node(state: _AgentState) -> dict:
            email_context = "\n\n---\n\n".join(f"[{n}]\n{c}" for n, c in state["email_bodies"].items())
            conv_str = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in state["conversation_history"][-6:])
            clarif_str = "\n".join(state["clarification_history"])
            parts = []
            if conv_str:
                parts.append(f"Prior conversation:\n{conv_str}")
            if clarif_str:
                parts.append(f"Interactions so far:\n{clarif_str}")
            parts.append(f"Retrieved e-mails:\n{email_context}")
            response = self._llm.invoke([SystemMessage(content=ANSWER_SYSTEM), HumanMessage(content="\n\n".join(parts))])
            return {"answer": response.content if hasattr(response, "content") else str(response), "done": True}

        def route_from_plan(state: _AgentState) -> str:
            action = state.get("next_action", "answer")
            if action == "retrieve" and state["pending_queries"]:
                return "retrieve"
            if action == "by_date" and state.get("pending_date_range", {}).get("start"):
                return "retrieve_by_date"
            if action == "clarify":
                return "clarify"
            return "answer"

        graph = StateGraph(_AgentState)
        graph.add_node("retrieve", retrieve_node)
        graph.add_node("retrieve_by_date", retrieve_by_date_node)
        graph.add_node("plan", plan_node)
        graph.add_node("clarify", clarify_node)
        graph.add_node("answer", answer_node)
        graph.set_entry_point("plan")
        graph.add_edge("retrieve", "plan")
        graph.add_edge("retrieve_by_date", "plan")
        graph.add_conditional_edges("plan", route_from_plan, {
            "retrieve": "retrieve", "retrieve_by_date": "retrieve_by_date",
            "clarify": "clarify", "answer": "answer",
        })
        graph.add_edge("clarify", "plan")
        graph.add_edge("answer", END)
        return graph.compile()

    def _run(self, question: str, mode: str, conversation_history: Optional[list[dict]] = None) -> str:
        initial: _AgentState = {
            "conversation_history": conversation_history or [],
            "clarification_history": [question],
            "pending_queries": [question],
            "pending_date_range": {},
            "executed_queries": [],
            "email_bodies": {},
            "iterations": 0,
            "mode": mode,
            "next_action": "plan",
            "clarification_question": "",
            "answer": "",
            "done": False,
        }
        return self._graph.invoke(initial)["answer"]

    def query(self, question: str) -> str:
        """Batch mode — fresh agent each call, no clarification (for testing)."""
        return self._run(question, "batch")

    def chat(self) -> None:
        print("Tool-using research agent. Type your question; 'exit'/'quit' to stop.\n")
        conversation_history: list[dict] = []
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in {"exit", "quit"}:
                print("Goodbye.")
                break
            if not user_input:
                continue
            try:
                answer = self._run(user_input, "chat", conversation_history)
            except Exception as e:
                print(f"Error: {e}")
                continue
            print(f"\nAssistant: {answer}\n")
            if answer.strip() == "Exiting":
                print("Goodbye.")
                break
            conversation_history.append({"role": "user", "content": user_input})
            conversation_history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    require_api_key()
    emails_dir = sys.argv[1] if len(sys.argv) > 1 else EMAILS_DIR_DEFAULT
    db = build_or_load_db(emails_dir, CHROMA_DIR)
    hybrid = HybridRetriever(emails_dir=emails_dir, db=db)
    ToolUsingAgent(hybrid=hybrid, emails_dir=emails_dir).chat()
