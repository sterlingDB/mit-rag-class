r"""Lab 6.2 (STARTER) — measuring the token cost of an agent and cutting it.

Run:  python lab_6_2_cost_measurement_starter.py <emails_dir> [--expensive]

This is the same tool-using agent you built in Lab 5.2 (plan -> retrieve/by_date/clarify/answer). 
The cost of a system like this is measured in tokens, and there are two
ways to spend less: use fewer tokens and use cheaper tokens. Before optimizing anything,x we
must be able to MEASURE — that is the whole point of this lab.

Your job (two small changes turn the Lab 5.2 agent into the Lab 6.2 agent):

  - Step 1a — per-role models: The agent makes two kinds of chat-LLM calls: the planner
    (decides the next action) and the answer model (writes the final reply). In Lab 5.2, both
    used one model; there is no reason they must. In __init__ the three model names are
    already stored for you (self._plan_model / self._answer_model / self._retriever_model);
    build the two chat clients `self._plan_llm` and `self._answer_llm` from them.
  - Step 1b — token tracking: Implement `_track_usage(response, model_name)`. Every
    LangChain chat response carries a `usage_metadata` field with `input_tokens`,
    `output_tokens`, and an `input_token_details` sub-dictionary (holds `cache_read` when the
    provider served part of the prompt from cache). Accumulate those per model into
    self._token_usage. (get/print/reset_token_usage and the `_track_usage` calls in
    plan_node/answer_node are already wired for you.)

Everything else — the hybrid retriever, the graph, retrieve_by_date, plan_node parsing,
and the `run_experiment` cost harness — is provided (you built the agent itself in Lab 5.2).

The faculty experiment (in `main()` below, ready to run once your TODOs are done):
  - Step 2: Run the small test suite with one model at a time across a cost ladder
    [google/gemma-4-31b-it:free, openai/gpt-4o-mini, openai/gpt-5.2-pro].
  - Step 3: mixed models — expensive planner (openai/gpt-5.2-pro), cheap answer
    (openai/gpt-4o-mini). Compare quality vs. cost.

    ┌──────────────────────────────────────────────────────────────────────────┐
    │  COST WARNING                                                              │
    │  openai/gpt-5.2-pro is roughly ~10x the price of the others. It (and the   │
    │  mixed run) are opt-in: pass --expensive. The default run uses only the    │
    │  free / cheap models, on a tiny 4-question set. Never point an expensive   │
    │  model at a large question set "just to see."                             │
    └──────────────────────────────────────────────────────────────────────────┘

Setup
-----
1. Ensure that you have completed the program's one-time environment and
   OpenRouter API key setup, and activate the configured environment.
2. Install any additional dependencies required for this lab, if they are
   not already available:
       pip install python-dotenv langchain-openai langchain-core langchain-chroma rank-bm25  
3. Email data: Place the company email files (one .txt per email) in a folder
   named 'detailedEmails' in this directory, or pass a folder path as the first
   argument. The folder MUST contain the .txt files.
"""
import glob
import json
import os
import re
import sys
from datetime import date
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
LLM_MODEL = "openai/gpt-5.4-mini"  # Latest small OpenAI model, fast; covered by course credits
EMBEDDING_MODEL = "openai/text-embedding-3-small"
CHROMA_DIR = "chroma_db"
NUM_RETRIEVED = 5
CANDIDATE_POOL = 10
WEIGHT_BM25 = 0.5
WEIGHT_VECTOR = 0.5
MAX_ITERATIONS = 5
EMAILS_DIR_DEFAULT = "detailedEmails"

# ─── Cost experiment configuration (faculty Steps 2 & 3) ─────────────
# Weak -> strong cost ladder; all exist on OpenRouter. gpt-5.2-pro is the expensive one.
CHEAP_MODELS = ["google/gemma-4-31b-it:free", "openai/gpt-4o-mini"]
EXPENSIVE_MODEL = "openai/gpt-5.2-pro"
MIXED_PLANNER = "openai/gpt-5.2-pro"   # strong model drives the plan
MIXED_ANSWER = "openai/gpt-4o-mini"    # cheap model writes the answer

# A small suite (keep it tiny — every question is real paid tokens). Adapt to your corpus.
EXPERIMENT_QUESTIONS = [
    "What e-mails were sent on 01/01/2014?",
    "What was decided about the government project?",
    "What products does Precision Paperclip Inc. make?",
    "Were there any supplier or delivery problems discussed?",
]

# Approximate prices in USD per 1,000,000 tokens — for a rough cost estimate only.
# Verify live prices at https://openrouter.ai/models before quoting real numbers.
# (gpt-5.2-pro's rate is an illustrative "expensive tier" placeholder.)
APPROX_PRICE_PER_MTOK = {
    "google/gemma-4-31b-it:free": {"input": 0.0, "output": 0.0},
    "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "openai/gpt-5.2-pro": {"input": 10.0, "output": 30.0},
    "openai/gpt-5.4-mini": {"input": 0.20, "output": 0.80},
}

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
  "end_date": "MM/DD/YYYY",            // optional; only for action=="by_date" and a range is needed
  "clarification": "question text",    // only for action=="clarify"
  "reasoning": "brief explanation"
}

Guidelines:
- "retrieve": You need more information via semantic search.
- "by_date": The question references specific dates or a short range ("start of the month",
  "this week", "between X and Y"). Retrieves ALL e-mails on that day/range.
- "clarify": The message isn't really a question and you must ask for more input. Use ONLY
  as a last resort, and only AFTER attempting to query the database.
- "answer": You have enough information, or the user asked to exit.
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
        return [1.0] * len(scores)  # All-tied -> treat as top (matches faculty hybridRetriever)
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
    def __init__(self, emails_dir: str, db: Chroma, retriever_model: Optional[str] = None):
        # retriever_model is accepted for parity with the agent's three-model design. This
        # consolidated retriever is pure BM25 + embedding-vector fusion (fixed embedding
        # model) and issues no chat-LLM calls, so it never appears in the token tally.
        self._retriever_model = retriever_model or LLM_MODEL
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
                 max_iterations: int = MAX_ITERATIONS, debug: bool = True,
                 plan_model: Optional[str] = None, answer_model: Optional[str] = None,
                 retriever_model: Optional[str] = None):
        # The three model NAMES are stored for you (each falls back to the single default).
        self._plan_model = plan_model or LLM_MODEL
        self._answer_model = answer_model or LLM_MODEL
        self._retriever_model = retriever_model or LLM_MODEL

        # TODO (Step 1a) — build two separate chat clients so the planner and the answer
        # step can use different models. Replace the None placeholders below with:
        #     self._plan_llm   = ChatOpenAI(model=self._plan_model,
        #                                   api_key=os.environ["OPENROUTER_API_KEY"],
        #                                   base_url=OPENROUTER_BASE_URL)
        #     self._answer_llm = ChatOpenAI(model=self._answer_model,
        #                                   api_key=os.environ["OPENROUTER_API_KEY"],
        #                                   base_url=OPENROUTER_BASE_URL)
        # (Same ChatOpenAI(...) call you already know — just one per role.)
        self._plan_llm = None    # <-- replace (Step 1a)
        self._answer_llm = None  # <-- replace (Step 1a)

        # The hybrid is injected (built once, reused). It makes no chat-LLM calls, so
        # retriever_model does not affect the token tally; we keep it for API parity.
        self._hybrid = hybrid
        self._emails_dir = emails_dir
        self._top_k = top_k
        self._max_iterations = max_iterations
        self._debug = debug
        self._token_usage: dict[str, dict[str, int]] = {}
        self._graph = self._build_graph()

    # ── token-usage tracking ────────────────────────────────────────
    def _track_usage(self, response, model_name: str) -> None:
        """Accumulate this response's token counts under `model_name`.

        TODO (Step 1b): Every LangChain chat response exposes `usage_metadata`, a dict with
        `input_tokens`, `output_tokens`, and `input_token_details` (which holds `cache_read`
        when the provider served part of the prompt from cache). Some responses/models omit
        it, so no-op when it is missing. Steps:
          1. usage = getattr(response, "usage_metadata", None); if not usage: return
          2. bucket = self._token_usage.setdefault(model_name,
                          {"input": 0, "output": 0, "cached": 0})
          3. bucket["input"]  += usage.get("input_tokens", 0)
             bucket["output"] += usage.get("output_tokens", 0)
          4. details = usage.get("input_token_details", {})
             bucket["cached"] += details.get("cache_read", 0)

        Delete the raise NotImplementedError line once your code works.
        """
        raise NotImplementedError("Implement _track_usage — see the TODO above.")

    def get_token_usage(self) -> dict[str, dict[str, int]]:
        """Return accumulated token counts keyed by model name (a safe copy)."""
        return {model: dict(counts) for model, counts in self._token_usage.items()}

    def print_token_usage(self) -> None:
        for model, counts in self._token_usage.items():
            print(f"{model}: {counts['input']} input, {counts['output']} output, {counts['cached']} cached")

    def reset_token_usage(self) -> None:
        self._token_usage.clear()

    # — tool: retrieve all e-mails on a date / short range (filename-encoded dates) —
    def retrieve_by_date(self, start_date: str, end_date: Optional[str] = None) -> list[tuple[str, str]]:
        """Return (filename, content) for every e-mail on start_date, or within the
        inclusive range [start_date, end_date] — both endpoints are included. With no
        end_date, only start_date is used. (Provided — you wrote this in Lab 5.2.)"""
        start = dateutil_parser.parse(start_date).date()
        end = dateutil_parser.parse(end_date).date() if end_date else start
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
            """The agent's brain: Pick the next action as a JSON object (planner LLM).

            Provided (you wrote it in Lab 5.2). Note the one new line vs Lab 5.2:
            After the planner invoke we call self._track_usage(response, self._plan_model)
            — that is the instrumentation you complete in Step 1b.
            """
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
            response = self._plan_llm.invoke([SystemMessage(content=PLAN_SYSTEM), HumanMessage(content=user_content)])
            self._track_usage(response, self._plan_model)
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
            # "Clarify only as a last resort, after querying" lives in PLAN_SYSTEM — it's
            # prompt-guided, not enforced in code; the router honors whatever the planner
            # returns. In batch mode (query()) there is no user to answer, so we record that
            # the clarification was unavailable; the clarify->plan edge then lets the agent
            # re-plan with what it has instead of blocking on input().
            question = state["clarification_question"]
            history = list(state["clarification_history"])
            if state["mode"] == "chat":
                print(f"\nAssistant: {question}")
                history.append(f"Q: {question}\nA: {input('You: ').strip()}")
            else:  # batch mode never blocks on input
                history.append(f"Q: {question}\n[no clarification available in batch mode]")
            return {"clarification_history": history, "clarification_question": ""}

        def answer_node(state: _AgentState) -> dict:
            """Compose the final answer (answer LLM). Provided — note the new
            self._track_usage(response, self._answer_model) instrumentation line."""
            email_context = "\n\n---\n\n".join(f"[{n}]\n{c}" for n, c in state["email_bodies"].items())
            conv_str = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in state["conversation_history"][-6:])
            clarif_str = "\n".join(state["clarification_history"])
            parts = []
            if conv_str:
                parts.append(f"Prior conversation:\n{conv_str}")
            if clarif_str:
                parts.append(f"Interactions so far:\n{clarif_str}")
            parts.append(f"Retrieved e-mails:\n{email_context}")
            response = self._answer_llm.invoke([SystemMessage(content=ANSWER_SYSTEM), HumanMessage(content="\n\n".join(parts))])
            self._track_usage(response, self._answer_model)
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
        """Batch mode — no clarification prompts (for the automated cost suite)."""
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


# ─── Provided: cost experiment harness ───────────────────────────────
def estimate_cost_usd(usage: dict[str, dict[str, int]]) -> float:
    """Rough USD estimate from a token tally using APPROX_PRICE_PER_MTOK.

    Naive: input * in_rate + output * out_rate. Cached tokens are counted at the full
    input rate here (real providers discount them), so this is an UPPER-ish estimate.
    Unknown models contribute 0 — add them to APPROX_PRICE_PER_MTOK to include them.
    """
    total = 0.0
    for model, counts in usage.items():
        price = APPROX_PRICE_PER_MTOK.get(model)
        if not price:
            continue
        total += counts["input"] / 1_000_000 * price["input"]
        total += counts["output"] / 1_000_000 * price["output"]
    return total


def run_experiment(questions: list[str], plan_model: str, answer_model: str, *,
                   emails_dir: str = EMAILS_DIR_DEFAULT, retriever_model: Optional[str] = None,
                   label: Optional[str] = None) -> dict[str, dict[str, int]]:
    """Run `query()` over a small question set with the given per-role models and report
    per-model token usage (+ a rough USD estimate). Returns the token tally.

    The two knobs are `plan_model` (Step 2/3 planner) and `answer_model` (Step 2/3 answer).
    Pass the same value for both to reproduce faculty Step 2 (one model everywhere); pass
    different values to reproduce Step 3 (mixed models). The retriever is unchanged.
    (Provided — it relies on the token tracking you implement above.)
    """
    heading = label or (f"{plan_model}" if plan_model == answer_model
                        else f"plan={plan_model}  answer={answer_model}")
    print("\n" + "=" * 78)
    print(f"EXPERIMENT: {heading}")
    print("=" * 78)

    db = build_or_load_db(emails_dir, CHROMA_DIR)
    hybrid = HybridRetriever(emails_dir=emails_dir, db=db, retriever_model=retriever_model)
    agent = ToolUsingAgent(hybrid=hybrid, emails_dir=emails_dir, debug=False,
                           plan_model=plan_model, answer_model=answer_model,
                           retriever_model=retriever_model)
    agent.reset_token_usage()

    for i, q in enumerate(questions, 1):
        answer = agent.query(q)
        one_line = " ".join(answer.split())
        preview = one_line if len(one_line) <= 160 else one_line[:157] + "..."
        print(f"\nQ{i}. {q}\n    -> {preview}")

    print("\nToken usage:")
    agent.print_token_usage()
    usage = agent.get_token_usage()
    print(f"Approx. cost (USD, see disclaimer): ${estimate_cost_usd(usage):.4f}")
    return usage


def _safe_experiment(*args, **kwargs) -> None:
    """Run one model's experiment; if the provider errors, skip it and keep going.
    Free/cheap models on OpenRouter are frequently rate-limited upstream (HTTP 429), so one
    bad model must not crash the whole lab.
    ponytail: catch-all so any model failure is non-fatal; that's the point of the harness."""
    label = kwargs.get("label") or (args[1] if len(args) > 1 else "?")
    try:
        run_experiment(*args, **kwargs)
    except Exception as e:
        print(f"[skipped] {label}: {type(e).__name__} — {e}")
        print("          Free models (e.g. gemma:free) are often rate-limited upstream; retry "
              "later or add your own OpenRouter key: https://openrouter.ai/settings/integrations")


def main() -> None:
    require_api_key()
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    run_expensive = "--expensive" in sys.argv
    emails_dir = args[0] if args else EMAILS_DIR_DEFAULT

    # Build the DB once up front (cheap models reuse the same persisted vectors).
    build_or_load_db(emails_dir, CHROMA_DIR)

    print("\n### Step 2 — one model at a time (cheap models by default) ###")
    for model in CHEAP_MODELS:
        _safe_experiment(EXPERIMENT_QUESTIONS, model, model, emails_dir=emails_dir)

    if run_expensive:
        print("\n!!! --expensive set: running gpt-5.2-pro (~10x cost) on the tiny suite !!!")
        _safe_experiment(EXPERIMENT_QUESTIONS, EXPENSIVE_MODEL, EXPENSIVE_MODEL, emails_dir=emails_dir)
        print("\n### Step 3 — mixed models (strong planner, cheap answer) ###")
        _safe_experiment(EXPERIMENT_QUESTIONS, MIXED_PLANNER, MIXED_ANSWER, emails_dir=emails_dir,
                         label="MIXED: plan=gpt-5.2-pro answer=gpt-4o-mini")
    else:
        print("\n[skipped] gpt-5.2-pro and the mixed run are OPT-IN — re-run with --expensive.")
        print("          (openai/gpt-5.2-pro is ~10x the price of the cheap models.)")


if __name__ == "__main__":
    main()
