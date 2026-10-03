# Lab 5.1 — An Agentic Retriever That Decides When to Retrieve More

Every retriever so far was a **fixed pipeline**: one query in, top-k e-mails out,
always the same number of steps. This lab makes retrieval a *decision*. The agent
retrieves, looks at what it got, and judges whether that's enough — issuing
follow-up queries and looping until it's satisfied or hits a safety cap.

```text
                     ┌──────── loop back: "not enough yet" ────────┐
                     │                                             │
                     ▼                                             │
   question ──►  RETRIEVE  ─────────────────────────►  ANALYZE ────┘
                     │                                     │
          search with every pending query          LLM reads the goal,
          add hits to email_bodies (deduped)       what was already tried,
          iterations += 1                          and what was found
                                                         │
                                                    done? / 3 rounds?
                                                         │ yes
                                                         ▼
                              every e-mail collected ──► LLM ──► answer
```

**The state is what makes the loop work.** One dict travels around the cycle and
grows on each pass:

```text
              original_query    "problems with the project and how resolved?"
              pending_queries   ─┐ filled by ANALYZE, drained by RETRIEVE
              executed_queries   │ so the agent never repeats a search
              email_bodies       │ accumulates — keyed by filename, deduped
              iterations         │ counts rounds, enforces the cap
              done              ─┘ ANALYZE's verdict, read by the branch

 round 1   iterations 0 → 1   bodies {}  → {5}    done=False  pending=[q2, q3]
 round 2   iterations 1 → 2   bodies {5} → {13}   done=True   pending=[]
                                                    │
                                          all 13 e-mails go to the answer
```

**Three guards make sure the loop always ends:**

```text
  iterations >= MAX   ──┐
  bad JSON from LLM   ──┼──►  done = True  ──►  END
  should_continue     ──┘
```

The loop lives entirely inside `retrievedContext()`, so `BaseRetriever` and the
chat loop are untouched. Only *"what to retrieve"* became a decision.

---

## Folder layout

```text
lab-5.1/
├── Live_Guided_Virtual_Lab_5_1_starter/
│   ├── lab_5_1_agentic_retriever_starter.py    ← learners fill in 1 TODO
│   ├── requirements.txt
│   └── detailedEmails/                         ← 7,894 .txt emails
└── Live_Guided_Virtual_Lab_5_1_solution/
    ├── lab_5_1_agentic_retriever_solution.py
    ├── requirements.txt
    └── detailedEmails/
```

## Setup

```bash
source .venv/bin/activate
pip install -r requirements.txt        # adds langgraph to the usual stack
echo "OPENROUTER_API_KEY=sk-or-your-key-here" > .env
```

```bash
python lab_5_1_agentic_retriever_starter.py             # uses ./detailedEmails
python lab_5_1_agentic_retriever_solution.py <folder>
```

### What a run produces

`DEBUG = True`, so every round of the loop is printed — this is the lab:

```text
Loading existing vector DB from chroma_db/
Chat over the email DB. Type your question; 'exit'/'quit' to stop.

You: What were the main problems with the government project and how were they resolved?
[agentic] retrieving for: 'What were the main problems with the government project and how were they resolved?'
[agentic] done=False reasoning='Found the problems but nothing about resolution or outcome'
[agentic] new queries: ['government project resolution outcome', 'steps taken to restart the government project']
[agentic] retrieving for: 'government project resolution outcome'
[agentic] retrieving for: 'steps taken to restart the government project'
[agentic] done=True reasoning='Retrieved e-mails now cover both the problems and how they were addressed'

Assistant: ...
```

No new files are written; `chroma_db/` is reused from Module 2. The thing to
watch is the **`reasoning` field** — it's the agent narrating why it isn't done
yet, and it's what makes the behaviour legible instead of magical.

> Cost per question is now **variable**: 1 retrieval + 1 analyze call at minimum,
> up to 3 rounds. A simple question stops after one round; a broad one loops.

---

## Core concepts in this lab

- **Agentic retrieval** — the system decides its own next action instead of
  following a fixed script. A researcher who reads what they found, notices a
  gap, and goes back to search again, rather than doing one search and stopping.
- **The retrieve → analyze loop** — act, observe, decide, repeat. This
  sense-act cycle is the core pattern behind essentially every LLM agent.
- **State machine / graph** — LangGraph models the flow as **nodes** (steps) and
  **edges** (transitions), with a *conditional* edge that either loops back or
  ends. The control flow becomes data you can inspect, rather than a `while` loop
  buried in a function.
- **Accumulating state** — `email_bodies` is a dict keyed by filename, so e-mails
  found by several queries are deduplicated for free and the set only ever grows.
- **Termination guarantees** — an LLM deciding "am I done?" could, in principle,
  never say yes. Three independent guards prevent an infinite loop (see below).
- **Bounded reasoning context** — the analyzer only sees the first 20 e-mails,
  even though all of them are returned to the chatbot. The *decision* prompt is
  deliberately cheaper than the *answer* prompt.

### The three stopping guards

Worth naming explicitly — this is the safety-critical part of any agent loop:

| Guard | Where | Catches |
| --- | --- | --- |
| `iterations >= max_iterations` at the top of `analyze_node` | before the LLM call | runaway looping (saves the call entirely) |
| JSON parse failure → `done = True` | in the `except` | a malformed LLM reply |
| `should_continue` re-checks `done` **and** `iterations` | the conditional edge | a node returning inconsistent state |

The defensive default is **stop**, not retry. An agent that fails by doing too
little is far safer than one that fails by looping forever, burning credits.

### Quick comparison to the earlier retrievers

| | Hybrid (2.2) | Multi-step (4.1) | Agentic (5.1) |
| --- | --- | --- | --- |
| Number of searches | 1 | 2–5, fixed up front | **decided at runtime** |
| LLM's role in retrieval | none | decompose once | decide repeatedly |
| Sees its own results? | — | no | **yes — that's the point** |
| LLM calls per question | 1 | 2 | 3–7 |
| Control flow | straight line | fan-out + merge | loop with exit conditions |

Lab 4.1 planned all its queries **before** seeing any results. Lab 5.1 plans the
next query **after** looking at what came back — the difference between a plan and
a strategy.

---

## What is already provided (walk through, don't rewrite)

| Piece | Why it matters |
| --- | --- |
| `SYSTEM_PROMPT`, `chat_loop()`, `BaseRetriever` | Unchanged since Lab 1.2, for the last time. |
| The whole hybrid stack | `tokenize`, `get_embeddings`, `build_or_load_db`, `_normalize`, `HybridRetriever` — Lab 2.2's code, used as the agent's one **tool**. |
| `ANALYZE_SYSTEM` | The decision prompt. Defines the exact JSON contract (`done`, `new_queries`, `reasoning`) and the "don't repeat queries" rule. |
| `_AgentState` (`TypedDict`) | The agent's memory: the original query, pending/executed queries, accumulated e-mails, iteration count, and the done flag. |
| `retrieve_node` | Runs the hybrid retriever for each pending query, merges results into `email_bodies`, clears `pending_queries`, increments `iterations`. |
| `should_continue` | The conditional edge — returns `END` or `"retrieve"`. |
| The graph wiring | `set_entry_point("retrieve")`, `add_edge("retrieve", "analyze")`, `add_conditional_edges("analyze", should_continue)`, then `.compile()`. |
| `retrievedContext()` | Seeds the initial state (with `pending_queries=[query]`), invokes the graph, formats every accumulated e-mail. |

### Notes on a couple of non-obvious bits

- **Nodes return *partial* state, not the whole thing.** `retrieve_node` returns
  only the four keys it changed; LangGraph merges that into the running state.
- **State is copied, not mutated** — `dict(state["email_bodies"])`,
  `list(state["executed_queries"])`. Node functions should be pure; mutating the
  incoming state in place is a classic source of bugs in graph frameworks.
- **`pending_queries` is the hand-off between nodes.** `analyze` fills it,
  `retrieve` drains it. That one field is the loop's entire payload.
- **`executed_queries` exists purely to feed the prompt** — it's what makes the
  "do not repeat queries already executed" rule enforceable. Without it the agent
  cheerfully re-runs the same search every round.
- **The 20-e-mail cap bounds only the decision prompt.** `retrievedContext()`
  still returns `final["email_bodies"]` in full. Easy to misread as a limit on
  the answer context.
- **The `retrieve` node is where a real agent would have many tools.** Here it has
  exactly one (hybrid search). Swapping in a tool-choice step is the natural
  extension to a full agent.

---

## The TODO for learners

### `analyze_node(state) -> dict`

```python
if state["iterations"] >= self._max_iterations:
    return {"done": True, "pending_queries": []}          # stop before spending a call

email_summary = "\n\n---\n\n".join(
    f"[{name}]\n{content}" for name, content in list(state["email_bodies"].items())[:20]
)
executed_str = "\n".join(f"- {q}" for q in state["executed_queries"])
user_content = (
    f"Original question: {state['original_query']}\n\n"
    f"Queries already executed:\n{executed_str}\n\n"
    f"E-mails retrieved so far ({len(state['email_bodies'])} total):\n\n{email_summary}"
)
response = self._llm.invoke([
    SystemMessage(content=ANALYZE_SYSTEM),
    HumanMessage(content=user_content),
])
raw = (response.content if hasattr(response, "content") else str(response)).strip()
if raw.startswith("```"):                                  # strip ```json ... ``` fences
    raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
try:
    result = json.loads(raw)
    done = bool(result.get("done", True))
    new_queries = result.get("new_queries", []) if not done else []
except (json.JSONDecodeError, ValueError):
    done, new_queries = True, []                           # parse failure ⇒ stop
return {"done": done, "pending_queries": new_queries}
```

Teaching points:
- **The iteration check comes first**, before the LLM call — the cap should save
  the API call, not just discard its result.
- The prompt feeds the agent **three things**: the goal (original query), what it
  has already tried (executed queries), and what it has found (e-mails). That
  triple is the general shape of agent context.
- **`.get("done", True)`** defaults to stopping if the key is missing. Every
  default in this function points toward termination.
- `new_queries` is forced to `[]` when `done` is true, so a confused reply saying
  "done, and here are more queries" can't restart the loop.
- The fence-stripping is the same free-text-to-data problem as Lab 4.1's
  decomposition and Lab 3.2's line splitting — third variation on one theme.
- **The `reasoning` field is never used by the code**, only printed. It costs a
  few tokens and buys you an explanation of every decision — cheap observability.

---

## Demo questions

| Question | What to watch |
| --- | --- |
| "What were the main problems with the government project and how were they resolved?" | Should loop: finds problems first, then queries for the resolution |
| "Give me the full picture of the budget concerns raised across the company." | Broad — usually 2–3 rounds, widening each time |
| "What was the status of the government project going into 2015?" | Narrow — should stop after one round |
| "What is the CEO's home phone number?" | Not in the corpus; watch it search, fail to find, and still refuse |

**The closing demo:** run the third question and the first back to back. Same
code, different number of retrieval rounds — the agent adapted its effort to the
question. Then set `MAX_ITERATIONS = 1` and re-run the first: it degrades exactly
to Lab 2.2's single-shot hybrid retriever, which is a neat way to show that all
five retrievers in this course sit on one continuum.

A fair closing caveat: agentic retrieval costs several times more per question
and can talk itself into unnecessary rounds. It earns its keep on open-ended
questions — not on every query.
