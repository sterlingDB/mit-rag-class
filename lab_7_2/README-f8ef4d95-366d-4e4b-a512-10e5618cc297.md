# Lab 7.2 — Optimizing Performance and Cost Under Load

A system can be correct and secure and still fail in production by being too slow or
too expensive. Lab 6.2 **measured** where the tokens go. This lab acts on that
measurement with two optimizations, then measures again to prove the saving — and
to check that the cheaper answers are still *good* answers.

The two facts from 6.2 that drive the whole design:

1. The full agent spends most of its tokens on the **PLAN loop**.
2. Only a minority of questions actually need that loop.

```text
                  ┌─────────────────────────────────────────────┐
  question ──────►│ 1. SEMANTIC CACHE   seen this before?        │
                  └─────────────────────────────────────────────┘
                        │ hit (+ validation)        │ miss
                        ▼                           ▼
                   cached answer        ┌─────────────────────────┐
                   (~free)              │ 2. ROUTER (gpt-5-nano)  │
                                        └─────────────────────────┘
                                          │ "direct"      │ "agent"
                                          ▼               │
                              1 retrieval + 1 small       │
                              model answer                │
                                          │               │
                                   satisfactory?          │
                                    │ yes    │ no         │
                                    ▼        └────────────┤
                                 return                   ▼
                                              FULL agent on gpt-5.4
                                              (plan → tools → answer)
```

**Three paths, three price points.** Cache hit is near-free; the direct path is one
cheap call; the full agent is the expensive fallback that only hard questions reach.

---

## Folder layout

```text
lab-7.2/
├── lab_7_2_efficient_agent_solution.py          ← completed reference
└── Live_Guided_Virtual_Lab_7_2_starter/
    ├── lab_7_2_efficient_agent_starter.py       ← 3 TODOs
    ├── requirements.txt
    └── detailedEmails/                          ← 7,894 .txt emails
```

## Setup

```bash
source .venv/bin/activate
pip install -r requirements.txt
echo "OPENROUTER_API_KEY=sk-or-your-key-here" > .env
```

```bash
python lab_7_2_efficient_agent_solution.py detailedEmails
```

### What a run produces

The script runs the same 4 questions twice — baseline, then optimized:

```text
### Step 3 — measure: un-optimized baseline vs optimized (cache + router) ###

==============================================================================
EXPERIMENT: BASELINE — un-optimized full agent (no cache, no router)
==============================================================================
Q1. What e-mails were sent on 01/01/2014?
    -> On 01/01/2014 six e-mails were sent, covering ...
...
Token usage:
openai/gpt-5.4-mini: 61204 input, 1486 output, 0 cached
Approx. cost (USD, see disclaimer): $0.0134

==============================================================================
EXPERIMENT: OPTIMIZED — semantic cache + query router
==============================================================================
...
Token usage:
openai/gpt-5-nano: 3120 input, 142 output, 0 cached
openai/gpt-5.4-mini: 24880 input, 760 output, 0 cached
Approx. cost (USD, see disclaimer): $0.0062
```

Two folders are written: `chroma_db/` (the corpus index) and **`semantic_cache/`**
(the question→answer cache).

> **The cache persists between runs.** Delete `semantic_cache/` to measure a cold
> cache. Running twice *without* deleting is the better experiment: the second run
> should hit the cache and drop sharply.

---

## Core concepts in this lab

- **Semantic caching.** Not a key-value cache — a *vector* cache. "Who is the CEO?"
  and "Who runs the company?" are different strings but near-identical embeddings,
  so one answer serves both.
- **A high threshold, deliberately.** `CACHE_SIMILARITY_THRESHOLD = 0.85`, far above
  a retrieval threshold, because **a wrong cached answer is worse than a cache miss.**
  Retrieval can afford loose matches; caching can't.
- **Query routing.** Match the model to the difficulty of the question. A near-free
  `gpt-5-nano` call decides whether an expensive path is warranted — a cheap gate in
  front of a costly resource.
- **Escalation with a safety net.** The direct path's answer is *checked*; if it's
  unsatisfactory, the full agent reruns on a stronger model. You get cheap-when-easy
  without accepting cheap-when-hard.
- **Spend where it changes the answer.** The unifying idea of Module 6–7: not one
  clever trick, but routing expense to the steps that actually need it.
- **Cheaper is not automatically equivalent.** The file says this loudly, and it's the
  real lesson — compare the *answers* question by question, not just the two cost lines.

### The three paths priced

| Path | What runs | Relative cost |
| --- | --- | --- |
| **Cache hit** | 1 embedding + 1 nano validation | ~free |
| **Direct** | 1 router + 1 retrieval + 1 mini answer + 1 nano satisfaction check | low |
| **Agent** | full plan→tool→plan loop, `gpt-5.4` on fallback | high |

Note the direct path still pays for *two* nano calls (route + satisfaction). That's
the overhead optimization buys you — worth it only because it diverts most traffic
away from the loop.

### Two guardrails worth knowing about

Both were added after real failures in testing, and both are good discussion material:

- **Exact-date questions always route `agent`.** The direct path has no `by_date`
  tool, so semantic search would return *near*-date e-mails — confidently wrong.
  The router prompt names this case explicitly.
- **The satisfaction check sees the retrieved e-mails.** Otherwise it can only judge
  whether an answer *sounds* complete. With the evidence in hand it can catch an
  answer that omits entities the e-mails actually contain.

### Quick comparison to Lab 6.2

| | Lab 6.2 | Lab 7.2 |
| --- | --- | --- |
| Activity | measure | optimize, then re-measure |
| Lever | per-role models | caching + routing + escalation |
| Models in play | 1–2 | 3 (nano router, mini worker, 5.4 fallback) |
| Paths per question | 1 | 3 |
| New state | none | a persistent vector cache |
| Risk introduced | none | **wrong cached answer, or a mis-routed question** |

6.2 was free to run and changed nothing. 7.2 changes behaviour, and that's why it
ends with "compare the answers, not just the prices."

---

## What is already provided (walk through, don't rewrite)

| Piece | Why it matters |
| --- | --- |
| The Lab 6.2 agent | Graph, nodes, tools, per-model token tracking — unchanged. |
| `_ROUTER_SYSTEM` | The routing policy, including the exact-date carve-out. |
| `_CACHE_VALIDATE_SYSTEM` | Rejects context-dependent questions ("What is her role?") before trusting a cache hit. |
| `_SATISFACTION_SYSTEM` | Judges usefulness *and* completeness against the retrieved e-mails. |
| `_SemanticCache.__init__` | A `chromadb.PersistentClient` collection configured with `hnsw:space = cosine`. |
| `_validate_cached_answer` / `_check_answer_satisfactory` | The two nano-model gates. |
| `_direct_retrieve_and_answer` | The cheap path: one `getTopK` + one LLM call. |
| Dual graphs | `self._graph` (small models) and `self._fallback_graph` (gpt-5.4) are built separately in `__init__`. |
| `run_experiment(use_cache=, use_router=)` | Both flags `False` reproduces the Lab 6.2 baseline exactly. |

### Notes on a couple of non-obvious bits

- **`use_cache=False, use_router=False` *is* the Lab 6.2 agent.** The baseline isn't a
  separate implementation — it's the same class with the optimizations switched off,
  which is what makes the comparison fair.
- **The cache reuses the retrieval embedding model** (`get_embeddings()`, including
  the OpenRouter `check_embedding_ctx_length=False` flag) but at a very different
  threshold. Same tool, different risk tolerance.
- **`distance = 1 - similarity`** because the collection is cosine — the same
  inversion from Lab 2.1, resurfacing here.
- **Eviction is oldest-first by `timestamp`**, not least-recently-used. Simple, and
  a reasonable thing to question in discussion.
- **The cache validation runs on *every* hit.** The docstring notes it could run only
  the first time; it's kept explicit for teaching, and it's a real per-hit cost.
- **The router defaults to `"agent"` on a parse failure** — the safe, more expensive
  path. Same fail-safe philosophy as 5.1 and 7.1, pointed the other way: here the
  conservative default costs money rather than saving it.
- **A warm cache makes repeat runs look near-free.** That's not cheating, but you
  must know which state you measured.

---

## The three TODOs for learners

### Step 1a — `_SemanticCache.lookup(question, threshold)`

```python
count = self._col.count()
if count == 0:
    return None
embedding = self._embedder.embed_query(question)
results = self._col.query(query_embeddings=[embedding], n_results=1,
                          include=["metadatas", "distances"])
if not results["ids"][0]:
    return None
similarity = 1.0 - results["distances"][0][0]     # cosine: distance = 1 - similarity
if similarity < threshold:
    return None
return results["metadatas"][0][0]["answer"], similarity
```

Teaching points:
- This is Lab 2.1's vector search applied to a completely different problem — same
  mechanism, caching instead of retrieval.
- The threshold is the whole design decision. Too low and users get other people's
  answers; too high and the cache never pays for itself.
- Returning the similarity alongside the answer makes hits debuggable.

### Step 1b — `_SemanticCache.store(question, answer)`

```python
embedding = self._embedder.embed_query(question)
if self._col.count() >= self._max_size:
    all_items = self._col.get(include=["metadatas"])
    oldest_id = min(zip(all_items["ids"], all_items["metadatas"]),
                    key=lambda x: x[1].get("timestamp", 0.0))[0]
    self._col.delete(ids=[oldest_id])
self._col.add(ids=[str(uuid.uuid4())], embeddings=[embedding],
              metadatas=[{"answer": answer, "timestamp": time.time(),
                          "question": question[:500]}])
```

Teaching points:
- **Bounded, or it grows forever.** Any production cache needs an eviction policy;
  this one is the simplest that works.
- `timestamp` exists *only* to make eviction possible — the metadata is carrying
  policy, not just data.
- `question[:500]` caps metadata size and is kept mainly for debugging output.

### Step 2 — `_route_query(question)`

```python
response = self._router_llm.invoke([
    SystemMessage(content=_ROUTER_SYSTEM),
    HumanMessage(content=f"Question: {question}"),
])
self._track_usage(response, self._router_model)
return self._parse_json_response(response).get("route", "agent")
```

Teaching points:
- **Three lines, and it halves the bill** on typical traffic. The leverage is in the
  prompt and the model choice, not the code.
- `_track_usage` is called here too — the router's own cost must appear in the tally,
  or the measurement lies about the savings.
- `.get("route", "agent")` is the fail-safe: unparseable means take the expensive,
  correct path.
- Compare with Lab 5.2's planner: same JSON-decision shape, but this one is a
  *binary* decision on the cheapest available model.

---

## Demo flow

1. **Delete `semantic_cache/`** and run. Compare the baseline and optimized cost
   lines — and read the answers side by side.
2. **Run again without deleting.** The second run should hit the cache on the
   repeated questions and drop sharply. This is the stronger validation.
3. Ask a reworded version of a cached question ("What does the company make?" after
   "What products does Precision Paperclip Inc. make?") to show semantic — not
   literal — cache matching.
4. Ask the date question and watch the router send it to `agent` rather than the
   cheap path. Explain why: no `by_date` tool on the direct path.
5. Find a question where the two paths disagree, and verify against the e-mails.

Closing point: this is the final lab, and it's the one that makes the others
affordable. Caching, routing, and escalation are the difference between a pipeline
that works in a demo and one that survives real traffic — provided you keep checking
that cheaper still means correct.
