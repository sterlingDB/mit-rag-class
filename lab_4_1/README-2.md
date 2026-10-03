# Lab 4.1 — Multi-step Retrieval by Query Decomposition

Every retriever so far treated the question as **one** search. But "What was the
government project, why was it paused, and who decided to revisit it?" is really
three questions, and no single query ranks all three answers highly. This lab has
the LLM split the question into focused sub-queries, retrieves for each, and
merges the results.

```text
question ──► LLM ("split this") ──► 2-5 sub-queries
                                          │
                        ┌─────────────────┼─────────────────┐
                     sub-q 1           sub-q 2           sub-q 3
                        │                 │                 │
              hybrid.getTopK(3k)  hybrid.getTopK(3k)  hybrid.getTopK(3k)
                        └─────────────────┼─────────────────┘
                                          │
                     merge: sum each e-mail's score across sub-queries
                                          │
                                   top-k e-mails ──► LLM ──► answer
```

---

## Folder layout

```text
lab-4.1/
├── Live_Guided_Virtual_Lab_4_1_starter/
│   ├── lab_4_1_multistep_retrieval_starter.py   ← learners fill in 2 TODOs
│   └── detailedEmails/                          ← 7,894 .txt emails
└── Live_Guided_Virtual_Lab_4_1_solution/
    ├── lab_4_1_multistep_retrieval_solution.py
    └── detailedEmails/
```

## Setup

```bash
source .venv/bin/activate
pip install python-dotenv langchain-openai langchain-core langchain-chroma rank-bm25
echo "OPENROUTER_API_KEY=sk-or-your-key-here" > .env
```

```bash
python lab_4_1_multistep_retrieval_starter.py             # uses ./detailedEmails
python lab_4_1_multistep_retrieval_solution.py <folder>
```

### What a run produces

`DEBUG = True` prints the decomposition before every answer — the whole point of
the lab is watching this happen:

```text
Loading existing vector DB from chroma_db/
Chat over the email DB. Type your question; 'exit'/'quit' to stop.

You: What was the government project, why was it paused, and who decided to revisit it?
[decompose] 'What was the government project, why was it paused, and who decided to revisit it?' ->
   1. details of the government project scope and purpose
   2. reasons the government project was put on hold
   3. who proposed restarting or revisiting the government project

Assistant: ...
```

No new files are written — this lab reuses `chroma_db/` from Module 2. Copy the
folder over to skip re-embedding 7,894 emails.

> Each question now costs **3 LLM calls minimum**: one to decompose, one to
> answer, plus the retrieval work per sub-query. Noticeably slower per turn than
> Lab 2.2.

---

## Core concepts in this lab

- **Query decomposition** — using the LLM to *rewrite the search*, not to answer.
  A researcher handed a broad question doesn't type it into the catalogue
  verbatim; they break it into several specific lookups.
- **The LLM inside the retrieval loop** — until now the model sat at the end of
  the pipeline consuming context. Here it also sits at the front, shaping what
  gets retrieved.
- **Score accumulation as evidence-gathering** — an email retrieved by *several*
  sub-queries sums a higher total, so documents relevant to multiple facets of
  the question float up. Voting, where an email that several sub-queries all
  consider relevant beats one that a single sub-query loved.
- **Over-fetch then narrow** — each sub-query pulls `3 * k` candidates so the
  merge has room to work; the final list is still `k`. Same widen-then-narrow
  shape as Lab 2.2's candidate pool.
- **Graceful degradation** — if the decomposition fails to parse, the code falls
  back to `[query]` and the whole thing quietly becomes plain hybrid retrieval.
  An extra pipeline stage is an extra thing that can break.

### Quick comparison to Lab 2.2

| | Hybrid (2.2) | Multi-step (4.1) |
| --- | --- | --- |
| Searches per question | 1 | 2–5 |
| LLM calls per question | 1 (answer) | 2+ (decompose, then answer) |
| Merges | two retrievers, one query | one retriever, many queries |
| Score combination | weighted sum, normalized | plain sum across sub-queries |
| Best for | any single-intent question | multi-part questions |

Lab 2.2 asked two specialists the same question. Lab 4.1 asks one specialist
several different questions and pools what comes back.

---

## What is already provided (walk through, don't rewrite)

| Piece | Why it matters |
| --- | --- |
| `SYSTEM_PROMPT`, `chat_loop()`, `BaseRetriever` | Unchanged since Lab 1.2. |
| The whole hybrid stack | `tokenize`, `get_embeddings`, `build_or_load_db`, `_normalize`, `HybridRetriever` — copied from Lab 2.2 and used here as a **component**, not modified. |
| `DECOMPOSE_SYSTEM` | The decomposition prompt: 2–5 sub-queries, each a distinct aspect, phrased as searches, returned as a JSON array. |
| `MultiStepRetriever.__init__` | Takes a `HybridRetriever` as a constructor argument rather than subclassing it. |
| `retrievedContext()` | Same `[filename]` + `---` formatting as every earlier lab. |

### Notes on a couple of non-obvious bits

- **Composition, not inheritance.** `MultiStepRetriever` *has a* `HybridRetriever`
  and calls its `getTopK()`. This is why Lab 2.2's code needed no changes — a new
  strategy wraps the old one.
- **Two different system prompts in one class.** `DECOMPOSE_SYSTEM` is used for
  the rewrite call; `SYSTEM_PROMPT` (from the base class) for the answer call. The
  same `self._llm` client serves both.
- **Scores are summed raw across sub-queries, not re-normalized.** Each sub-query
  returns scores already normalized to `[0,1]` by the hybrid fusion, so summing
  four of them can exceed 1.0. That's intentional — the total *is* the vote count.
- **`DEBUG = True` by default**, which is unusual for shipped code but deliberate
  here: the decomposition is the lesson.

---

## The two TODOs for learners

### Step 1 — `_decompose_query(query) -> list[str]`

```python
messages = [SystemMessage(content=DECOMPOSE_SYSTEM), HumanMessage(content=query)]
response = self._llm.invoke(messages)
raw = (response.content if hasattr(response, "content") else str(response)).strip()
if raw.startswith("```"):                       # tolerate ```json ... ``` fences
    raw = raw.strip("`")
    raw = raw[raw.find("["): raw.rfind("]") + 1] if "[" in raw else raw
try:
    parsed = json.loads(raw)
    if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed) and parsed:
        return parsed
except (json.JSONDecodeError, ValueError):
    pass
return [query]                                   # fallback: behave like plain hybrid
```

Teaching points:
- **Asking for JSON is not the same as receiving JSON.** Models wrap output in
  code fences, add preamble, or return a bare string. Every LLM call that feeds
  *code* rather than a human needs this kind of defensive parsing.
- The three validation checks — is it a list, are all elements strings, is it
  non-empty — each guard a real failure mode. A `[]` would retrieve nothing.
- The `return [query]` fallback is the design decision worth dwelling on: a
  failure degrades to Lab 2.2 behaviour rather than raising. The user gets a
  slightly worse answer instead of an error.
- Compare with Lab 3.2's line-splitting parser — same problem (free text → data),
  different format, same lesson.

### Step 2 — `getTopK(query, k)`

```python
sub_queries = self._decompose_query(query)
content_map: dict[str, str] = {}
score_map: dict[str, float] = {}
for sq in sub_queries:
    for name, content, score in self._hybrid.getTopK(sq, 3 * k):
        content_map[name] = content
        score_map[name] = score_map.get(name, 0.0) + score   # accumulate
top = sorted(score_map.items(), key=lambda kv: kv[1], reverse=True)[:k]
return [(name, content_map[name], score) for name, score in top]
```

Teaching points:
- `score_map.get(name, 0.0) + score` is the entire merge strategy: **first
  appearance starts at its score, every later appearance adds to it.**
- Filename is again the join key, exactly as in Lab 2.2's fusion.
- `3 * k` widens each sub-query's pool. With `k` alone, the sub-queries would
  rarely overlap and the sum would never accumulate.
- Two dicts, two jobs: `content_map` deduplicates the text, `score_map`
  accumulates the votes. Keeping them separate avoids re-storing the body on
  every hit.
- An email retrieved once with a high score can still beat one retrieved twice
  with low scores — the sum, not the count, decides.

---

## Demo questions

| Question | What to watch |
| --- | --- |
| "What was the government project, why was it paused, and who decided to revisit it?" | Splits into ~3 sub-queries; each supplies emails the others miss |
| "Summarize the budget concerns and the people raising them." | Two facets: the concerns, and the people |
| "What was the status of the government project going into 2015?" | Single-intent — decomposition adds little; a fair look at the cost |

Run the third one to keep it honest: multi-step retrieval costs an extra LLM call
on *every* question, and only pays off on the multi-part ones. Choosing when the
complexity is worth it is the real takeaway.
