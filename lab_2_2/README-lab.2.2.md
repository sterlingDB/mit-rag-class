# Lab 2.2 — Hybrid Retrieval (BM25 + Vectors) over Company Emails

Lab 1.2 matched **words**. Lab 2.1 matched **meaning**. Each wins on questions the
other fumbles. This lab runs both retrievers on every query and *fuses* their
rankings into one list, so exact terms and paraphrases are both covered.

```
                    ┌─► BM25 index    ──► 10 candidates (higher = better) ─┐
question ──► ───────┤                                                      ├─► normalize
                    └─► chroma_db/    ──► 10 candidates (lower  = better) ─┘   + weighted sum
                                                                                    │
                                                                          top-4 emails
                                                                                    │
                                                     system prompt + context + question
                                                                                    │
                                                                       LLM ──► grounded answer
```

---

## Folder layout

```
lab-2.2/
├── Live_Guided_Virtual_Lab_2_2_starter/
│   ├── lab_2_2_hybrid_retrieval_starter.py    ← learners fill in 2 TODOs
│   └── detailedEmails/                        ← the corpus (one .txt per email)
└── Live_Guided_Virtual_Lab_2_2_solution/
    ├── lab_2_2_hybrid_retrieval_solution.py   ← completed reference implementation
    └── detailedEmails/                        ← same corpus
```

Same *Precision Paperclip Inc.* corpus as Labs 1.2 and 2.1, so all three
retrievers can be compared question-for-question.

## Setup

```bash
source .venv/bin/activate
pip install python-dotenv langchain-openai langchain-core langchain-chroma rank-bm25
echo "OPENROUTER_API_KEY=sk-or-your-key-here" > .env
```

Run from inside either folder:

```bash
python lab_2_2_hybrid_retrieval_starter.py             # uses ./detailedEmails
python lab_2_2_hybrid_retrieval_solution.py <folder>   # or pass a folder
```

### What the first run produces

```
Building vector DB (first run — embedding the emails)...
  Indexed <N> emails into chroma_db/
Hybrid retriever ready over <N> emails.
Chat over the email DB. Type your question; 'exit'/'quit' to stop.

You: _
```

Two indexes now exist side by side:

| Index | Where it lives | Built when |
| --- | --- | --- |
| Vector (Chroma) | `chroma_db/` on disk (~70 MB) | once; reloaded instantly afterwards |
| BM25 | in memory only | every startup, from the `.txt` files |

The BM25 line appears on *every* run because word statistics are cheap to
recompute; only the embeddings are worth persisting. Copying `chroma_db/` over
from Lab 2.1 saves re-embedding the corpus.

---

## Core concepts in this lab

- **Hybrid retrieval / fusion** — run both retrievers, merge the results. Two
  specialists reviewing the same shortlist: one checks the exact wording, the
  other checks the gist, and you trust a document more when both like it.
- **Candidate pool** — each side proposes `CANDIDATE_POOL = 10` documents, but
  only `NUM_RETRIEVED = 4` reach the LLM. Fusing a wider pool than you keep is
  what gives the merge something to actually re-rank.
- **The scale problem** — BM25 returns unbounded scores where higher is better
  (`8.3` is a strong match); Chroma returns a distance where lower is better
  (`0.21` is a strong match). Adding them raw is meaningless — like summing
  degrees Celsius and wind speed.
- **Min-max normalization** — rescale each list to `[0, 1]` so the best candidate
  becomes 1.0 and the worst 0.0. Now both sides speak the same language.
- **Inversion** — the vector side is normalized *and flipped* (`1 - n`) so that
  small distance becomes a high score, matching BM25's direction.
- **Weights** — `WEIGHT_BM25` and `WEIGHT_VECTOR` (both 0.5) are the dial between
  literal and semantic matching. They're the one knob worth experimenting with.

### The fusion, on a worked example

Query: *"who signs off on spending?"*

| Email | BM25 raw | BM25 norm | Vec dist | Vec norm | Fused (0.5/0.5) |
| --- | --- | --- | --- | --- | --- |
| `mail_A` "budget approval pending" | 0.0 (not in pool) | — → 0.0 | 0.18 | 1.00 | **0.50** |
| `mail_B` "please sign off on the spending report" | 9.1 | 1.00 | 0.44 | 0.20 | **0.60** |
| `mail_C` "signature required" | 4.2 | 0.46 | 0.31 | 0.62 | **0.54** |

`mail_B` wins on words alone, `mail_A` on meaning alone — but `mail_C`, merely
decent at both, outranks `mail_A`. That's fusion working: agreement between two
independent signals beats a single strong opinion.

### Quick comparison across the three labs

| | BM25 (1.2) | Vectors (2.1) | Hybrid (2.2) |
| --- | --- | --- | --- |
| Matches on | exact words | meaning | both |
| Score direction | higher = better | lower = better | higher = better (normalized) |
| Strong at | exact IDs, rare terms | paraphrases, synonyms | mixed queries |
| Cost per query | ~free | 1 embedding call | 1 embedding call |

Lab 1.2 was `Ctrl+F`; Lab 2.1 was the librarian who has read everything. Hybrid
asks both and prefers the documents they agree on.

---

## What is already provided (walk through, don't rewrite)

| Piece | Why it matters |
| --- | --- |
| `SYSTEM_PROMPT`, `require_api_key()`, `chat_loop()`, `BaseRetriever` | Unchanged for the third lab running. Only `retrievedContext()` ever differs. |
| `tokenize()` | Lifted verbatim from Lab 1.2 — the same tokenizer must serve indexing and querying. |
| `get_embeddings()`, `build_or_load_db()` | Lifted verbatim from Lab 2.1, including the load-if-exists cache branch. |
| `HybridRetriever.__init__` | Builds the BM25 index over `self._paths` / `self._contents` **and** holds the Chroma handle. Both indexes cover the same files. |
| `_bm25_topk()` / `_vector_topk()` | The two labs' `getTopK()` methods, renamed as private helpers. Both return `(filename, content, score)` — the shared shape is what makes fusion possible. |
| `retrievedContext()` | Identical to both earlier labs: `[filename]` labels and `---` dividers. |

### Notes on a couple of non-obvious bits

- **Filename is the join key.** The two retrievers return unrelated objects; the
  only thing tying a BM25 hit to a vector hit is `metadata["source"]` matching
  the basename. That's why the vector store had to carry the filename.
- **The two scores in `_bm25_topk` and `_vector_topk` point in opposite
  directions** even though both functions have the same signature. The type
  hints don't warn you — this is the bug to expect if fusion looks random.
- **Candidate pools overlap only partly.** A document found by one retriever and
  not the other still gets scored; it simply contributes `0.0` from the missing
  side.

---

## The two TODOs for learners

### Step 1 — `_normalize(scores, invert=False) -> list[float]`

```python
if not scores:
    return []
lo, hi = min(scores), max(scores)
if hi == lo:
    return [0.5] * len(scores)          # all equal → neutral
norm = [(s - lo) / (hi - lo) for s in scores]
return [1.0 - n for n in norm] if invert else norm
```

Teaching points:
- Min-max always maps the best candidate to 1.0 and the worst to 0.0 — the two
  scales become comparable regardless of their original units.
- The `hi == lo` guard prevents a divide-by-zero. It happens for real: a query
  whose words appear in no email gives BM25 all-zero scores.
- `invert=True` is used **only** for the vector distances. Getting this backwards
  silently returns the *least* relevant emails, which is a great thing to
  demonstrate on purpose.
- Normalization is *relative to the pool*, not absolute. A pool of ten terrible
  matches still produces a 1.0 — normalized scores rank, they don't measure
  quality.

### Step 2 — `getTopK(query, k)`

```python
bm  = self._bm25_topk(query, CANDIDATE_POOL)
vec = self._vector_topk(query, CANDIDATE_POOL)

content_by_name, bm_norm, vec_norm = {}, {}, {}
if bm:
    for (name, content, _), val in zip(bm, _normalize([s for _, _, s in bm])):
        content_by_name[name] = content
        bm_norm[name] = val
if vec:
    for (name, content, _), val in zip(vec, _normalize([d for _, _, d in vec], invert=True)):
        content_by_name[name] = content
        vec_norm[name] = val

fused = [
    (name, content, WEIGHT_BM25 * bm_norm.get(name, 0.0) + WEIGHT_VECTOR * vec_norm.get(name, 0.0))
    for name, content in content_by_name.items()
]
fused.sort(key=lambda t: t[2], reverse=True)
return fused[:k]
```

Teaching points:
- `content_by_name` is a **union** of both pools keyed by filename, which is what
  de-duplicates a document found by both retrievers.
- `.get(name, 0.0)` encodes "missing from this retriever scores zero" — so a
  document must be genuinely strong on one side to survive on that alone.
- The pool is 10 per side but only `k = 4` are returned: retrieve wide, fuse,
  then narrow.
- Swapping `WEIGHT_BM25 = 1.0 / WEIGHT_VECTOR = 0.0` reduces this to Lab 1.2, and
  the reverse gives Lab 2.1 — a quick way to show the dial in action.

---

## Demo questions

| Question | Expected behaviour |
| --- | --- |
| "What was the status of the government project going into 2015?" | Grounded answer: the project was on hold/paused. |
| "Who was involved in discussions about restarting the government project?" | Names people from the emails (Finance Director, COO, …). |
| "What is the CEO's home phone number?" | **Refuses** — not in the corpus. |

**The closing demo:** take the paraphrase that broke BM25 in Lab 1.2 *and* the
exact identifier (order number, product code) that the embeddings blurred in Lab
2.1, and run both here. Hybrid should handle each — the point being that
production RAG systems rarely pick one retriever.

Then adjust the weights to 0.9/0.1 and back to 0.1/0.9 on the same question to
show the ranking shift. There is no universally correct setting; it depends on
whether your users search by phrasing or by identifier.
