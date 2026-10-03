# Lab 1.2 — Keyword Retrieval (BM25) over Company Emails

This lab builds the first real RAG pipeline of the course: a **keyword retriever**.
Instead of dumping the whole email corpus into the prompt, we *retrieve* the few
most relevant emails for each question and let the LLM answer **only** from them.

```
question ──► tokenize ──► BM25 scoring over all emails ──► top-k emails
                                                              │
                                       system prompt + context + question
                                                              │
                                                            LLM ──► grounded answer
```

---

## Folder layout

```
lab-2/
├── Live_Guided_Virtual_Lab_1_2_starter/
│   ├── lab_1_2_keyword_retrieval_starter.py   ← learners fill in 3 TODOs
│   └── detailedEmails/                        ← the corpus (one .txt per email)
└── Live_Guided_Virtual_Lab_1_2_solution/
    ├── lab_1_2_keyword_retrieval_solution.py  ← completed reference implementation
    └── detailedEmails/                        ← same corpus
```

`detailedEmails/` holds ~thousands of plain-text emails from the fictional
*Precision Paperclip Inc.* Filenames follow `mail_<dd>_<mm>_<yy>_<id>.txt`; the
filename is used as the citation label in the context we send to the model.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install python-dotenv langchain-openai langchain-core rank-bm25
echo "OPENROUTER_API_KEY=sk-or-your-key-here" > .env
```

Run from inside either folder:

```bash
python lab_1_2_keyword_retrieval_starter.py            # uses ./detailedEmails
python lab_1_2_keyword_retrieval_solution.py <folder>  # or pass a folder
```

---

## What is already provided (walk through, don't rewrite)

| Piece | Why it matters |
| --- | --- |
| `SYSTEM_PROMPT` | The **grounding rules**: answer only from provided emails, say "I don't know" otherwise. This is what makes the bot refuse the "CEO's home phone number" question. |
| `require_api_key()` | Fails fast with a readable message instead of a `KeyError` deep inside the client. |
| `chat_loop(response)` | Simple REPL. It takes a *function* as an argument, so any retriever can be plugged in. |
| `BaseRetriever` (ABC) | The reusable pattern for Modules 2–3. It owns the LLM client and history; subclasses only implement `retrievedContext(query)`. Vector (2.1) and hybrid (2.2) retrievers subclass the exact same base. |
| `_build_user_message()` | Formats one user turn as `Context (e-mails): ... \n\n Question: ...` — retrieval is injected *per message*, not baked into the system prompt. |
| `query()` vs `queryWHistory()` | `query()` is stateless (one-shot). `queryWHistory()` appends to `self._history` so follow-ups work. Note it `pop()`s the message if the LLM call raises, so a failed call doesn't corrupt the conversation. |
| `Bm25Retriever.__init__` | Loads and sorts the `.txt` files, keeps `self._paths` (names) and `self._contents` (text) **index-aligned**, then builds the BM25 index. This alignment is the key: BM25 returns positions, and we map position → filename/content. |

### Notes on a couple of non-obvious bits

- **`BM25Okapi([...])` is built once, at startup.** Indexing is the expensive
  part; scoring a query afterwards is cheap. Same idea as a vector store, just
  with word statistics instead of embeddings.
- **BM25 in one sentence:** score a document higher when it contains the query's
  rare words often, with a penalty for long documents. No semantics — a query
  word that never appears literally contributes nothing. That's exactly the
  limitation Module 2 (vector search) fixes.
- **`errors="replace"`** when opening files: the corpus contains some non-UTF-8
  bytes; we prefer a stray `�` over a crash.

---

## The three TODOs for learners

### Step 1 — `tokenize(text) -> list[str]`

Turn raw text into the "words" BM25 counts.

```python
return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS]
```

Teaching points:
- **The same tokenizer must be used for documents and queries.** If indexing
  lowercases but the query doesn't, nothing will ever match.
- Lowercasing → case-insensitive matching. The regex strips punctuation so
  `"project."` and `"project"` become the same token.
- Stopwords (`the`, `is`, `of`, …) appear everywhere, carry no signal, and just
  add noise/cost. Removing them sharpens the score.

### Step 2 — `getTopK(query, k) -> list[(filename, content, score)]`

```python
scores = self._bm25.get_scores(tokenize(query))
top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
return [(self._paths[i], self._contents[i], scores[i]) for i in top]
```

Teaching points:
- `get_scores` returns **one score per document, in index order** — it does not
  sort or filter. We sort *indices* (not scores) so we can still recover which
  document each score belongs to.
- Returning the score too is useful for debugging: print it to see whether
  retrieval is confident or grasping at straws.
- `k` (= `TOP_K = 5`) is the classic recall-vs-context-window trade-off.

### Step 3 — `retrievedContext(query) -> str`

```python
results = self.getTopK(query, self._top_k)
return "\n\n---\n\n".join(f"[{name}]\n{content}" for name, content, _ in results)
```

Teaching points:
- This is the method required by `BaseRetriever`; implementing it is what makes
  the whole chat pipeline work.
- The `[filename]` label gives the model a citation handle, and the `---`
  divider tells it where one email ends and the next begins — without it the
  model can blend two unrelated emails into one "fact".

---

## Demo questions

| Question | Expected behaviour |
| --- | --- |
| "What was the status of the government project going into 2015?" | Grounded answer: the project was on hold/paused. |
| "Who was involved in discussions about restarting the government project?" | Names people from the emails (Finance Director, COO, …). |
| "What is the CEO's home phone number?" | **Refuses** — not in the corpus. Demonstrates the system prompt doing its job. |

Good closing demo: ask the same thing with different wording (e.g. "paused" vs
"on hold"). Keyword search is brittle to paraphrase — a perfect motivation for
the vector retriever in Lab 2.1 and the hybrid retriever in Lab 2.2.
