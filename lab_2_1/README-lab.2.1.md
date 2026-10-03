# Lab 2.1 — Vector Retrieval (Chroma + Embeddings) over Company Emails

This lab swaps the *word matching* of Lab 1.2 for **meaning matching**. Each email
is turned into an embedding vector once, stored in a Chroma database, and every
question is embedded the same way so we can retrieve the emails that are
semantically *nearest* — even when they share no literal words with the query.

```
                        ┌── one-time build ──────────────────────────┐
emails ──► Document ──► │ embedding model ──► vectors ──► chroma_db/ │
                        └────────────────────────────────────────────┘

question ──► embed ──► nearest-neighbour search ──► top-k emails
                                                        │
                                 system prompt + context + question
                                                        │
                                                      LLM ──► grounded answer
```

---

## Folder layout

```
lab-3/
├── Live_Guided_Virtual_Lab_2_1_starter/
│   ├── lab_2_1_vector_retrieval_starter.py    ← learners fill in 3 TODOs
│   └── detailedEmails/                        ← the corpus (one .txt per email)
└── Live_Guided_Virtual_Lab_2_1_solution/
    ├── lab_2_1_vector_retrieval_solution.py   ← completed reference implementation
    └── detailedEmails/                        ← same corpus
```

Same *Precision Paperclip Inc.* corpus as Lab 1.2, so the two retrievers can be
compared question-for-question. A `chroma_db/` folder appears next to the script
on first run — that's the persisted index, and deleting it forces a rebuild.

## Setup

```bash
source .venv/bin/activate
pip install python-dotenv langchain-openai langchain-core langchain-chroma rank-bm25
echo "OPENROUTER_API_KEY=sk-or-your-key-here" > .env
```

Run from inside either folder:

```bash
python lab_2_1_vector_retrieval_starter.py             # uses ./detailedEmails
python lab_2_1_vector_retrieval_solution.py <folder>   # or pass a folder
```


### What the first run produces

```
Building vector DB (first run — embedding the emails)...
  Indexed <N> emails into chroma_db/
Chat over the email DB. Type your question; 'exit'/'quit' to stop.

You: _
```

It creates a `chroma_db/` folder next to the script:

```
chroma_db/
├── chroma.sqlite3            ← documents, metadata, ids  (~70 MB here)
└── <uuid>/                   ← the HNSW nearest-neighbour index
    ├── data_level0.bin       ← the actual vectors  (~50 MB here)
    ├── header.bin  length.bin  link_lists.bin
    └── index_metadata.pickle
```

Two things worth pointing out live: the index is **much larger than the emails
themselves** (that's the cost of storing a vector per document), and running the
script a second time prints `Loading existing vector DB from chroma_db/` and
starts instantly — no embedding calls. Delete the folder to force a rebuild.

---

## Core concepts in this lab

- **Embeddings** — each email becomes a vector encoding its *meaning*. Think of it
  as a coordinate on a giant map: budget emails cluster in one neighbourhood,
  shipping-delay emails in another. Ideas decide the location, not words.
- **Vector search** — the question gets a coordinate on the same map; we take the
  4 nearest emails. That's why "who signs off on spending?" can find an email
  that only says "budget approval".
- **Distance, not score** — Chroma returns how *far* apart things are, so **lower
  is better** — the opposite of BM25. Easy trap, and the one that matters in 2.2.
- **Index once, query many** — embedding costs API calls, so it's persisted to
  `chroma_db/`; later runs are just fast lookups.
- **One shared vector space** — the query must be embedded by the *same* model as
  the documents, or the coordinates are meaningless: two people, two maps.

### Text → vector, concretely

`text-embedding-3-small` maps any string to 1536 numbers:

```
"budget approval is pending"  ──►  [ 0.021, -0.114,  0.087, ...,  0.003 ]
"who signs off on spending?"  ──►  [ 0.019, -0.109,  0.091, ...,  0.005 ]
"the shipment arrives Friday" ──►  [-0.132,  0.076, -0.044, ..., -0.061 ]
```

The first two share **no content words**, yet their vectors are nearly identical —
BM25 scores them 0 against each other; vector search ranks them as a top match.

The classic demonstration that the space encodes *meaning* rather than spelling
is that you can do arithmetic on it:

```
vector("king") − vector("man") + vector("woman")  ≈  vector("queen")
```

The direction "man → woman" is the same direction as "king → queen", so gender
lives as a consistent offset in the space. Same for "Paris − France + Italy ≈
Rome". Nothing about the letters `k-i-n-g` produces this; it falls out of the
model having seen how the words are used.

### How "nearest" is decided

The default metric in Chroma is **cosine distance** — the angle between two
vectors, ignoring their length:

```
cosine_similarity(a, b) = (a · b) / (|a| · |b|)      → 1.0 identical, 0.0 unrelated
cosine_distance        = 1 − cosine_similarity        → 0.0 identical, 1.0 unrelated
```

A tiny two-dimensional version, where the axes are "money-ish" and "logistics-ish":

| Text | Vector | Distance to query |
| --- | --- | --- |
| *query:* "who signs off on spending?" | `[0.9, 0.1]` | — |
| "budget approval is pending" | `[0.8, 0.2]` | **0.02** ← nearest |
| "invoice needs a signature" | `[0.7, 0.4]` | 0.10 |
| "the shipment arrives Friday" | `[0.1, 0.9]` | 0.69 |

Angle rather than distance-in-a-straight-line is what makes length irrelevant: a
two-line email and a twenty-line email about the same topic point the same way,
so the long one isn't penalised just for being long.

Scanning every vector would be slow, so Chroma builds an **HNSW** index (those
`.bin` files) — a graph that hops toward the nearest neighbours instead of
comparing against all of them. It's approximate, and fast.

### Quick comparison to Lab 1.2

| | BM25 (1.2) | Vectors (2.1) |
| --- | --- | --- |
| Matches on | exact words | meaning |
| Index cost | free, instant | API calls, persisted |
| Score | higher = better | lower = better (distance) |
| Strong at | exact IDs, rare terms | paraphrases, synonyms |
| Weak at | rewording | precise codes/numbers |

BM25 is `Ctrl+F` — literal, fast, blind to synonyms. Vector search is asking a
librarian who has read everything: she finds the right book from a vague
description, but may fumble an exact catalogue number. Lab 2.2 hires both.

---

## What is already provided (walk through, don't rewrite)

| Piece | Why it matters |
| --- | --- |
| `SYSTEM_PROMPT` | Unchanged from Lab 1.2 — same grounding rules. Only *retrieval* changed, which is the whole point of the base-class design. |
| `require_api_key()` | Fails fast with a readable message instead of a `KeyError` inside the client. |
| `chat_loop(response)` | Same REPL, takes a function, so any retriever plugs in. |
| `BaseRetriever` (ABC) | **Identical to Lab 1.2.** It owns the LLM client, history, and message formatting; subclasses implement only `retrievedContext(query)` — swapping BM25 → vectors touched *one method*. |
| `get_embeddings()` | Builds an `OpenAIEmbeddings` client pointed at OpenRouter. The embedding model (`text-embedding-3-small`) is separate from and much cheaper than the chat model. |
| `build_or_load_db()` — the cache branch | If `chroma_db/` exists and is non-empty, load it instead of re-embedding. Note it must be given the *same* `embedding_function`, because queries have to land in the same vector space as the stored documents. |

### Notes on a couple of non-obvious bits

- **Two different models are in play.** The embedding model converts text →
  vector for search; the chat model writes the answer. They are unrelated calls.
- **No chunking here.** Emails are short, so each is embedded whole. Real corpora
  (PDFs, manuals) need a splitter — the "one document = one vector"
  shortcut only works because of the data.
- **Persistence is the expensive/cheap split again.** Indexing costs API calls
  and time; querying is a fast nearest-neighbour lookup. Exactly the same
  build-once/query-many shape as the BM25 index — just with meaning instead of
  word counts.
- **`errors="replace"`** when opening files: the corpus has some non-UTF-8 bytes;
  a stray `�` beats a crash.
- **`NUM_RETRIEVED = 4`** (vs `TOP_K = 5` in Lab 1.2) — same recall vs.
  context-window trade-off.

---

## The three TODOs for learners

### Step 1 — load the emails as `Document` objects

```python
docs = []
for fname in sorted(os.listdir(emails_dir)):
    if not fname.endswith(".txt"):
        continue
    with open(os.path.join(emails_dir, fname), encoding="utf-8", errors="replace") as fh:
        docs.append(Document(page_content=fh.read(), metadata={"source": fname}))
```

Teaching points:
- `Document` = `page_content` (what gets embedded) + `metadata` (what does not).
- **Storing the filename in metadata is what makes citation possible later.** In
  Lab 1.2 we kept two index-aligned lists by hand; here the vector store carries
  the label along with the vector, so there's nothing to keep in sync.

### Step 2 — build and persist the store

```python
db = Chroma.from_documents(docs, get_embeddings(), persist_directory=chroma_dir)
return db
```

Teaching points:
- One call embeds every document (batched) and writes the index to disk.
- `persist_directory` is why the second run is instant.

### Step 3 — `getTopK()` and `retrievedContext()`

```python
def getTopK(self, query, k):
    results = self._db.similarity_search_with_score(query, k=k)
    return [(doc.metadata.get("source", "unknown"), doc.page_content, score)
            for doc, score in results]

def retrievedContext(self, query):
    results = self.getTopK(query, self._num_retrieved)
    return "\n\n---\n\n".join(f"[{name}]\n{content}" for name, content, _ in results)
```

Teaching points:
- `similarity_search_with_score` embeds the query and returns the `k` nearest
  documents **already sorted** — no manual sorting, unlike BM25's `get_scores`,
  which returned an unsorted score per document.
- **The score is a DISTANCE: lower = more similar.** This is the opposite of
  BM25, where higher = better — the trap in Lab 2.2 when the two
  score scales get combined.
- `.get("source", "unknown")` is defensive — a document indexed without metadata
  shouldn't crash the chat.
- `retrievedContext()` is byte-for-byte the same as Lab 1.2's: `[filename]`
  labels give the model citation handles and `---` marks where one email ends,
  so it doesn't blend two unrelated emails into one "fact".

---

## Demo questions

| Question | Expected behaviour |
| --- | --- |
| "What was the status of the government project going into 2015?" | Grounded answer: the project was on hold/paused. |
| "Who was involved in discussions about restarting the government project?" | Names people from the emails (Finance Director, COO, …). |
| "What is the CEO's home phone number?" | **Refuses** — not in the corpus. |

**The money demo:** re-run the paraphrase that broke BM25 in Lab 1.2 — ask about
the project being "paused" when the emails say "on hold", or "who signs off on
spending" when the emails say "budget approval". Vector search finds them;
keyword search did not.

Then show the reverse to keep it honest: an exact identifier (an order number, a
rare product code) is often nailed by BM25 and *missed* by embeddings, which blur
precise tokens. Neither retriever wins outright — which is the setup for the
hybrid retriever in Lab 2.2.
