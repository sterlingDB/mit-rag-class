# Lab 4.2 — Graph-Augmented Retrieval over the Email Database

Similarity search returns emails that *look like* the question. But "how did the
discussion unfold?" needs the rest of the **conversation**, and "how did this
develop over time?" needs the **earlier** emails on the topic — neither of which
is necessarily similar to the question at all. This lab builds a knowledge graph
of the corpus and uses relationships, not just similarity, to assemble context.

```text
annotatedEmails.json ──► NetworkX graph
   (7,894 emails)          email / thread / person / topic nodes

question ──► hybrid search ──► 5 seed e-mails
                                    │
                    for each seed, walk the graph:
                    ├─ belongs_to ─► thread ─► 5 closest thread neighbours
                    └─ relates_to ─► topic  ─► 5 earliest e-mails per topic
                                    │
                       deduplicate, label each section
                                    │
              RETRIEVED / THREAD CONTEXT / TOPIC CONTEXT ──► LLM ──► answer
```

---

## Folder layout

```text
lab-4.2/
├── Live_Guided_Virtual_Lab_4_2_starter/
│   ├── lab_4_2_graph_retrieval_starter.py   ← learners fill in 1 TODO
│   └── annotatedEmails.json                 ← 7,894 annotated emails
└── Live_Guided_Virtual_Lab_4_2_solution/
    ├── lab_4_2_graph_retrieval_solution.py
    └── annotatedEmails.json
```

No `detailedEmails/` folder — `annotatedEmails.json` carries both the metadata
*and* the `rawText` of every email, so the whole lab runs from that one file.

## Setup

```bash
source .venv/bin/activate
pip install python-dotenv langchain-openai langchain-core langchain-chroma rank-bm25 networkx
echo "OPENROUTER_API_KEY=sk-or-your-key-here" > .env
```

```bash
python lab_4_2_graph_retrieval_starter.py                     # defaults to annotatedEmails.json
python lab_4_2_graph_retrieval_solution.py <path/to/json>
```

### What a run produces

```text
Loading annotatedEmails.json ...
  7894 emails loaded.
Loading existing vector DB from chroma_db/
Building knowledge graph ...
  graph: <N> nodes, <M> edges.
Chat over the email DB. Type your question; 'exit'/'quit' to stop.
```

Nothing is written to disk except `chroma_db/` (reused from Module 2 if you copy
it over). **The graph is rebuilt from JSON on every startup** — it's pure
in-memory computation with no API calls, so it's fast and always current.

The artifact to inspect is the *context* the LLM receives, which is now labelled
by relationship:

```text
=== RETRIEVED E-MAIL: mail_03_12_15_1893.txt ===
...
--- THREAD CONTEXT for mail_03_12_15_1893.txt: mail_03_11_15_1889.txt ---
...
--- TOPIC CONTEXT "budget planning" (via mail_03_12_15_1893.txt): mail_01_08_14_243.txt ---
...
```

---

## Core concepts in this lab

- **Knowledge graph** — model the corpus as *things and their relationships*, not
  a flat pile of documents. Nodes are emails, threads, people, and topics; edges
  say who sent what, what belongs to which conversation, what is about what.
- **Structure vs. similarity** — the previous four retrievers all answered "what
  resembles this query?" A graph answers "what is *connected* to this?" The reply
  three messages later in a thread may share almost no wording with the question
  and still be the thing you need.
- **Seed-and-expand** — search finds an entry point, traversal does the rest.
  Like finding one relevant paper, then reading its citations: the search engine
  gets you in the door; the links get you the context.
- **Typed edges** — `belongs_to`, `sent`, `received_by`, `mentions`,
  `relates_to`. The type is what makes traversal meaningful; "follow the
  `belongs_to` edge to the thread" is a different question from "follow
  `mentions` to a person".
- **Labelled context** — the LLM is told which emails were matched and which are
  supporting context, so it can use the extra material for continuity without
  treating it as the direct answer.
- **Offline annotation** — sender, recipients, mentions, `threadID`, and `topics`
  were extracted ahead of time. Doing that with an LLM over 7,894 emails is slow
  and expensive; it's the kind of work that belongs in an indexing pipeline, not
  in the request path.

### The graph schema

| Node type | Key | Created from |
| --- | --- | --- |
| `email:` | filename | every record |
| `thread:` | `threadID` | records that have one |
| `person:` | lowercased email address | `sender`, `toLst`, `ccLst`, `mentions` |
| `topic:` | topic string | `topics` list |

| Edge | Direction | Meaning |
| --- | --- | --- |
| `belongs_to` | email → thread | this email is part of that conversation |
| `sent` | person → email | that person wrote it |
| `received_by` | email → person | it was addressed to them (to + cc) |
| `mentions` | email → person | they're referred to in the body |
| `relates_to` | email → topic | it's about that subject |

Only `belongs_to` and `relates_to` are used by the retrieval in this lab. The
person edges are built and left available — "which emails did this person send
about that topic?" is the obvious next traversal.

### Quick comparison across Module 4

| | Hybrid (2.2) | Multi-step (4.1) | Graph (4.2) |
| --- | --- | --- | --- |
| Expands the… | scoring | query | result set |
| Extra LLM calls | none | 1 (decompose) | none |
| Needs | corpus | corpus | corpus **+ annotations** |
| Best for | single-intent questions | multi-part questions | conversations, timelines |

Lab 4.1 asked the LLM to work harder. Lab 4.2 asks the *data model* to work
harder — and needs no extra model calls to do it.

---

## What is already provided (walk through, don't rewrite)

| Piece | Why it matters |
| --- | --- |
| `GRAPH_SYSTEM_PROMPT` | `SYSTEM_PROMPT` plus an explanation of the three context labels. The prompt has to teach the model to read the new context format. |
| `load_annotated()` | Reads the JSON into `emails` plus a `filename -> rawText` lookup. |
| `HybridRetriever` | Lab 2.2's fusion, now a **plain class** (no `BaseRetriever`) because here it's only a search component, never a chat interface. |
| `build_graph()` | Constructs all nodes and edges. Note the `if not G.has_node(...)` guards — thread/person/topic nodes are shared across many emails. |
| `_seq_num()` | Pulls the trailing number out of `mail_01_01_14_225.txt`. This integer is the corpus's only ordering signal, and both "closest in thread" and "earliest on topic" depend on it. |
| `GraphRetriever.__init__` | Pre-indexes thread members and topic members into dicts, each sorted by `seq_num`, so traversal is a lookup instead of a scan. |
| `_closest_thread_neighbors()` | Walks outward from the seed's position in the thread, alternating earlier/later. |
| `_earliest_topic_emails()` | Takes the head of the sorted topic list — *earliest*, deliberately, to show how a topic developed. |
| `_assemble_context()` | Formats the labelled sections. |

### Notes on a couple of non-obvious bits

- **The pre-indexing in `__init__` is not just an optimisation.** Sorting members
  by `seq_num` once is what lets "closest neighbours" and "earliest on topic" be
  simple list slices.
- **`nx.DiGraph` is directed**, so `sent` (person → email) and `received_by`
  (email → person) are distinguishable. `out_edges` is used throughout for exactly
  this reason.
- **Topic context is `[:TOPIC_EARLIEST]`, not the most relevant.** For "how did
  this develop over time?" the *origin* of a topic is more useful than its
  best-matching entry — a deliberate choice worth questioning.
- **Context size grows fast.** 5 seeds × (1 + 5 thread + up to 5 per topic) can
  reach dozens of emails. Dedup keeps it from being worse; watch the context
  window on broad questions.
- **Graph quality is annotation quality.** A wrong `threadID` silently produces
  wrong thread context, and nothing in the pipeline will flag it.

---

## The TODO for learners

### `retrievedContext(query) -> str`

```python
seeds = [name for name, _, _ in self._hybrid.getTopK(query, HYBRID_FETCH_K)]
union: dict[str, str] = {}
thread_neighbors: dict[str, list[str]] = {}
topic_context: dict[str, dict[str, list[str]]] = {}

for fname in seeds:
    union[fname] = "seed"                        # seeds always win the label
    email_node = f"email:{fname}"
    if not self._G.has_node(email_node):
        continue
    nbrs = [n for n in self._closest_thread_neighbors(email_node) if n != fname]
    thread_neighbors[fname] = nbrs
    for nb in nbrs:
        union.setdefault(nb, "thread")           # setdefault: never downgrade a seed
    tctx: dict[str, list[str]] = {}
    for topic_node in self._topics_for_email(email_node):
        earliest = [ef for ef in self._earliest_topic_emails(topic_node) if ef != fname]
        if earliest:
            tctx[topic_node.removeprefix("topic:")] = earliest
            for ef in earliest:
                union.setdefault(ef, "topic")
    topic_context[fname] = tctx

# drop thread/topic neighbours that are themselves seeds
for fname in seeds:
    thread_neighbors[fname] = [n for n in thread_neighbors.get(fname, []) if union.get(n) != "seed"]
    topic_context[fname] = {
        t: [ef for ef in files if union.get(ef) != "seed"]
        for t, files in topic_context.get(fname, {}).items()
    }
return self._assemble_context(seeds, thread_neighbors, topic_context)
```

Teaching points:
- **This is seed-and-expand in code**: one line of search, then traversal. The
  hybrid retriever is called exactly once.
- `union` tracks *how* each email entered the context. `union[fname] = "seed"`
  assigns directly, but neighbours use **`setdefault`** — so an email that is both
  a seed and someone's thread neighbour keeps the stronger "seed" label. Direct
  assignment there would be a real bug.
- **The `f"email:{fname}"` prefix** is the node-naming convention that keeps the
  four node types in one namespace without collisions. Forgetting it is the most
  likely mistake here.
- `if not self._G.has_node(...)` — hybrid search can return a filename that never
  made it into the graph; skip rather than crash.
- The final filtering pass exists because a seed's thread neighbour may itself be
  another seed. Without it the same email is pasted into the context twice,
  burning tokens and possibly over-weighting it.
- `topic_node.removeprefix("topic:")` strips the namespace for display — the
  label in the context should read `budget planning`, not `topic:budget planning`.

---

## Demo questions

| Question | What to watch |
| --- | --- |
| "How did the discussion about the government project unfold?" | Thread context reconstructs the conversation around the seeds |
| "Trace how the budget concerns developed over time." | Topic context supplies the *earliest* emails, giving chronology |
| "What is the CEO's home phone number?" | Still refuses — more context doesn't weaken the grounding rules |

**The comparison to make:** run one of the first two against Lab 2.2's hybrid
retriever with a similar email budget. Hybrid returns 5 individually-relevant
emails; graph returns 5 seeds *plus* the surrounding conversation. On "how did it
unfold", the difference should be obvious.

Closing point: this is the last retriever in the course and the only one that
required changing the **data model** rather than the search algorithm. That
tradeoff — richer indexing for better retrieval — is usually where the biggest
gains hide, and it's paid for up front in the annotation pipeline.
