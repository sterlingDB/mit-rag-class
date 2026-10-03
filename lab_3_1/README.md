# Lab 3.1 — Evaluating the Retrievers with RAGAS

Modules 1–2 built three retrievers. Nobody has yet answered the obvious question:
**which one is actually better?** This lab replaces "it felt right in the demo"
with a number — run a fixed set of questions through each retriever and have an
LLM judge grade every answer against reference notes.

```
testInputs.json ──► 60 questions ──┐
                                   │
        ┌── bm25 ──┐               ▼
query ──┼── vector ┼──► retriever.query() ──► answer ──┐
        └── hybrid ┘                                   │
                                                       ▼
                              grading_notes ──► LLM judge ──► pass / fail
                                                       │
                                            ragas_experiments/…​.csv  +  "42/60 passed"
```

---

## Folder layout

```
lab-3.1/
├── Live_Guided_Virtual_Lab_3_1_starter/
│   ├── lab_3_1_evaluation_starter.py     ← learners fill in 1 TODO, then tune the metric
│   ├── testInputs.json                   ← 60 questions + grading notes + source emails
│   ├── requirements.txt                  ← pinned dependency set
│   ├── environment.yml                   ← conda alternative
│   └── detailedEmails/                   ← the corpus (49 .txt files)
└── Live_Guided_Virtual_Lab_3_1_solution/
    └── … same files, with the experiment implemented
```

All three retrievers from Labs 1.2 / 2.1 / 2.2 are copied in as provided code —
the only new material is the evaluation harness.

## Setup

```bash
source .venv/bin/activate
pip install -r requirements.txt        # RAGAS 0.4.x — 0.2.x lacks the Experiments API
echo "OPENROUTER_API_KEY=sk-or-your-key-here" > .env
```

Run once per retriever, from inside either folder:

```bash
python lab_3_1_evaluation_starter.py bm25   detailedEmails
python lab_3_1_evaluation_starter.py vector detailedEmails
python lab_3_1_evaluation_starter.py hybrid detailedEmails
```

### What a run produces

```
Loaded 60 questions. Evaluating with 'hybrid' retriever...
Experiment complete: 44/60 passed.
Results saved to: …/ragas_experiments/experiments/<timestamp>-<name>.csv
```

It creates a `ragas_experiments/` folder next to the script:

```
ragas_experiments/
├── datasets/
│   └── email_db_eval.csv              ← the questions + grading notes, as loaded
└── experiments/
    └── <timestamp>-<name>.csv         ← one row per question, one file per run
```

The results CSV is the artifact worth opening: each row has the `question`, the
`grading_notes`, which `retriever` produced it, the full `response`, and the
judge's `score`. **Sort by `score` and read the failures** — that's where you see
whether the retriever fetched the wrong emails or the judge was too harsh. The
headline pass count is only a summary; the failures are the actual lesson.

Each run writes a new timestamped file, so the three retrievers can be compared
side by side afterwards.

> Budget 5–10 minutes for all three runs: every question costs one answer call
> plus one judge call, so a 60-question run is ~120 LLM calls. The first
> vector/hybrid run also embeds the corpus into `chroma_db/`.

---

## Core concepts in this lab

- **Evaluation dataset** — a fixed set of questions with known-good answers.
  Fixed is the operative word: the same 60 questions every time, so a change in
  the score means a change in the *system*, not the questions.
- **Grading notes, not exact answers** — the reference is a prose summary of what
  a good answer contains, not a string to match. A free-text answer can be
  correct in unlimited phrasings, so exact comparison is useless here.
- **LLM-as-judge** — a second model reads the answer and the notes and rules
  pass/fail. Like a grader with the marking scheme in hand: they don't need the
  student's wording to match, only the substance.
- **`DiscreteMetric`** — the judge must return one of `allowed_values`
  (`pass` / `fail`). Constraining the output makes the results countable; a judge
  writing free-form critique would give you nothing to sum.
- **The metric prompt is a design decision.** Grading *strictly* ("must cover all
  key points") fails nearly everything and teaches you nothing; grading *loosely*
  passes everything and teaches you nothing. A useful metric is the one that
  separates the retrievers.
- **Experiment tracking** — every run is timestamped and saved. Evaluation is
  comparative by nature; you need the previous run to know if you improved.

### Why "pass rate" is not the whole story

A judge disagreeing with you is *information*. Three different causes hide behind
one failed row, and the CSV is how you tell them apart:

| What went wrong | What you see in the CSV |
| --- | --- |
| Retrieval missed | Answer says "the e-mails don't contain…" but the notes have content |
| Retrieval worked, generation drifted | Answer is fluent, detailed, and contradicts the notes |
| Retrieval + generation fine, **judge** too strict | Answer looks right to you but scored `fail` |

The third case is why Step 2 exists: tune the metric before you conclude anything
about the retrievers.

### Quick comparison across the module

| | Labs 1.2 / 2.1 / 2.2 | Lab 3.1 |
| --- | --- | --- |
| Question asked | "how do I retrieve?" | "how well does it retrieve?" |
| Interface | interactive chat loop | batch run over a dataset |
| Output | one answer on screen | a CSV + a pass count |
| Judged by | you, informally | an LLM against fixed notes |
| Models in play | answering + embedding | answering + embedding + **judge** |

Labs 1.2–2.2 were building the machine; this one is putting it on a test bench.

---

## What is already provided (walk through, don't rewrite)

| Piece | Why it matters |
| --- | --- |
| `BaseRetriever` + all three retrievers | Copied verbatim from Modules 1–2. Note `BaseRetriever` here has **no history** — evaluation runs each question independently, so a stale conversation can't contaminate a later answer. |
| `make_retriever(kind, …)` | Maps the CLI argument to a retriever instance, which is what lets one script evaluate all three. |
| `make_judge()` | Builds the judge via `llm_factory` with a raw `OpenAI` client pointed at OpenRouter. Called from `main()` *after* the API-key check, so a missing key gives a readable message rather than a `KeyError` at import. |
| `correctness_metric` | The `DiscreteMetric` — a prompt template with `{response}` / `{grading_notes}` placeholders and `allowed_values=["pass", "fail"]`. |
| `load_dataset()` | Reads `testInputs.json` into a RAGAS `Dataset` backed by local CSV. Note it keeps only `question` and `grading_notes` — the `sources` field is *not* passed to the judge. |
| `main()` | Argument parsing, conditional DB build (skipped for `bm25`), the run, the pass tally, and the CSV save. |

### Notes on a couple of non-obvious bits

- **Three model roles, two of them easy to confuse.** `LLM_MODEL` answers,
  `EMBEDDING_MODEL` retrieves, `JUDGE_MODEL` grades. They default to the same
  model for answering and judging, which means **the model is grading its own
  work** — a real methodological weakness, and one of the things Step 2 asks you
  to change.
- **`judge` is a module-level global**, assigned in `main()` and declared
  `global` there. It's a workaround for the API-key check: building it at import
  time would crash before `require_api_key()` could print its friendly message.
- **`check_embedding_ctx_length=False`** in `get_embeddings()` — OpenRouter needs
  raw text; without it LangChain sends pre-tokenized input and embedding fails.
- **`testInputs.json` rows carry a `sources` list** of the emails an answer should
  be drawn from. It's unused by this metric — but it's exactly what you'd need to
  measure *retrieval* quality (did we fetch the right emails?) as opposed to
  *answer* quality. Worth pointing at as the natural extension.
- **`@experiment()` + `.arun()` are async** so questions are processed
  concurrently rather than one at a time — otherwise 120 sequential API calls
  would make this lab unbearable.

---

## The exercises for learners

### Step 1 — implement `run_experiment(row)`

```python
@experiment()
async def run_experiment(row):
    response = retriever.query(row["question"])
    score = correctness_metric.score(
        llm=judge,
        response=response,
        grading_notes=row["grading_notes"],
    )
    return {**row, "retriever": kind, "response": response, "score": score.value}
```

Teaching points:
- Three lines, three stages: **answer**, **grade**, **record**. That is the whole
  evaluation loop; everything else in the file is scaffolding.
- The judge sees the `response` and the `grading_notes` — **never the question's
  retrieved context**. It grades the answer, not the retrieval.
- `{**row, …}` spreads the original question and notes into the output, so the
  CSV is self-contained and readable without cross-referencing the input file.
- `score.value` extracts the string (`"pass"`) from the score object; storing the
  object itself would serialise badly into the CSV.
- Returning `"retriever": kind` is what makes the three runs comparable when the
  CSVs are combined.

### Step 2 — tune the metric and re-run

The starting prompt deliberately grades on *main points*, not full coverage.
Things to try, then diff the pass counts:

- **Tighten it:** require every key point in the notes. Expect the pass rate to
  collapse — the notes are multi-point summaries and few answers cover all of it.
- **Loosen it:** pass anything not contradicting the notes. Expect near-100%, and
  no separation between retrievers.
- **Add a middle grade:** `allowed_values=["pass", "partial", "fail"]`. More
  signal, but the `passes` tally in `main()` counts only `"pass"` — worth noticing
  that the reporting code has an assumption baked into it.
- **Change `JUDGE_MODEL`** to something other than `LLM_MODEL` so the grader isn't
  marking its own homework.

Teaching points:
- The scores move when the *metric* changes and the system does not. An
  evaluation number is only meaningful relative to a fixed metric.
- Aim for a metric that **discriminates** — if all three retrievers score
  identically, the metric isn't measuring anything useful.
- Read a handful of failures by hand and decide whether you agree with the judge.
  That manual check is what calibrates the automated one.

---

## Demo flow

1. Run `bm25`, then `vector`, then `hybrid`, and put the three pass counts side
   by side. Hybrid should be at or near the top — the payoff for Lab 2.2.
2. Open a results CSV and read two failures aloud: one where retrieval clearly
   missed, one where the judge was arguably wrong.
3. Change the metric prompt, re-run one retriever, show the count move without a
   single line of retrieval code changing.

The closing point: an evaluation harness is infrastructure, not a one-off script.
Once it exists, every future change — a new embedding model, different weights, a
larger `NUM_RETRIEVED` — can be judged by re-running it instead of arguing about
which demo felt better.
