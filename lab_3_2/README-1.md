# Lab 3.2 — Growing the Test Set with Paraphrase Variants

Lab 3.1 measured the retrievers against 53 fixed questions — each phrased exactly
one way. Real users don't phrase things the way your test set does. This lab uses
an LLM to generate paraphrases of every question, so you can re-run the evaluation
and find out whether a retriever that "passed" only passed the *wording* it was
given.

```
testInputs.json ──► for each question ──► LLM ("rephrase this") ──► 3 paraphrases
   (53 rows)                                                              │
                                    each variant inherits the original's  │
                                       grading_notes + sources ───────────┘
                                                    │
                                    testInputs_variants.json  (212 rows)
                                                    │
                                        manual review — delete the drifted ones
                                                    │
                                   Lab 3.1  --inputs testInputs_variants.json
```

---

## Folder layout

```
lab-3.2/
├── Live_Guided_Virtual_Lab_3_2_starter/
│   ├── lab_3_2_test_variants_starter.py   ← learners fill in 1 TODO
│   ├── testInputs.json                    ← the 53 originals from Lab 3.1
│   └── requirements.txt
└── Live_Guided_Virtual_Lab_3_2_solution/
    ├── lab_3_2_test_variants_solution.py
    ├── testInputs.json
    ├── testInputs_variants.json           ← pre-generated output (212 rows)
    └── requirements.txt
```

No `detailedEmails/` here — this lab only rewrites questions. The corpus is needed
again only when the expanded set is fed back into Lab 3.1.

## Setup

```bash
source .venv/bin/activate
pip install -r requirements.txt
echo "OPENROUTER_API_KEY=sk-or-your-key-here" > .env
```

```bash
python lab_3_2_test_variants_starter.py                          # defaults: testInputs.json, n=3
python lab_3_2_test_variants_solution.py testInputs.json 3 out.json
```

Positional arguments are `[input] [n_variants] [output]`; the output defaults to
`<input>_variants.json`, so **the original file is never overwritten**.

### What a run produces

```
Generating up to 3 paraphrase(s) for each of 53 question(s)...

[1] original: What significant topics were suggested for discussion during the year-end...
      -> variant: Which major subjects were proposed for the end-of-year strategy meeting...
      -> variant: What key items were put forward for discussion at the December 2013...
      -> variant: What topics did people suggest covering in the year-end strategy session?
...
========================================================================
Done: 53 original question(s) + 159 generated variant(s) = 212 total.
Written to: testInputs_variants.json
```

The output JSON has the same shape as `testInputs.json` — `question`,
`grading_notes`, `sources` — so it drops straight into Lab 3.1 via `--inputs`.
Originals and their variants sit adjacent in the file, which makes the manual
review pass easy to skim.

> One LLM call per question (53 calls), so this runs in well under a minute — far
> cheaper than a Lab 3.1 evaluation.

---

## Core concepts in this lab

- **Robustness testing** — a passing score on one phrasing is a weak result. Asking
  the same thing five ways and passing all five is a much stronger claim. Like
  testing a search box with what users actually type, not just the query you had
  in mind when you built it.
- **Synthetic data generation** — the LLM is used here as a *tooling* step, not as
  the product. Writing 159 paraphrases by hand is a day of work; generating them
  is one call per question.
- **Invariance** — a paraphrase must preserve the answer. The grading notes are
  reused unchanged, which is precisely the assertion being made: *same meaning,
  same expected answer*. If that's false for a row, the row is broken.
- **Drift, and why review is mandatory** — LLMs quietly narrow, broaden, or shift
  a question ("What did Liam do about employee concerns?" → "What were the
  employee concerns?"). That variant now has the wrong grading notes and will fail
  for a reason that has nothing to do with retrieval.
- **Generated data is a draft, not an artifact.** The human pass at the end is
  what makes the test set trustworthy.

### Reading the results after re-running Lab 3.1

| What you see | What it means |
| --- | --- |
| Variants score ≈ originals | The retriever generalises across phrasing |
| Variants score well below originals | Brittle to rephrasing — the Lab 3.1 number was optimistic |
| One retriever drops more than another | Expect BM25 to suffer most; paraphrases replace the exact words it depends on |

The BM25 drop is the point the whole module has been building toward — Lab 1.2's
weakness, now with a number attached rather than a hand-picked demo question.

### Quick comparison to Lab 3.1

| | Lab 3.1 | Lab 3.2 |
| --- | --- | --- |
| Question asked | how well does it retrieve? | does that hold when wording changes? |
| LLM's role | answering + judging | generating test data |
| Input | 53 questions | same 53 |
| Output | pass/fail CSV | a larger `.json` question set |
| Cost | ~2 calls per question | 1 call per question |
| Needs the corpus | yes | no |

---

## What is already provided (walk through, don't rewrite)

| Piece | Why it matters |
| --- | --- |
| `VARIANT_SYSTEM_PROMPT` | Pins the two things that matter: preserve the meaning so the same answer applies, and return one paraphrase per line with no numbering. The second half exists purely to make parsing trivial. |
| `require_api_key()` | Same fail-fast helper as every other lab. |
| `main()` — the loop | Appends the original first, then its variants, so related rows stay adjacent for review. |
| `main()` — the inheritance | Each variant copies `grading_notes` and `sources` from its parent. This is the whole trick: no new reference answers need to be written. |
| `main()` — output path | `Path(in_path).with_name(stem + "_variants.json")` — writes beside the input, never over it. |
| The "Next steps" print block | Ends with the exact Lab 3.1 command, `--inputs` included. |

### Notes on a couple of non-obvious bits

- **Output format is enforced by prompt, not by code.** The parser assumes
  one-per-line, which holds only because the system prompt demands it. A chattier
  model that adds "Sure! Here are three paraphrases:" would put that preamble into
  the test set as a question.
- **`entry.get("sources", [])`** uses `.get` because `sources` is optional in the
  input schema; `grading_notes` is accessed directly since a row without it is
  unusable.
- **No de-duplication.** Ask for 3 paraphrases and you may get 2 distinct ones plus
  a near-copy. That's part of what the manual review is for.
- **The docstring mentions `environment.yml`** for conda setup, but only
  `requirements.txt` ships in these folders.

---

## The TODO for learners

### `make_variants(llm, question, n) -> list[str]`

```python
messages = [
    SystemMessage(content=VARIANT_SYSTEM_PROMPT),
    HumanMessage(content=f"Question: {question}\n\nProduce {n} paraphrases."),
]
response = llm.invoke(messages)
text = response.content if hasattr(response, "content") else str(response)
variants = [line.strip("-•* ").strip() for line in text.splitlines() if line.strip()]
return variants[:n]
```

Teaching points:
- Same `SystemMessage` + `HumanMessage` shape as every retriever in the course —
  the system message sets the rules, the human message carries the payload. Only
  the task changed.
- The response is **free text that has to be parsed back into data**. Splitting on
  newlines works only because the prompt asked for one per line: prompt and parser
  are a matched pair, and changing one breaks the other.
- `.strip("-•* ")` defends against bullets appearing despite the instruction.
  Models comply with formatting rules *most* of the time.
- The `if line.strip()` filter drops blank lines; without it, empty strings become
  questions in the test set.
- `[:n]` truncates — asking for 3 and receiving 5 is common, and the extra rows
  would silently unbalance the set.
- `hasattr(response, "content")` is the same defensive unwrap used in
  `BaseRetriever.query()`.

### Then: the manual review

Open the generated file and delete any variant that:
- changed the meaning or scope of the original question,
- became ambiguous or unanswerable,
- duplicates the original or another variant.

Keeping a bad variant means Lab 3.1 will report a failure caused by your test
data, not by the retriever. This step is the actual lesson of the lab.

---

## Demo flow

1. Run the generator on a handful of questions and read the paraphrases aloud.
2. Find a drifted one live — there's usually at least one in 10 — and delete it.
3. Re-run Lab 3.1 with `--inputs testInputs_variants.json` for `bm25` and
   `hybrid`, and compare against the Lab 3.1 baselines.

Closing point: a test set is a living asset. Every rephrasing that breaks the
system is a row worth keeping permanently, and the same generate-review-evaluate
loop is how evaluation sets grow in production.
