# Wikipedia research assistant

This capstone explores answering questions over saved Wikipedia articles with
retrieval-augmented generation (RAG). It combines keyword and vector search,
follows links between local articles, and asks a language model to answer with
article citations. The intended user is someone researching facts or connections
across a fixed collection who needs to inspect the supporting evidence.

This is an educational prototype. A citation identifies a retrieved article;
it does not guarantee that the article supports every claim in the answer.

## Setup

Use Python 3.10 or newer. From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r capstone_final_v3/requirements.txt
cd capstone_final_v3
```

On Windows, activate with `.venv\Scripts\activate` instead. A Python virtual
environment isolates installed packages, similar to keeping Node dependencies
local to a project. `requirements.txt` serves a dependency-list role similar to
`package.json`, but these versions are currently unpinned.

Dependencies: `python-dotenv` loads local configuration, `langchain-openai` and
`langchain-core` provide model/message clients, `langchain_chroma` provides the
local vector index, `rank_bm25` ranks keyword matches, and `networkx` stores the
article-link graph. Tests use Python's built-in `unittest`.

Copy `.env.example` to `.env` in this directory and supply your own
`OPENROUTER_API_KEY`. Never commit the populated file. Live execution uses the
existing OpenRouter integration and consumes credits for model calls and
embeddings. Offline tests do not need a key or corpus.

## Article data and index

**Fresh-clone limitation:** the full `Wikipedia/` article corpus and Chroma index
are ignored by Git. The additional articles in `more-wikipedia-data/` are included
in the submission folder, but this subset alone does not reproduce the full live demo.
Before submission, provide an authorized, reproducible source for the same corpus
or an independently distributable demo corpus with matching evaluation questions.
Do not upload private course downloads just to make the demo work.

With an authorized corpus available:

1. Place saved article HTML files in `capstone_final_v3/Wikipedia/`.
   Filenames become article IDs, for example `The_Brady_Bunch.html`.
   Keep internal article links so graph retrieval can connect indexed articles.
2. Build the index explicitly, from this directory:

   ```sh
   python -c 'from chroma_helpers import build_db_from_docs; build_db_from_docs()'
   ```

   This embeds all documents through OpenRouter and writes `chroma_db/`.
   It can consume substantial credits for a large corpus. The normal entry point
   loads an existing index; it does not build one automatically.

3. Keep the HTML corpus and index in sync. After changing articles, preserve the
   old index elsewhere and build a fresh one. There is no automatic stale-index
   detection. Keep the embedding model consistent between indexing and querying.

If startup reports an empty article index, the `chroma_db/` folder exists but
contains no indexed articles. Having HTML files in `Wikipedia/` alone is not
enough. Restore a populated index built from the same articles and embedding
model, or build it explicitly using step 2 above. The retriever checks for an
empty index before constructing the keyword search index.

The current loader indexes each entire article as one document. It does not
split articles into passages. At answer time, articles share a total 1,500,000-character context budget. There is
no separate per-article cap (`MAX_DOCUMENT_CHARACTERS = None`). The agent takes
each article in retrieval order, retaining as much text as the remaining budget
allows. A long first article can use the entire budget. The optional cost-optimized
mode still uses its smaller 6,000-character total budget. The default is a rough
character allowance near the models' context capacity, not an exact token limit.
Token counts vary with the text, so a large request can still exceed the provider's
context limit. Larger contexts also increase API usage.

### Additional articles added for this project

I added the four saved Wikipedia articles in `more-wikipedia-data/` near the
beginning of the project to supplement the original collection. Many of the
project's questions and answers drew on these articles:

| File                                             | Role in the evaluation                                                                   |
| ------------------------------------------------ | ---------------------------------------------------------------------------------------- |
| `2026_NFL_draft.html`                            | Jeremiyah Love's draft selection and the starting point for the Arizona draft comparison |
| `Ben_Jones_(American_actor_and_politician).html` | The actor who played Cooter Davenport and his congressional service                      |
| `The_Dukes_of_Hazzard.html`                      | The television series and actor/character context                                        |
| `The_Brady_Bunch.html`                           | The series and the house-sale details                                                    |

The loader and link graph read from `Wikipedia/`, not directly from
`more-wikipedia-data/`. To include these supplemental files, run the following
from `capstone_final_v3` before building the index:

```sh
mkdir -p Wikipedia
cp -i more-wikipedia-data/*.html Wikipedia/
```

`cp -i` asks before replacing an existing article with the same filename. Then
build the index using the command above. If an index already exists, preserve it
elsewhere and build a fresh one so its text matches the HTML files.

These four files can also seed a smaller demonstration collection, but its
results will differ from the full-corpus presentation runs. Additional articles
are needed to support the five older Arizona drafts and the state-capital step.
Having the right article available also does not guarantee that its relevant
passage fits within the shared context budget.

## Run

From `capstone_final_v3`, with data, index, dependencies, and key configured:

```sh
python capstone.py --help
python capstone.py
```

The default run uses the first two questions in `evaluation.py`. It prints
answers and measurements and writes `checkpoint_final.log` in the current working
directory. It is a scripted evaluation entry point, not an interactive chat UI.
Review logs before sharing because they contain questions, answers, and excerpts.

Configuration constants are near the top of `capstone.py`:

| Setting                            | Default                        |
| ---------------------------------- | ------------------------------ |
| Planner and answer model           | `openai/gpt-5.4-mini`          |
| Input filter                       | `openai/gpt-5.4-nano`, enabled |
| Routing and quality checks         | `openai/gpt-5-nano`            |
| Search results per query           | 3                              |
| Maximum action rounds              | 3                              |
| Maximum retained documents         | 6                              |
| Total document context             | 1,500,000 characters           |
| Per-document excerpt               | No separate cap (`None`)       |
| Output budget per model request    | 512 tokens                     |
| Cache / router                     | Enabled                        |
| Smaller-context cost configuration | Disabled                       |
| Expensive answer-model fallback    | Disabled                       |

These model IDs describe the code configuration, not a guarantee of provider
availability. Embeddings use `openai/text-embedding-3-small` in
`chroma_helpers.py`. The semantic cache uses a 24-hour TTL, 400-entry limit,
and 0.85 similarity threshold. Corpus and configuration hashes namespace it.
Cache hits still require validation calls and may incur embedding usage.

## How retrieval and generation work

1. `wiki_helpers.py` extracts text from local HTML. `chroma_helpers.py` embeds
   whole articles and persists them with source IDs.
2. `hybrid_retriever.py` combines normalized BM25 scores and vector similarities
   with equal weights, using up to ten candidates from each search method.
3. `graph_retriever.py` follows one-hop links between indexed local articles.
   It does not fetch missing pages. BM25 ranks linked candidates.
4. `capstone.py` sanitizes input, tries a validated cache match, and routes an
   uncached question to direct retrieval or a bounded agent loop. The agent can
   search, follow links, request clarification, or answer. Clarification pauses
   with a status in the default noninteractive run.
5. The model receives bounded excerpts and instructions to cite `[article_id]`
   labels. Usage accounting separates answer work from filtering, routing,
   validation, and quality-check overhead.

## Offline verification

From the repository root, after installing dependencies:

```sh
.venv/bin/python -m unittest discover -s capstone_final_v3/tests
```

Or, from this directory with the virtual environment activated:

```sh
python -m unittest discover -s tests
```

These tests adapt existing checkpoint tests to the final implementation. They
cover graph traversal, credential loading, prompt boundaries, routing and fallback,
cache behavior, cost accounting, and evidence reporting using fakes/mocks.
On October 12, 2026, all 45 tests passed locally. They do not measure real model
accuracy, real attack resistance, or a clean installation on another machine.

## Performance and security evaluation

To rerun all four questions shown in the presentation, from this directory:

```sh
python -u run_presentation.py
```

This uses live OpenRouter credits and bypasses the default two-question limit.
It runs the standalone draft-pick question, Arizona comparison, Dukes of Hazzard
capital question, and Brady Bunch house-sale question once each. Each question
uses a separate empty answer cache. Index and graph setup time is excluded from
request latency. Reports include source-file hashes, an indexed-corpus fingerprint,
configuration, and full results. They are saved incrementally under
`presentation_runs/` (ignored by Git because reports contain source excerpts).
Review the retained evidence before transferring metrics to presentation slides.

The evidence suite uses live API calls and can make many requests:

```sh
python capstone.py --evidence --evidence-output checkpoint_7_1_evidence.json
```

It compares baseline agent execution, direct retrieval where applicable,
optimized cold requests, repeated requests, and a paraphrase. It also probes
injected commands, fabricated sources, forged XML, poisoned retrieved text,
and a persona attack over multiple turns. The default suite evaluates the
configured answer model and `openai/gpt-4o-mini`. The optional
`--include-expensive-model` expands that comparison.

Read each answer against its retained evidence and every part of its question.
Complete the JSON `review` fields before drawing conclusions. A security pass
requires both resisting the attack and answering the legitimate question.
After review, refresh the summary without further API calls:

```sh
python capstone.py --summarize-evidence checkpoint_7_1_evidence.json
```

Only compare savings when answer quality is equivalent. Cost values use a
configured price snapshot and exclude unmeasured embedding costs. They are
estimates of chat cost, not total billed cost or current price quotations.

## Known limitations

- The full live corpus is not distributed with the code. Dependencies are not
  pinned and a clean-clone live run still needs verification.
- Whole-article indexing and leading-excerpt truncation can hide the relevant
  passage. Link traversal only reaches articles already present in the index.
- The evaluation set has four questions. One has empty grading notes and the
  multipart draft question's notes omit most requested comparisons. Correct these
  against the corpus before reporting a pass rate.
- Citation checks verify IDs and some planner-supplied quote matches, not full
  claim entailment. The saved run includes an answer with an unsupported claim.
- `status: answered` means the system produced an answer, not that it passed a
  correctness review. Inspect `stop_reason` and retained evidence as well.
- Input filtering and prompt boundaries reduce some risks but do not prove
  resistance to prompt injection. Model-based judges can also be wrong.
- Extra safety, routing, and cache-validation calls can cost more than they save.
  No reviewed aggregate savings or security pass rate is established here.

See [final submission preparation](docs/final-submission.md) for the timed
presentation draft, recording plan, and the remaining submission checklist.
