> Archived presentation draft. Source-code filenames refer to `capstone_final_v3/`;
> result paths refer to `presentation/`. Use `capstone-slides.html` in the presentation
> folder for the current slideshow and notes.

# Final submission preparation

Updated October 12, 2026 with four fresh requests for `capstone_final_v3`.

The assignment requires an approximately 8–10 minute recording and an accessible
GitHub repository containing the working system and review documentation. The
attached PowerPoint is an eight-slide design-review template. Its prompts cover
the problem, solution, retrieval/evaluation, decisions, and limitations. The
twelve-slide draft below follows those topics and adds space for evidence and a demo.
No design-review or refinement-workshop feedback was received, according to the
presenter. Describe improvements as your own evaluation-driven work.

The [browser slideshow](../presentation/capstone-slides.html) now contains these
twelve slides and their narration. See its [controls and recording instructions](../presentation/README.md).
This document is the presentation script, not a PowerPoint file or recording. The
current evidence supports a candid prototype review. Complete the remaining
work below before calling the submission final.

## Submission readiness

| Requirement | Current evidence | Remaining work |
| --- | --- | --- |
| Source code | Final implementation exists locally | Commit the intended final files and push after review |
| README | Setup, architecture, commands, dependencies, limitations documented | Verify instructions in a clean environment |
| Reproducible data | 2,423 local HTML files counted during inspection | Provide authorized data access or a distributable demo corpus |
| Performance evidence | Four fresh requests measured and evidence-reviewed | Fix observed failures and evaluate a larger set before claiming general accuracy |
| Evaluation criteria | Four questions exist | Repair empty/incomplete grading notes against the corpus |
| Engineering story | Timed draft below | Personalize wording, insert reviewed final results, rehearse |
| Recording | Not created | Record 8–10 minutes and check audio and link access |
| GitHub access | Local origin is `https://github.com/sterlingDB/mit-rag-class` | Verify intended repository contents and facilitator access |

The `capstone_final_v3` directory was untracked at inspection. Existing root
history also contains course worksheets, a course PDF, and logs. Review sharing
rights and contents before changing repository visibility. A `.gitignore` rule
does not remove files already tracked or erase history. No commit, push,
visibility change, or assignment submission was performed during preparation.

## Presentation draft: target 9 minutes

The time budget includes a optional 15–20 second screen demonstration on slide 7.
Rehearse with a timer: speaking speed, pauses, and switching windows determine
the actual duration. Put only the short on-screen text on slides. Use the
narration as speaker notes and adapt it to your own voice.

### Slide 1 — Wikipedia research assistant (0:00–0:25)

On screen:

- Retrieval-augmented answers over saved Wikipedia articles
- Final capstone presentation
- Your name

Narration:

“My capstone is a research assistant for a saved collection of Wikipedia articles. The goal is to answer questions and preserve the evidence behind the answer. I will explain the retrieval workflow, show what the system actually did, and discuss where my design still needs improvement.”

Source: `README.md`, `capstone.py`.

### Slide 2 — Example research questions (0:25–1:10)

On screen:

**One fact**

> What overall pick was Jeremiyah Love in the 2026 NFL draft?

Retrieve the selection and its supporting passage.

**A comparison across drafts**

> what spot did Jeremiyah Love get drafted and compare it to the same round of 5 older drafts for Arizona.

Resolve the round and draft years, then support every comparison. The original
question does not specify which five older drafts.

**A connection across articles**

> What is the capital of the state that the actor who played Cooter Davenport in The Dukes of Hazzard represented in Congress?

Support the actor, represented state, and capital.

**A date and an amount**

> When and for how much did the Brady Bunch house sell?

Support both the sale date and the price.

Narration:

“These examples show different research tasks. The first asks for Jeremiyah Love’s overall draft pick, which needs one supported fact. The second compares his draft position with five older Arizona drafts. That needs evidence across multiple drafts and clarification of which years to compare. The Dukes of Hazzard question connects Cooter Davenport to the actor who played him, the state that actor represented in Congress, and its capital. Each link needs support. The Brady Bunch question asks when and for how much the house sold, so a complete answer needs both the date and the price. Across all four questions, finding a related article is only the beginning. Every requested part needs evidence.”

Source: `evaluation.py:get_eval_set`, `production_evidence.py:evidence_cases`.
These are the existing evaluation prompts, not verified answers or measured results.
The evidence suite also includes the paraphrase “At which overall selection was
Jeremiyah Love chosen in the 2026 NFL draft?” to test reuse of equivalent questions.

### Slide 3 — System workflow (1:10–2:05)

On screen:

1. Extract text and links from saved HTML
2. Search with BM25 and vector similarity
3. Route to direct retrieval or a bounded agent
4. Generate an answer with article citations

Narration:

“The data starts as local HTML. I extract the article text, embed it, and store it in a local Chroma index. The same article IDs connect that index to a graph of links between saved pages. At query time, keyword search and vector search each return candidates. I normalize and combine their scores with equal weights. The system first considers a validated cached answer. Otherwise, a router chooses a direct answer path or an agent that can search again and follow article links. The final answer uses retained excerpts and article IDs. The graph only includes local indexed pages, so following a link does not browse the web or automatically expand the collection.”

Source: `wiki_helpers.py`, `chroma_helpers.py`, `hybrid_retriever.py`,
`graph_retriever.py`, `capstone.py:answer_efficient`.

### Slide 4 — Retrieval choices and context limits (2:05–2:55)

On screen:

- Equal-weight keyword and vector scores
- Local article links for related evidence
- Three action rounds and 8,000 context characters
- Whole articles indexed, leading excerpts retained

Narration:

“Keyword search helps with exact names and terms. Vector search adds a different matching signal when wording changes. Combining them is my design choice, but I have not established that these weights are optimal. Link traversal gives the agent a way to explore related articles. To bound execution, I limit action rounds and the amount of document text in the prompt. The current implementation indexes whole articles and retains at most the first two thousand characters of each selected article. This keeps the implementation simple, but it can discard the passage that actually answers the question. Passage-level retrieval is an important next improvement.”

Source: `hybrid_retriever.py` constants and `capstone.py` retrieval/context logic.

### Slide 5 — Evaluation approach (2:55–3:35)

On screen:

- Four fresh requests include answers, sources, latency, and usage
- Review every requested fact against retained evidence
- Repair incomplete grading notes before reporting accuracy

Narration:

“I ran all four presentation questions with the same settings, giving each an empty answer cache. I recorded request time, model calls, chat tokens, and estimated chat cost, then checked each answer against its sources. The draft-pick and Brady house answers were supported. The Arizona comparison was incomplete, and the Dukes capital claim lacked support in its cited article. These four examples do not establish an overall accuracy rate.”

Source: `docs/latest-run-results.json`, `evaluation.py`, `run_presentation.py`.
Update the evaluation status when the submission results change.

### Slide 6 — Draft pick (3:35–4:05)

> What overall pick was Jeremiyah Love in the 2026 NFL draft?

| Metric | October 12 fresh request |
| --- | --- |
| Latency | 17.35 s |
| Model calls | 5 |
| Chat tokens | 8,095 |
| Estimated chat cost | $0.00454 |

Evidence review: No draft pick returned. The answer acknowledged missing evidence.

One request per question with an empty answer cache. Latency excludes setup.
Chat estimates exclude embeddings and use configured rates.

Narration:

“This request took 17.35 seconds, used 5 model calls and 8,095 chat tokens, and had an estimated chat cost of 0.00454 dollars, excluding embeddings. No draft pick returned. The answer acknowledged missing evidence. The indexed article contains the player at character 2,559, beyond the retained 2,000-character prefix. The right article was in the index, but the answer model did not receive the relevant passage. This points to passage selection as a concrete improvement.”

Source: docs/observed-results.json; local presentation_runs/20261012T172424284856Z/results.json; case draft_pick. Assistant review against retained excerpts. Python source hashes and indexed-corpus fingerprint recorded.

### Slide 7 — Arizona draft comparison (4:05–5:00)

> what spot did Jeremiyah Love get drafted and compare it to the same round of 5 older drafts for Arizona.

| Metric | October 12 fresh request |
| --- | --- |
| Latency | 14.26 s |
| Model calls | 5 |
| Chat tokens | 8,068 |
| Estimated chat cost | $0.00490 |

Evidence review: No draft position or five-draft comparison returned. The answer acknowledged missing evidence.

One request per question with an empty answer cache. Latency excludes setup.
Chat estimates exclude embeddings and use configured rates.

Narration:

“This request took 14.26 seconds, used 5 model calls and 8,068 chat tokens, and had an estimated chat cost of 0.00490 dollars, excluding embeddings. No draft position or five-draft comparison returned. The answer acknowledged missing evidence. The requested comparison also leaves the five draft years unspecified. This measures an unsuccessful attempt, not a completed comparison. I would clarify the years and retrieve evidence for each requested selection.”

Source: docs/observed-results.json; local presentation_runs/20261012T172424284856Z/results.json; case arizona_comparison. Assistant review against retained excerpts. Python source hashes and indexed-corpus fingerprint recorded.

Demo: Optional 15–20 second demo: show this saved run’s question, context-budget stop, and missing-evidence answer. Keep within the slide timing.

### Slide 8 — The Dukes of Hazzard (5:00–5:45)

> What is the capital of the state that the actor who played Cooter Davenport in The Dukes of Hazzard represented in Congress?

| Metric | October 12 fresh request |
| --- | --- |
| Latency | 14.80 s |
| Model calls | 5 |
| Chat tokens | 7,503 |
| Estimated chat cost | $0.00466 |

Evidence review: The answer supplied a capital while admitting it lacked evidence for that claim.

One request per question with an empty answer cache. Latency excludes setup.
Chat estimates exclude embeddings and use configured rates.

Narration:

“This request took 14.80 seconds, used 5 model calls and 7,503 chat tokens, and had an estimated chat cost of 0.00466 dollars, excluding embeddings. The answer supplied a capital while admitting it lacked evidence for that claim. The actor and state are supported in the Ben Jones excerpt. The capital is absent from the retained evidence. Acknowledging the gap did not prevent the model from adding the unsupported claim. A safer response would withhold that claim or retrieve its support.”

Source: docs/observed-results.json; local presentation_runs/20261012T172424284856Z/results.json; case dukes_capital. Assistant review against retained excerpts. Python source hashes and indexed-corpus fingerprint recorded.

### Slide 9 — The Brady Bunch house (5:45–6:15)

> When and for how much did the Brady Bunch house sell?

| Metric | October 12 fresh request |
| --- | --- |
| Latency | 17.05 s |
| Model calls | 5 |
| Chat tokens | 7,676 |
| Estimated chat cost | $0.00429 |

Evidence review: No sale date or price returned. The answer acknowledged missing evidence.

One request per question with an empty answer cache. Latency excludes setup.
Chat estimates exclude embeddings and use configured rates.

Narration:

“This request took 17.05 seconds, used 5 model calls and 7,676 chat tokens, and had an estimated chat cost of 0.00429 dollars, excluding embeddings. No sale date or price returned. The answer acknowledged missing evidence. The indexed sale date and price occur at characters 19,537 and 19,679, beyond the 2,000-character prefix. The right article was in the index, but the answer model did not receive the relevant passage. This points to passage selection as a concrete improvement.”

Source: docs/observed-results.json; local presentation_runs/20261012T172424284856Z/results.json; case brady_house. Assistant review against retained excerpts. Python source hashes and indexed-corpus fingerprint recorded.

### Slide 10 — Safety and cost trade-offs (6:15–7:15)

On screen:

- Input cleaning and XML escaping preserve prompt boundaries
- Cached answers require validation
- Safety and routing add model calls
- Chat estimates exclude embedding costs

Narration:

“The system treats questions and retrieved documents as untrusted data. It cleans some injected instructions and escapes XML characters so pasted text cannot simply create a new prompt section. Those measures do not prove the model will resist every attack. My security evaluation also needs to check that the system still answers the legitimate question. Caching and routing introduce another trade-off. A cache hit can avoid retrieval and answer generation, but the system still validates the cached answer. Filtering, routing, and quality checks also consume tokens and time. I account for that overhead separately. The reported cost is an estimated chat cost using configured rates, with embedding costs still unmeasured. Any savings claim therefore needs equivalent answer quality and a clear statement of what the estimate includes.”

Source: `security_cost_helpers.py`, `semantic_answer_cache.py`,
`production_costs.py`, `production_evidence.py`.

### Slide 11 — Limitations and next improvements (7:15–8:20)

On screen:

- Retrieve relevant passages within long articles
- Require evidence for each answer claim
- Expand and repair the evaluation rubric
- Make the demo reproducible from a fresh clone

Narration:

“My first priority is passage retrieval. The rerun confirmed that the draft player and Brady Bunch sale details exist in the index beyond the first two thousand characters, so the model never receives those passages. My second priority is claim-level support: the capital response admitted missing evidence and still supplied the claim. I also need complete grading criteria and a larger evaluation set. The local corpus and index remain excluded from Git, so a reviewer needs authorized access or a distributable sample. These findings connect each next improvement to an observed limitation.”

Source: `README.md`, `evaluation.py`, `capstone.py:citations_supported`,
`.gitignore`, observed results. Revise this slide as blockers are resolved.

### Slide 12 — Engineering conclusion (8:20–9:00)

On screen:

- Hybrid retrieval and local links make the research process inspectable
- Source labels help review, but do not guarantee support
- Submission: working repository and recorded walkthrough

Narration:

“The main lesson from this project is that retrieval, evidence selection, and answer verification need to work together. Hybrid search and article links give the system several ways to find information. Limits keep execution manageable, but they can also remove necessary evidence. The saved LLM runs show which answers are supported and where quality problems remain. The next version should make those failures measurable with a complete rubric and a reproducible corpus. That is the engineering basis for improving the system, rather than assuming that adding more model calls will solve it.”

At recording time, show the actual accessible repository URL. Add a reviewed
successful example alongside the failures if available, keeping the total under
ten minutes.

## Before recording

- Choose an authorized corpus distribution method. Include its provenance and
  instructions and verify the index build with that corpus.
- Repair each question's grading rubric. For the multipart comparison, specify
  which five drafts and all facts required for each comparison. Validate each
  expected answer from the supplied corpus rather than model memory.
- Capture a successful answer with its supporting excerpts, a question needing
  multiple retrieval steps, and a missing-evidence case. Inspect every claim.
- If discussing caching or security quantitatively, run the appropriate live
  evidence suite, finish its manual review fields, and retain a shareable report.
- Record the commit and corpus version used for final results. The October 12 report records Python file hashes, model settings,
  and an indexed-corpus fingerprint. Preserve that report with the results.
- Check setup in a fresh clone. Keep the recording's claims aligned with the
  revision used for the saved LLM runs.
- Prepare terminal windows in advance, keep keys out of view, and use a saved
  reviewed result as a fallback if a live provider request is slow or unavailable.
  Label a replay as a saved result.

## Recording and submission

1. Move the short slide text into PowerPoint or your preferred slide tool and
   use the narration as notes. Replace “Your name” and update resolved limitations.
2. Rehearse once with the demo. Aim for nine minutes, leaving room for pauses.
3. Record the slides and demonstration with clear narration. Watch the exported
   recording and check that text is readable and the audio is audible.
4. Review the intended Git changes for secrets, logs, and private course material.
   Commit and push the approved submission contents. Check the repository link
   with the same access a facilitator will have.
5. Upload the recording to your chosen service and verify the shareable link's
   permissions. Submit both that link and the GitHub repository link in the LMS.

## Questions to rehearse

- Why hybrid retrieval? It combines exact-term matching with semantic matching;
  comparative evaluation is still needed to justify the chosen weights.
- Why a graph? Article links can expose related local evidence; missing local
  pages and irrelevant links limit its usefulness.
- Why these context limits? They bound work and prompt size, but observed failures
  show that relevant evidence can be lost. They are configuration choices, not
  proven optimal settings.
- Does a citation prove correctness? No. The October 12 capital-question result
  demonstrates the difference between an existing citation ID and claim support.
- Did caching save money? The code supports measurements, but the four fresh requests with empty caches
  do not establish cache savings. Quality-equivalent comparisons are
  required, and embedding costs are not included in chat estimates.
- Is the system secure or production-ready? It is a learning prototype with
  defenses and LLM evaluation tools. That does not establish general security or deployment readiness.
