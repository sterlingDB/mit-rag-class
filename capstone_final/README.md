## Install

From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r capstone_final_v3/requirements.txt
```

Required pip packages (listed in `requirements.txt`):

- `python-dotenv`
- `langchain-openai`
- `langchain-core`
- `langchain_chroma`
- `rank_bm25`
- `networkx`

## Run

```sh
cd capstone_final_v3
python capstone.py
```

## File guide

| File | Purpose |
| --- | --- |
| `capstone.py` | Main program: cleans questions, chooses a retrieval path, generates cited answers, and logs results. |
| `evaluation.py` | Questions and grading notes used by the program and LLM evaluations. |
| `hybrid_retriever.py` | Combines keyword search (BM25) with vector search (Chroma). |
| `graph_retriever.py` | Follows links between saved Wikipedia articles to find related evidence. |
| `chroma_helpers.py` | Loads the article index, creates embedding clients, and provides an index-building function. |
| `wiki_helpers.py` | Converts saved Wikipedia HTML into article text. |
| `sample_docs.py` | Loads the Wikipedia articles for rebuilding the index. |
| `semantic_answer_cache.py` | Saves answers and finds cached matches for exact or similar questions. |
| `security_cost_helpers.py` | Limits conversation history, tracks model usage, and prints cost summaries. |
| `production_costs.py` | Calculates chat-cost estimates and separates answer work from safety and routing overhead. |
| `production_evidence.py` | Runs LLM performance and security comparisons and saves reviewable reports. |
| `run_presentation.py` | Runs the four presentation questions with fresh answer caches and saves results under `../presentation/presentation_runs/`. |
| `requirements.txt` | Python packages needed to run the project. |
| `.env.example` | Template for the API-key environment configuration. |
| `.gitignore` | Lists local data, caches, and generated files Git should ignore. |
| `checkpoint_final.log` | Appended run log containing actions, answers, evidence, and usage measurements. |
| `README.md` | Installation, run instructions, and presentation links. |
| `.DS_Store` | macOS folder metadata; not used by the program. |

Folders: `Wikipedia/` holds saved articles; `more-wikipedia-data/` holds supplemental article files; `chroma_db/` stores the article index; `semantic_cache/` stores cached answers; `__pycache__/` contains automatically generated Python bytecode.

## Slideshow

[Wikipedia Research Assistant — Capstone.pdf](<../presentation/Wikipedia Research Assistant — Capstone.pdf>) is the capstone slideshow.

## Recorded presentation

[Morris-John-RAG-Capstone-Final.mov](https://infobasedacis-my.sharepoint.com/:v:/g/personal/jmorris_dacis_com/IQDxNruqwG3ATLXVUhBcXJghATGjXYkIuE7Re6KRaYAET1Q?e=erxyrb) is the recorded capstone presentation.
