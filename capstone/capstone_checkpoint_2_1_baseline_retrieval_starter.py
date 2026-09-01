r"""Capstone Checkpoint 2.1 — Retrieval Strategy Design and Baseline Implementation (starter).
Jupytext-style cell markers (# %% / # %% [markdown]) — runnable as a
plain script AND openable as cells in VS Code / PyCharm / Jupytext.
"""

# %% [markdown]
# # Capstone Checkpoint 2.1 — Retrieval Strategy Design and Baseline Implementation
# **MO-LLM Module 2 / Required Capstone Checkpoint (120 minutes)**
#
# ## What this checkpoint is
#
# In Checkpoint 1.1, you showed that a plain LLM can't reliably answer questions about
# your corpus. Now, you will **add retrieval**: Design a retrieval strategy for your scenario
# and build a **baseline retrieval system** that finds the most relevant documents for
# a query, so the model can ground its answers in them.
#
# This mirrors the Module 2 labs — keyword (BM25), vector (semantic), and hybrid
# retrieval — applied to your own capstone corpus. The graded deliverable is the completed 
# Capstone Checkpoint 2.1 worksheet, which includes your written responses and evidence of your 
# retrieval system implementation and testing. This script provides a small working example of 
# baseline retrieval. Use it to understand the retrieval workflow, then adapt the code to implement 
# and test a baseline retriever using your selected capstone dataset.
#
# **Learning outcomes (Module 2):**
# 1. Design a retrieval strategy appropriate for a given dataset and query type.
# 2. Implement and test a baseline retrieval system using structured and/or semantic
#    approaches.

# %% [markdown]
# ## Step 1 — Keep your capstone scenario
#
# Use the **same scenario** you chose in Checkpoint 1.1.
#
# | Scenario | Corpus | Retrieval considerations |
# |---|---|---|
# | **Research Paper Navigator** | ~150 research-paper PDFs (`Labs/CapstoneDatasets/ResearchPapers/`) | long documents; you'll likely chunk them; questions often name a specific paper or compare papers. |
# | **Wikipedia Retrieval Engine** | ~2,400 Wikipedia HTML articles (`Labs/CapstoneDatasets/Wikipedia/`) | many short-to-medium articles; questions name a figure/place or span several articles. |
#
# A good baseline is keyword (BM25), semantic (embeddings + vector search), or a
# hybrid of both — exactly what you built in Labs 1.2–2.2.

# %% [markdown]
# ## Setup (~5 min)
#
# 1. **Python 3.11 or 3.12**
# 2. `pip install langchain-openai langchain-core python-dotenv`
# 3. Use the OpenRouter API key provided for this program. This checkpoint uses
#  the `openai/gpt-5.4-mini` model, with usage covered by the course credits. (this uses the paid gpt-5.4-mini chat model — covered by your course credits — and a keyword retriever, no embeddings).
# 4. Create a `.env` file next to this script: `OPENROUTER_API_KEY=sk-or-v1-...`
#
# This runs on a tiny built-in sample corpus, so you do not need to prepare your own
# dataset. It still requires an OpenRouter API key to run the LLM (it is not offline or
# free of API calls). Your real baseline (over your full corpus) is what you describe in
# the writeup.

# %%
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import os
import re
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from chroma_helpers import build_or_load_db, get_embeddings
from wiki_helpers import load_docs_from_directory, read_wikipedia_article
#from rank_bm25 import BM25Okapi



# %%
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "openai/gpt-5.4-mini"  # latest small OpenAI model, fast; covered by course credits
TEMPERATURE = 0.2
TOP_K = 3
LOG_PATH = Path.cwd() / "checkpoint_2_1_retrieval.log"
CHROMA_DIR = "chroma_db"
EMBEDDING_MODEL = "openai/text-embedding-3-small"
MAX_CONTEXT_CHARS_PER_DOC = 6000

# === SET THIS to the scenario you chose in Checkpoint 1.1 ===
SCENARIO = "wikipedia"   # "research_papers" or "wikipedia"

ANSWER_SYSTEM = (
    "You are a helpful assistant. Answer the question using ONLY the provided "
    "documents, and quote from them where you can. If the documents do not contain "
    "the answer, say so rather than guessing."
)


# %%
def check_api_key() -> str:
    load_dotenv()
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Use the OpenRouter API key "
            "provided for this course, put it in a .env file next to this "
            "script, and rerun."
        )
    return key


def make_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        api_key=check_api_key(),
        base_url=OPENROUTER_BASE_URL,
    )


def log(label: str, text: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(f"[{ts}] {label}\n{text}\n{'-' * 72}\n")


# %% [markdown]
# ## A tiny sample corpus (stands in for your real one)
#
# There are six short "documents" on distinct topics so a baseline retriever has something to
# discriminate between. Your real corpus is the PDFs/articles in `Labs/CapstoneDatasets/`,
# which came with the course in Module 1. Point your code at your local copy of that
# folder, and update the path if your checkout puts it elsewhere.



# all 2400+ wiki docs...
#SAMPLE_DOCS = load_docs_from_directory()

# small test set of 10 wiki docs
SAMPLE_DOCS = [
    {"id": "doc1", "text": read_wikipedia_article("28th_Tony_Awards.html")},
    {"id": "doc2", "text": read_wikipedia_article("The_Beach_Boys.html")},
    {"id": "doc3", "text": read_wikipedia_article("The_Brady_Bunch.html")},
    {"id": "doc4", "text": read_wikipedia_article("The_Dukes_of_Hazzard.html")},
    {"id": "doc5", "text": read_wikipedia_article("Ben_Jones_(American_actor_and_politician).html")},
    {"id": "doc6", "text": read_wikipedia_article("2026_NFL_draft.html")},
    {"id": "doc7", "text": read_wikipedia_article("Wide_Mouth_Mason.html")},
    {"id": "doc8", "text": read_wikipedia_article("United_States_Air_Force.html")},
    {"id": "doc9", "text": read_wikipedia_article("USS_Mizpah.html")},
    {"id": "doc10", "text": read_wikipedia_article("Pineapple.html")},
]
DOC_BY_ID = {d["id"]: d for d in SAMPLE_DOCS}
#print(SAMPLE_DOCS[4])



# %% [markdown]
# ## Step 2 — The baseline retriever (provided)
#
# A simple keyword-overlap retriever scores each document by how many query words
# it shares, and returns the top-k. This is the smallest possible baseline (a stand-in
# for the BM25 / vector / hybrid retriever you built in the Module 2 labs). The
# `answer` function then asks the LLM using only the retrieved documents.

# %%
STOPWORDS = {
    "a", "about", "above", "across", "after", "along", "an", "and", "are",
    "around", "as", "at", "be", "been", "before", "being", "below", "between",
    "but", "by", "did", "do", "does", "during", "for", "from", "had", "has",
    "have", "he", "her", "him", "his", "how", "i", "if", "in", "into", "is",
    "it", "its", "me", "my", "no", "nor", "not", "of", "on", "onto", "or",
    "our", "out", "over", "she", "so", "that", "the", "their", "them",
    "these", "they", "this", "those", "through", "to", "under", "up", "upon",
    "us", "was", "we", "were", "what", "when", "where", "which", "who", "why",
    "with", "yet", "you", "your",
    "1st", "2nd", "3rd",
}

def _normalize_token(token: str) -> str:
    """Handle a couple of simple word endings for this beginner baseline."""
    if token.endswith("ed") and len(token) > 4:
        return token[:-2]
    return token


def _tokens(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {
        _normalize_token(token)
        for token in tokens
        if token not in STOPWORDS and not (token.isdigit() and len(token) < 4)
    }


def retrieve(query: str, k: int = TOP_K) -> list[tuple[str, float]]:
    """Baseline keyword retrieval: Score each doc by shared-word count, return top-k."""
    q = _tokens(query)
    scored = [(d["id"], float(len(q & _tokens(d["text"])))) for d in SAMPLE_DOCS]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [(doc_id, score) for doc_id, score in scored[:k] if score > 0]


def answer(llm: ChatOpenAI, query: str, doc_ids: list[str]) -> str:
    context = "\n\n".join(
        f"[{i}]\n{DOC_BY_ID[i]['text'][:MAX_CONTEXT_CHARS_PER_DOC]}"
        for i in doc_ids
        if i in DOC_BY_ID
    )
    messages = [
        SystemMessage(content=ANSWER_SYSTEM),
        HumanMessage(content=f"Documents:\n{context}\n\nQuestion: {query}"),
    ]
    return llm.invoke(messages).content


# %% [markdown]
# ## Step 3 — Your representative queries (TODO)
#
# Submission item #2 asks for **3-5 representative queries** for your scenario and the
# results your system retrieves for each. Write those queries here. Some good ones include questions that:
#
# - Are answerable from **one** document (tests precision),
# - Need **several** documents (tests recall / aggregation),
# - Have wording that **differs** from the document's wording (i.e., tests whether
#   keyword vs. semantic retrieval matters for your corpus)
#
# Return a list of 3-5 query strings.

# %%
def my_representative_queries() -> list[str]:
    """Return 3-5 representative queries for YOUR chosen scenario.

    TODO — your turn. See the guidance above. Each item is a query string. Pick
    queries that a real user of your system would ask and that require different
    retrieval behaviors (single-doc, multi-doc, paraphrased).

    Delete the raise NotImplementedError line once your code works.
    """
    return [
        "Who were the 1st 3 players drafted in the NFL in 2026?", # happened after training date
        "when did Cooter Davenport die in real life?", # died after training date
        "When and for how much, did the brady bunch house sell?", # sold again after training date
    ]


# %% [markdown]
# ## Step 4 — Run the baseline and capture the evidence
#
# This runs each query through the baseline retriever and the LLM, printing the
# retrieved document ids/scores and the grounded answer, and logging everything to
# `checkpoint_2_1_retrieval.log`. The retrieved documents from the output are the rest of the evidence for
# submission item #2.

# %%
def run() -> None:
    llm = make_llm()
    queries = my_representative_queries()
    print(f"Checkpoint 2.1 — baseline retrieval  |  scenario: {SCENARIO}\n")
    for i, query in enumerate(queries, 1):
        hits = retrieve(query, TOP_K)
        print("=" * 72)
        print(f"QUERY {i}: {query}")
        print(f"  retrieved: {hits}")
        if not hits:
            print("  (nothing matched — note this in your writeup)")
            continue
        ans = answer(llm, query, [doc_id for doc_id, _ in hits])
        print(f"  answer: {ans}\n")
        log(f"QUERY {i}: {query}", f"retrieved={hits}\nanswer={ans}")
    print("=" * 72)
    print("Done. Use the retrieved document results above as evidence in your writeup, and "
          "describe your REAL baseline (over your full corpus) in the submission.")


if __name__ == "__main__":
    run()

# %% [markdown]
# ## Step 5 — Your written submission (the graded deliverable)
#
# Use your completed retrieval implementation and test results to complete the Capstone Checkpoint 2.1
# worksheet. In the worksheet, you will document your retrieval approach, provide evidence that your 
# system is functioning, include 3–5 representative queries and retrieved results, and reflect on where
# your approach performs well and where it struggles.  
 
# Save your completed Python file in the appropriate checkpoint folder in your GitHub repository. 
# Upload the completed worksheet only to the learning platform as your graded submission.
