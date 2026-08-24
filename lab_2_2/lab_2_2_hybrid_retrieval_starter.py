r"""Lab 2.2 (STARTER) — hybrid retrieval: fuse BM25 keyword + vector search.

Run:  python lab_2_2_hybrid_retrieval_starter.py <emails_dir>

Your job (see the Lab 2.2 walkthrough) is to combine keyword and vector search:
  - Step 1: Implement _normalize() — min-max scale scores to [0,1], with an
            invert option for the vector distance (lower distance = better).
  - Step 2: Implement HybridRetriever.getTopK() — pull a candidate pool from each
            retriever, normalize both, and combine with a weighted sum.
Both single retrievers (_bm25_topk, _vector_topk), the base class, and the chat
loop are provided. The whole point of this lab is the fusion step.

Setup
-----
1. Create the environment (one-time). Either use conda:
       conda env create -f environment.yml
       conda activate ragcourse
   or a plain virtual environment + pip:
       python -m venv .venv
       #  Windows:      .venv\Scripts\activate
       #  macOS/Linux:  source .venv/bin/activate
       pip install python-dotenv langchain-openai langchain-core langchain-chroma rank-bm25
2. Add your OpenRouter API key (free at https://openrouter.ai/keys). Create a file
   named ".env" in this folder containing a single line:
       OPENROUTER_API_KEY=sk-or-your-key-here
   (or set it in your shell —  Windows:  setx OPENROUTER_API_KEY sk-or-...
    macOS/Linux:  export OPENROUTER_API_KEY=sk-or-...)
3. Email data: Place the company email files (one .txt per email) in a folder
   named 'detailedEmails' in this directory, or pass a folder path as the first
   argument. The folder MUST contain .txt files.
The vector DB is persisted to ./chroma_db (delete it to rebuild).

Sample questions to try (over the email corpus)
-----------------------------------------------
    "What was the status of the government project going into 2015?"
        -> Grounded answer: The project had been put on hold / paused.
    "Who was involved in discussions about restarting the government project?"
        -> Names people from the emails (e.g., the Finance Director and COO).
    "What is the CEO's home phone number?"
        -> This is NOT in the emails, so the assistant should refuse to answer rather than guess.
Compare the retrievers on the same questions: Keyword (1.2) is strong with exact terms,
vector (2.1) handles paraphrases, hybrid (2.2) combines both.
"""
import os
import re
import sys
from abc import ABC, abstractmethod

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from rank_bm25 import BM25Okapi

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "openai/gpt-oss-120b"
EMBEDDING_MODEL = "openai/text-embedding-3-small"
CHROMA_DIR = "chroma_db"
NUM_RETRIEVED = 4          # Emails sent to the LLM as context
CANDIDATE_POOL = 10        # Candidates pulled from EACH retriever before fusion
WEIGHT_BM25 = 0.5
WEIGHT_VECTOR = 0.5

SYSTEM_PROMPT = """You are a helpful assistant for Precision Paperclip Inc. \
You answer questions by drawing information exclusively from the company e-mails \
provided to you as context in each message.

Rules:
- If the answer can be found in the provided e-mails, answer clearly and concisely.
- If the provided e-mails do not contain enough information to answer the question, \
say so explicitly and do not speculate or use outside knowledge.
- Do not answer questions that are unrelated to the content of the provided e-mails."""


def require_api_key() -> None:
    """Exit early with a clear message if OPENROUTER_API_KEY is not set instead of
    failing later with a KeyError when the model client is created."""
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit(
            "\n[setup] OPENROUTER_API_KEY is not set.\n"
            "  1. Get a free key at https://openrouter.ai/keys\n"
            "  2. Create a file named '.env' in this folder with one line:\n"
            "         OPENROUTER_API_KEY=sk-or-your-key-here\n"
            "     or set it in your shell  (Windows: setx OPENROUTER_API_KEY sk-or-... ;\n"
            "     macOS/Linux: export OPENROUTER_API_KEY=sk-or-...).\n"
        )


# ─── Provided: chat loop + base class ────────────────────────────────
def chat_loop(response):
    print("Chat over the email DB. Type your question; 'exit'/'quit' to stop.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break
        if not user_input:
            continue
        try:
            result = response(user_input)
        except Exception as e:
            print(f"Error: {e}")
            continue
        print(f"\nAssistant: {result}\n")


class BaseRetriever(ABC):
    def __init__(self, llm_model: str = LLM_MODEL):
        self._llm = ChatOpenAI(
            model=llm_model,
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url=OPENROUTER_BASE_URL,
        )
        self._history = [SystemMessage(content=SYSTEM_PROMPT)]

    @abstractmethod
    def retrievedContext(self, query: str) -> str: ...

    def _build_user_message(self, query: str, context: str) -> str:
        return f"Context (e-mails):\n{context}\n\nQuestion: {query}"

    def query(self, question: str) -> str:
        context = self.retrievedContext(question)
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=self._build_user_message(question, context)),
        ]
        response = self._llm.invoke(messages)
        return response.content if hasattr(response, "content") else str(response)

    def queryWHistory(self, question: str) -> str:
        context = self.retrievedContext(question)
        self._history.append(HumanMessage(content=self._build_user_message(question, context)))
        try:
            response = self._llm.invoke(self._history)
            answer = response.content if hasattr(response, "content") else str(response)
        except Exception:
            self._history.pop()
            raise
        self._history.append(AIMessage(content=answer))
        return answer

    def chat(self) -> None:
        chat_loop(self.queryWHistory)


# ─── Provided: keyword + vector building blocks ──────────────────────
_STOPWORDS = {
    "a", "an", "the", "and", "but", "or", "nor", "so", "yet", "for",
    "in", "on", "at", "to", "of", "by", "with", "from", "into", "onto", "upon",
    "about", "above", "below", "between", "through", "during", "before", "after",
    "under", "over", "around", "along", "across", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "i", "we", "you", "he", "she", "it", "they", "me", "us", "him", "her", "them",
    "my", "our", "your", "his", "its", "their", "this", "that", "these", "those",
    "as", "if", "up", "out", "not", "no",
}


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS]


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=OPENROUTER_BASE_URL,
    )


def build_or_load_db(emails_dir: str, chroma_dir: str = CHROMA_DIR) -> Chroma:
    if os.path.isdir(chroma_dir) and os.listdir(chroma_dir):
        print(f"Loading existing vector DB from {chroma_dir}/")
        return Chroma(persist_directory=chroma_dir, embedding_function=get_embeddings())
    print("Building vector DB (first run — embedding the emails)...")
    docs = []
    for fname in sorted(os.listdir(emails_dir)):
        if not fname.endswith(".txt"):
            continue
        with open(os.path.join(emails_dir, fname), encoding="utf-8", errors="replace") as fh:
            docs.append(Document(page_content=fh.read(), metadata={"source": fname}))
    db = Chroma.from_documents(docs, get_embeddings(), persist_directory=chroma_dir)
    print(f"  Indexed {len(docs)} emails into {chroma_dir}/")
    return db


# ─── The fusion (your turn) ──────────────────────────────────────────
def _normalize(scores: list[float], invert: bool = False) -> list[float]:
    """Min-max scale a list of scores to [0, 1]. If invert is True, flip the scores 
    so a LOW raw value (e.g., a small vector distance = very similar) becomes a HIGH
    normalized score.


    TODO (Step 1): 
        - Return [] if the list of scores is empty.
        - Return a neutral 0.5 for each score if all scores are equal (to avoid divide-by-zero).
        - Otherwise, normalize the scores using (s - min) / (max - min), where s = the current 
          score, min = minimum of all scores, and max = maximum of all scores.
        - Return 1 minus each score if invert is True.
 


    Delete the raise NotImplementedError line once your code works.
    """
    raise NotImplementedError("Implement _normalize() — see the TODO above.")


class HybridRetriever(BaseRetriever):
    def __init__(self, emails_dir: str, db: Chroma, **kwargs):
        super().__init__(**kwargs)
        paths = sorted(
            os.path.join(emails_dir, f) for f in os.listdir(emails_dir) if f.endswith(".txt")
        )
        self._paths = [os.path.basename(p) for p in paths]
        self._contents = []
        for p in paths:
            with open(p, encoding="utf-8", errors="replace") as fh:
                self._contents.append(fh.read())
        self._bm25 = BM25Okapi([tokenize(doc) for doc in self._contents])
        self._db = db
        print(f"Hybrid retriever ready over {len(self._paths)} emails.")

    # Provided single-retriever helpers. BM25: higher score = better.
    # Vector: returns a DISTANCE, lower = better.
    def _bm25_topk(self, query: str, k: int) -> list[tuple[str, str, float]]:
        scores = self._bm25.get_scores(tokenize(query))
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self._paths[i], self._contents[i], scores[i]) for i in top]

    def _vector_topk(self, query: str, k: int) -> list[tuple[str, str, float]]:
        results = self._db.similarity_search_with_score(query, k=k)
        return [(d.metadata.get("source", "unknown"), d.page_content, s) for d, s in results]

    def getTopK(self, query: str, k: int) -> list[tuple[str, str, float]]:
        """Fuse the two retrievers into one ranking.

        TODO (Step 2):
          1. Pull a candidate pool from each: self._bm25_topk(query, CANDIDATE_POOL)
             and self._vector_topk(query, CANDIDATE_POOL).
          2. Normalize the BM25 scores with _normalize(...) and the vector
             distances with _normalize(..., invert=True).
          3. For every candidate email, compute a fused score:
             WEIGHT_BM25 * bm_norm + WEIGHT_VECTOR * vec_norm
             (a doc missing from one retriever contributes 0 from that side).
          4. Sort by fused score descending and return the top k as
             (filename, content, fused_score).

        Delete the raise NotImplementedError line once your code works.
        """
        raise NotImplementedError("Implement getTopK() fusion — see the TODO above.")

    # Provided: once getTopK works, this formats the context for the LLM.
    def retrievedContext(self, query: str) -> str:
        results = self.getTopK(query, NUM_RETRIEVED)
        return "\n\n---\n\n".join(f"[{name}]\n{content}" for name, content, _ in results)


if __name__ == "__main__":
    require_api_key()
    emails_dir = sys.argv[1] if len(sys.argv) > 1 else "detailedEmails"
    db = build_or_load_db(emails_dir, CHROMA_DIR)
    HybridRetriever(emails_dir=emails_dir, db=db).chat()