r"""Lab 2.1 (SOLUTION) — vector retrieval over the emails with Chroma.

Run:  python lab_2_1_vector_retrieval_solution.py <emails_dir>

Builds (once) a Chroma vector database of the emails, then chats using semantic
similarity search. Compare it against the BM25 keyword retriever from Lab 1.2 on
the same queries: Vector search handles paraphrased queries that the keyword search method misses, but
keyword search performs better on queries with exact terms.

Setup
-----
1. Create the environment (one-time). Either use conda:
       conda env create -f environment.yml
       conda activate ragcourse
   or a plain virtual environment + pip:
       python -m venv .venv
       #  Windows:      .venv\Scripts\activate
       #  macOS/Linux:  source .venv/bin/activate
       pip install python-dotenv langchain-openai langchain-core langchain-chroma
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
        -> Names people from the emails (e.g. the Finance Director and COO).
    "What is the CEO's home phone number?"
        -> NOT in the emails, so the assistant should refuse rather than guess.
Compare the retrievers on the same questions: Keyword (1.2) is strong on exact terms,
vector (2.1) handles paraphrases, and hybrid (2.2) combines both.
"""
import os
import sys
from abc import ABC, abstractmethod

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "openai/gpt-oss-120b"
EMBEDDING_MODEL = "openai/text-embedding-3-small"  # via OpenRouter (~$0.02/M tokens)
NUM_RETRIEVED = 4
CHROMA_DIR = "chroma_db"

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


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=OPENROUTER_BASE_URL,
    )


def build_or_load_db(emails_dir: str, chroma_dir: str = CHROMA_DIR) -> Chroma:
    """Load the persisted Chroma DB if it exists; otherwise, build it from the
    emails (each stored whole — they're short) and persist it."""
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


class VectorRetriever(BaseRetriever):
    def __init__(self, db: Chroma, num_retrieved: int = NUM_RETRIEVED, **kwargs):
        super().__init__(**kwargs)
        self._db = db
        self._num_retrieved = num_retrieved

    def getTopK(self, query: str, k: int) -> list[tuple[str, str, float]]:
        results = self._db.similarity_search_with_score(query, k=k)
        return [(doc.metadata.get("source", "unknown"), doc.page_content, score) for doc, score in results]

    def retrievedContext(self, query: str) -> str:
        results = self.getTopK(query, self._num_retrieved)
        return "\n\n---\n\n".join(f"[{name}]\n{content}" for name, content, _ in results)


if __name__ == "__main__":
    require_api_key()
    emails_dir = sys.argv[1] if len(sys.argv) > 1 else "detailedEmails"
    db = build_or_load_db(emails_dir, CHROMA_DIR)
    VectorRetriever(db=db).chat()