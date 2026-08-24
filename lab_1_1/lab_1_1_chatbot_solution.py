r"""Lab 1.1 (SOLUTION) — a chatbot, with and without conversation memory.

Run:  python lab_1_1_chatbot_solution.py

By default, this runs the memory-enabled chatbot. To see the difference, set
USE_HISTORY = False and try a two-turn conversation:
    You: When was the War of 1812?
    You: And who won?
Without history, the follow-up question isn't answered correctly ("which competition?"); with history,
it answers in context. In the final step, probe failure modes through interaction (hallucination, knowledge staleness, and inconsistency).

Setup
-----
1. Create the environment (one-time). Either use conda:
       conda env create -f environment.yml
       conda activate ragcourse
   or a plain virtual environment + pip:
       python -m venv .venv
       #  Windows:      .venv\Scripts\activate
       #  macOS/Linux:  source .venv/bin/activate
       pip install python-dotenv langchain-openai langchain-core
2. Add your OpenRouter API key (free at https://openrouter.ai/keys). Create a file
   named ".env" in this folder containing a single line:
       OPENROUTER_API_KEY=sk-or-your-key-here
   (or set it in your shell —  Windows:  setx OPENROUTER_API_KEY sk-or-...
    macOS/Linux:  export OPENROUTER_API_KEY=sk-or-...)
3. This lab needs no dataset — it talks to the model directly.

Sample questions to try
-----------------------
General-knowledge questions (answered from the model's own training):
    You: When was the War of 1812?   ->  ~"June 1812 to February 1815."
    You: And who won?                ->  memory ON : "a stalemate; the Treaty of Ghent
                                                      restored the pre-war borders."
                                         memory OFF: confused ("which competition?") —
                                                     it forgot the previous turn.
This contrast is the point of the lab. Then probe failure modes (Step 4): Ask
something obscure (hallucination), company-specific (knowledge gap), or very recent
(stale knowledge), and rephrase a question to spot inconsistency.
"""
import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

# OpenRouter is the gateway to the LLMs for this course.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "openai/gpt-oss-120b"  # "openai/gpt-4o-mini" (paid, cheap) also good
SYSTEM_PROMPT = "You are a helpful assistant. Answer the user's questions concisely."

USE_HISTORY = True  # set False to run the no-memory version


def require_api_key() -> None:
    """Exit early with a clear message if OPENROUTER_API_KEY is not set, instead of
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
    """Minimal command-line chat loop: Reads input, prints response(input)."""
    print("Type your question and press Enter. Type 'exit' or 'quit' to stop.\n")
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


def make_llm():
    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=OPENROUTER_BASE_URL,
    )


def build_no_history_respond():
    """Step 1-2: a chatbot with NO memory (each turn is independent)."""
    llm = make_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])
    chain = prompt | llm

    def respond(user_input):
        response = chain.invoke({"question": user_input})
        return response.content if hasattr(response, "content") else str(response)

    return respond


def build_history_respond():
    """Step 3: a chatbot that remembers the conversation by passing the full
    message history (System + Human/AI turns) to the LLM each time."""
    llm = make_llm()
    history = [SystemMessage(content=SYSTEM_PROMPT)]

    def respond(user_input):
        history.append(HumanMessage(content=user_input))
        try:
            response = llm.invoke(history)
            answer = response.content if hasattr(response, "content") else str(response)
        except Exception:
            history.pop()  # roll back the user turn if the call failed
            raise
        history.append(AIMessage(content=answer))
        return answer

    return respond


if __name__ == "__main__":
    require_api_key()
    respond = build_history_respond() if USE_HISTORY else build_no_history_respond()
    chat_loop(respond)