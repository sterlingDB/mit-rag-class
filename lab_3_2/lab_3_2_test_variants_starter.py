r"""Lab 3.2 (STARTER) — grow the test set with paraphrase variants.

In Lab 3.2, you probe how brittle each retriever is to rephrasings of the same
question. A systematic way to test this is to expand the test set. For each
question, ask an LLM for a few paraphrases a real user might type, and add them
(sharing the original's grading_notes and sources). Then re-run Lab 3.1 to see
whether a retriever that "passed" still passes on the rephrasings.

Your job (see the Lab 3.2 walkthrough): Implement make_variants().

IMPORTANT: Automatic paraphrases drift. After generating the variants, review them manually and delete any variant that changes the meaning, introduces ambiguity, or is a duplicate/near-duplicate of the original. Keep only variants for which the same grading notes and answer still apply.

Run:  python lab_3_2_test_variants_starter.py [testInputs.json] [n_variants] [out.json]
Writes to <input>_variants.json by default (the original is left untouched).

Setup
-----
1. Create the environment (one-time). Either use conda:
       conda env create -f environment.yml
       conda activate ragcourse
   or a plain virtual environment + pip:
       python -m venv .venv
       #  Windows:      .venv\Scripts\activate
       #  macOS/Linux:  source .venv/bin/activate
       python -m pip install --upgrade pip
       pip install -r requirements.txt   # pinned versions — avoids dependency-drift errors
2. Add the OpenRouter API key provided for this program. Create a file
   named ".env" in this folder containing a single line:
       OPENROUTER_API_KEY=sk-or-your-key-here
   (or set it in your shell —  Windows:  setx OPENROUTER_API_KEY sk-or-...
    macOS/Linux:  export OPENROUTER_API_KEY=sk-or-...)
3. This lab does not need the email corpus. It only rewrites the questions in
   testInputs.json (provided, in this folder). The email folder ('detailedEmails')
   is only needed later, when you evaluate the expanded set with Lab 3.1.

What it prints
--------------
For each question it shows the original and the paraphrase(s) it generated, for example:
    [1] original: What was the status of the government project going into 2015?
          -> variant: How was the government project progressing as it entered 2015?
          -> variant: What state was the government project in at the start of 2015?
Then it shows a summary (N originals + M variants = total) and the file it wrote. Each
variant reuses its original's grading_notes, so the same answer should apply. Your
job is to delete any whose meaning drifted, then evaluate the set with Lab 3.1.
"""
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "openai/gpt-5.4-mini"  # Latest small OpenAI model, fast; covered by course credits

VARIANT_SYSTEM_PROMPT = (
    "You rephrase questions. Given a question, produce alternative phrasings a "
    "real user might type, preserving the meaning so the SAME answer applies. "
    "Return one paraphrase per line, with no numbering or bullets."
)


def require_api_key() -> None:
    """Exit early with a clear message if OPENROUTER_API_KEY is not set instead of
    failing later with a KeyError when the model client is created."""
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit(
            "\n[setup] OPENROUTER_API_KEY is not set.\n"
            "  1. Use the OpenRouter API key provided for this program.\n"
            "  2. Create a file named '.env' in this folder with one line:\n"
            "         OPENROUTER_API_KEY=sk-or-your-key-here\n"
            "     or set it in your shell  (Windows: setx OPENROUTER_API_KEY sk-or-... ;\n"
            "     macOS/Linux: export OPENROUTER_API_KEY=sk-or-...).\n"
        )


def make_variants(llm: ChatOpenAI, question: str, n: int) -> list[str]:
    """Ask the LLM for n paraphrases of `question`, then return the paraphrases as a list of strings.

    TODO: Build a SystemMessage(VARIANT_SYSTEM_PROMPT) + HumanMessage asking for n
    paraphrases of `question`, call llm.invoke(messages), take the reply text, and
    split it into one paraphrase per non-empty line (strip bullets/whitespace).
    Return at most n of them.

    Delete the raise NotImplementedError line once your code works.
    """
    raise NotImplementedError("Implement make_variants() — see the TODO above.")


def main():
    in_path = sys.argv[1] if len(sys.argv) > 1 else "testInputs.json"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    out_path = sys.argv[3] if len(sys.argv) > 3 else str(
        Path(in_path).with_name(Path(in_path).stem + "_variants.json")
    )

    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    llm = ChatOpenAI(
        model=LLM_MODEL,
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=OPENROUTER_BASE_URL,
    )

    print(f"Generating up to {n} paraphrase(s) for each of {len(data)} question(s)...\n")
    augmented = []
    variant_count = 0
    for i, entry in enumerate(data, 1):
        augmented.append(entry)  # keep the original
        print(f"[{i}] original: {entry['question']}")
        for variant in make_variants(llm, entry["question"], n):
            augmented.append({
                "question": variant,
                "grading_notes": entry["grading_notes"],
                "sources": entry.get("sources", []),
            })
            print(f"      -> variant: {variant}")
            variant_count += 1
        print()

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(augmented, f, indent=2)

    print("=" * 72)
    print(f"Done: {len(data)} original question(s) + {variant_count} generated variant(s) "
          f"= {len(augmented)} total.")
    print(f"Written to: {out_path}")
    print("Each variant reuses its original's grading_notes, so the SAME answer should apply.")
    print("\nNext steps:")
    print("  1. Open the file above and delete any variant whose meaning drifted.")
    print("  2. Evaluate the expanded set with Lab 3.1 (that step needs the email folder):")
    print(f"       python lab_3_1_evaluation_starter.py hybrid detailedEmails --inputs {out_path}")
    print("     If the variants score lower than the originals, the retriever is brittle to")
    print("     rephrasing — which is exactly what this lab is probing.")


if __name__ == "__main__":
    require_api_key()
    main()
