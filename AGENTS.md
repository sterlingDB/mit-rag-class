# AGENTS.md

## Project Context

This is a school project for learning about Retrieval-Augmented Generation (RAG) and large language models (LLMs). Prioritize clarity, explanation, and small teachable changes over production complexity.

The user is new to Python but familiar with Node.js. When explaining Python code, connect concepts back to JavaScript/Node.js equivalents where useful.

## How to Work in This Repo

- Keep code beginner-readable and avoid clever abstractions unless they make the lesson clearer.
- Prefer simple, explicit Python examples before introducing frameworks or larger architectures.
- When introducing Python syntax, briefly explain the Node.js equivalent or contrast if it helps.
- Call out Python-specific basics such as indentation, virtual environments, imports, lists/dicts, and package installation.
- Explain important RAG concepts in comments only when the code would otherwise be confusing.
- Do not add paid services, external APIs, or vector databases unless the user asks for them.
- Never commit secrets, API keys, tokens, downloaded private course material, or personal data.
- If adding dependencies, keep them minimal and document why they are needed.
- If adding generated outputs, keep them small and reproducible.

## RAG/LLM Learning Priorities

- Make retrieval steps visible: loading data, chunking, embedding, search, and answer generation.
- Preserve source references where possible so answers can be traced back to retrieved material.
- Prefer deterministic examples for tests and demos when possible.
- Be careful not to present model output as guaranteed fact; include citations or source snippets when the app supports them.

## Testing and Verification

- For small Python scripts, run the script directly with `python main.py` or the relevant entry point.
- If tests are added later, include the command here and keep examples easy to run locally.
- Verify that code still works without requiring unavailable API credentials unless the user explicitly wants an API-backed example.

## Style

- Use straightforward Python naming and formatting.
- Keep files focused and short while the project is in an early learning stage.
- Favor comments that teach why a RAG step matters, not comments that restate the code.
