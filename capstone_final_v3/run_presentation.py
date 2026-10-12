"""Run the four presentation questions once each. Uses OpenRouter credits."""
import hashlib
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import capstone as agent
from production_evidence import evidence_cases
from semantic_answer_cache import cache_namespace


def main():
    root = Path(__file__).resolve().parent
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = root / "presentation_runs" / stamp
    output.mkdir(parents=True)
    agent.LOG_PATH = output / "agent.log"
    report = {
        "created_at_utc": stamp,
        "status": "running",
        "protocol": "One request per question; separate empty answer cache per request. "
                    "Existing article index; setup excluded from request latency. "
                    "Chat cost uses configured rates and excludes embedding costs.",
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(root.glob("*.py"))},
        "configuration": {name: getattr(agent, name) for name in (
            "LLM_MODEL", "PERSONA_FILTER_MODEL", "ROUTER_MODEL", "USE_PERSONA_FILTER",
            "USE_ROUTER", "USE_CACHE", "RUN_COST_OPTIMIZATIONS", "MAX_STEPS", "TOP_K",
            "MAX_CONTEXT_CHARACTERS", "MAX_CONTEXT_DOCS", "MAX_DOCUMENT_CHARACTERS",
            "MAX_OUTPUT_TOKENS", "PRICE_PER_MTOK", "USE_EXPENSIVE_FALLBACK")},
        "records": [],
    }

    def save():
        (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    print(f"Report: {output / 'results.json'}", flush=True)
    try:
        agent.check_api_key()
        retriever = agent.HybridRetriever(num_retrieved=agent.TOP_K)
        print("Building the local link graph from the HTML corpus...", flush=True)
        graph = agent.GraphRetriever(retriever)
        print("Local graph ready.", flush=True)
        report["indexed_documents"] = len(retriever.get_documents())
        report["corpus_fingerprint"] = cache_namespace(retriever.get_documents(), {})
        planner, answer = agent.make_models(agent.RUN_COST_OPTIMIZATIONS)
        router = agent.make_llm(agent.ROUTER_MODEL)
        evaluation = agent.get_eval_set()
        cases = [
            ("draft_pick", evidence_cases(evaluation)[0]["question"]),
            ("arizona_comparison", evaluation[0]["question"]),
            ("dukes_capital", evaluation[1]["question"]),
            ("brady_house", evaluation[3]["question"]),
        ]
        for name, question in cases:
            record = {"case": name, "question": question, "review": None}
            report["records"].append(record)
            print(f"Running {name}...", flush=True)
            try:
                # An empty cache for each question prevents cross-question reuse.
                with tempfile.TemporaryDirectory(prefix="presentation-cache-") as directory:
                    agent.CACHE_PATH = Path(directory)
                    cache = agent.make_answer_cache(retriever) if agent.USE_CACHE else None
                    record["result"] = agent.answer_efficient(
                        planner, retriever, graph, question, answer_llm=answer,
                        router_llm=router, cache=cache, use_router=agent.USE_ROUTER,
                    )
                result = record["result"]
                print(json.dumps({key: result.get(key) for key in (
                    "path", "elapsed_seconds", "model_calls", "estimated_chat_cost_usd",
                    "status", "stop_reason", "answer")}), flush=True)
            except Exception as error:
                record["error_type"] = type(error).__name__
                print(f"Request failed: {type(error).__name__}", flush=True)
                if getattr(error, "status_code", None) in (401, 402, 403):
                    report["status"] = "blocked_by_authentication_or_credits"
                    break
            finally:
                save()
        if report["status"] == "running":
            report["status"] = ("completed_with_errors" if any("error_type" in r for r in report["records"])
                                else "measurements_complete_review_pending")
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        raise
    except Exception as error:
        report["status"] = "setup_failed"
        report["error_type"] = type(error).__name__
        print(f"Setup failed: {type(error).__name__}", flush=True)
    finally:
        save()
        print(f"Status: {report['status']}", flush=True)


if __name__ == "__main__":
    main()
