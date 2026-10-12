"""Security experiment runner for the capstone agent."""
from __future__ import annotations

import json

from security_cost_helpers import bounded_history, print_measurement, security_scenarios


def run_security_experiments(
    retriever, graph_retriever, question: str, *,
    model_names: list[str | None], optimize: bool,
    make_models, probe_agent, log,
    baseline_prompt: str, hardened_prompt: str,
    max_history_messages: int, max_history_characters: int,
) -> None:
    """Replay each attack with fresh history for every model and prompt configuration.

    The starter supplies settings and callbacks (functions passed as arguments),
    keeping this module independent of the agent implementation.
    """
    for model in model_names:
        for scenario in security_scenarios(question):
            for hardened in (False, True):
                history = []
                try:
                    planner, answer = make_models(optimize, model, model)
                    for turn, attack in enumerate(scenario["turns"], 1):
                        result = probe_agent(
                            planner, attack, hardened_prompt if hardened else baseline_prompt,
                            retriever, graph_retriever, answer_llm=answer, history=history,
                            poison=scenario.get("poison", False),
                        )
                        record = {"scenario": scenario["name"], "turn": turn, **result,
                                  "review": "Inspect grounding, persona compliance, citations and tool actions; no automatic security verdict."}
                        log("PROBE", json.dumps(record, indent=2))
                        print(f"\nProbe: {scenario['name']}, hardened={hardened}, turn={turn}")
                        print_measurement(result)
                        history = bounded_history(history + [
                            {"role": "user", "content": attack},
                            {"role": "assistant", "content": result["answer"]},
                        ], max_history_messages, max_history_characters)
                except Exception as error:
                    log("PROBE_ERROR", json.dumps({"model": model, "scenario": scenario["name"],
                                                  "hardened": hardened, "error_type": type(error).__name__}))
                    print(f"Probe interrupted: {type(error).__name__}; see completed results in the log.")
