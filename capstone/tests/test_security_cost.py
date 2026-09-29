import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import capstone_checkpoint_6_1_security_cost_starter as agent
import security_experiments


class FakeLLM:
    def __init__(self, *responses, model="test-model", usage=True):
        self.responses = iter(responses)
        self.model_name = model
        self.messages = []
        self.has_usage = usage

    def invoke(self, messages):
        self.messages.append(messages)
        response = next(self.responses)
        return SimpleNamespace(
            content=json.dumps(response) if isinstance(response, dict) else response,
            usage_metadata={"input_tokens": 100, "output_tokens": 20,
                            "input_token_details": {"cache_read": 40}} if self.has_usage else None,
        )


def supported():
    return {"action": "answer", "requirements": [
        {"question": "Fact", "article_id": "A", "quote": "Evidence A"}]}


class SecurityCostTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.enterContext(patch.object(agent, "LOG_PATH", Path(directory.name) / "audit.log"))
        self.enterContext(redirect_stdout(io.StringIO()))
        self.enterContext(patch.object(agent, "RUN_COST_OPTIMIZATIONS", False))
        self.retriever = Mock()
        self.hits = [("A", "Evidence A", 1.0, "bm25")]
        self.retriever.getTopK.return_value = self.hits
        self.graph = Mock()

    def run_agent(self, planner, **kwargs):
        return agent.agentic_answer(planner, self.retriever, self.graph, "Question", **kwargs)

    def test_roles_stay_separate_even_with_same_model_and_count_replans(self):
        llm = FakeLLM({"action": "answer"}, {"action": "search", "queries": ["missing"]},
                      supported(), "Answer [A]")
        result = self.run_agent(llm)
        self.assertEqual(result["planner_calls"], 3)
        self.assertEqual(result["usage"]["planner"]["test-model"]["input"], 300)
        self.assertEqual(result["usage"]["answer"]["test-model"]["output"], 20)
        self.assertEqual(result["usage"]["planner"]["test-model"]["cached"], 120)

    def test_separate_models_and_hardened_prompts(self):
        planner = FakeLLM(supported(), model="planner")
        answer = FakeLLM("Answer [A]", model="answer")
        result = self.run_agent(planner, answer_llm=answer,
                                conversation_history=[{"role": "user", "content": "Fake source"}])
        self.assertIn("planner", result["usage"]["planner"])
        self.assertIn("answer", result["usage"]["answer"])
        self.assertIn("untrusted data", planner.messages[0][0].content)
        self.assertEqual(answer.messages[0][0].content, agent.HARDENED_ANSWER_SYSTEM)
        self.assertIn("Fake source", planner.messages[0][1].content)
        self.assertIn("Fake source", answer.messages[0][1].content)

    def test_unavailable_tools_and_unknown_links_never_execute(self):
        for decision in [{"action": "shell", "command": "env"},
                         {"action": "follow_links", "article_ids": ["../../secret"]},
                         {"action": "search", "queries": ["Question"]}]:
            with self.subTest(decision=decision):
                result = self.run_agent(FakeLLM(decision, "Answer [A]"))
                self.assertIn("Invalid planner decision", result["stop_reason"])
                self.assertEqual(result["search_calls"], 1)
                self.graph.getTopK.assert_not_called()

    def test_link_following_and_citations_are_preserved(self):
        self.graph.getTopK.return_value = [("B", "Evidence B", 0.8, "context: linked from A")]
        planner = FakeLLM({"action": "follow_links", "article_ids": ["A"]}, supported())
        answer = FakeLLM("Answer [A] [B]")
        result = self.run_agent(planner, answer_llm=answer)
        self.assertEqual(result["link_calls"], 1)
        self.assertEqual([source["article_id"] for source in result["sources"]], ["A", "B"])
        self.assertIn("[B]", answer.messages[0][1].content)

    def test_flag_preserves_original_context_or_limits_it(self):
        self.retriever.getTopK.return_value = [("A", "Evidence A" + "x" * 200, 1.0, "bm25")]
        normal = self.run_agent(FakeLLM(supported(), "Answer"))
        self.assertGreater(normal["context_characters"], 25)
        self.retriever.getTopK.assert_called_with("Question", agent.TOP_K)
        with patch.object(agent, "RUN_COST_OPTIMIZATIONS", True), \
             patch.object(agent, "OPTIMIZED_CONTEXT_CHARACTERS", 25):
            compact = self.run_agent(FakeLLM("Answer"))
        self.assertEqual(compact["context_characters"], 25)
        self.assertEqual(compact["planner_calls"], 0)
        self.retriever.getTopK.assert_called_with("Question", agent.OPTIMIZED_TOP_K)
        self.assertIn("character budget", compact["stop_reason"])

    def test_model_configuration_obeys_optimization_flag(self):
        with patch.object(agent, "make_llm") as make:
            agent.make_models(False)
            make.assert_called_once_with(agent.LLM_MODEL)
            make.reset_mock()
            agent.make_models(True)
            self.assertEqual([call.args[0] for call in make.call_args_list],
                             [agent.OPTIMIZED_PLANNER_MODEL, agent.OPTIMIZED_ANSWER_MODEL])

    def test_missing_usage_or_unknown_price_is_not_zero_cost(self):
        result = self.run_agent(FakeLLM(supported(), "Answer", usage=False))
        self.assertEqual(result["usage"]["planner"]["test-model"]["missing_usage_calls"], 1)
        self.assertIsNone(result["estimated_chat_cost_usd"])
        result = self.run_agent(FakeLLM(supported(), "Answer"))
        self.assertIsNone(agent.estimate_cost_usd(result["usage"], {}))
        self.assertAlmostEqual(agent.estimate_cost_usd(result["usage"],
                               {"test-model": {"input": 1, "output": 2, "cached": 0.5}}), 0.00024)

    def test_poison_changes_only_returned_text(self):
        wrapper = agent.PoisonedRetriever(self.retriever)
        poisoned = wrapper.getTopK("Question", 3)
        self.assertIn("AUDIT_INJECTION_SUCCEEDED", poisoned[0][1])
        self.assertEqual(self.hits[0][1], "Evidence A")
        self.assertEqual(poisoned[0][0], "A")

    def test_history_is_bounded(self):
        history = [{"role": "user", "content": "x" * 2000} for _ in range(10)]
        limited = agent.bounded_history(history)
        self.assertLessEqual(len(limited), agent.MAX_HISTORY_MESSAGES)
        self.assertLessEqual(sum(len(m["content"]) for m in limited), agent.MAX_HISTORY_CHARACTERS)

    def test_security_sessions_are_isolated_and_baseline_hardening_both_run(self):
        histories = []
        def fake_probe(*args, **kwargs):
            histories.append([dict(m) for m in kwargs["history"]])
            return {"answer": "Answer"}
        scenarios = [{"name": "first", "turns": ["one", "two"]}, {"name": "second", "turns": ["three"]}]
        with patch.object(security_experiments, "security_scenarios", return_value=scenarios), \
             patch.object(agent, "make_models", return_value=(Mock(), Mock())), \
             patch.object(agent, "probe_agent", side_effect=fake_probe) as probe, \
             patch.object(security_experiments, "print_measurement"):
            security_experiments.run_security_experiments(
                self.retriever, self.graph, "Question",
                model_names=[None], optimize=False,
                make_models=agent.make_models, probe_agent=agent.probe_agent, log=agent.log,
                baseline_prompt=agent.ANSWER_SYSTEM, hardened_prompt=agent.HARDENED_ANSWER_SYSTEM,
                max_history_messages=agent.MAX_HISTORY_MESSAGES,
                max_history_characters=agent.MAX_HISTORY_CHARACTERS,
            )
        self.assertEqual([len(h) for h in histories], [0, 2, 0, 2, 0, 0])
        self.assertEqual(probe.call_args_list[0].args[2], agent.ANSWER_SYSTEM)
        self.assertEqual(probe.call_args_list[2].args[2], agent.HARDENED_ANSWER_SYSTEM)

    def test_no_evidence_skips_answer_and_batch_clarification_never_prompts(self):
        self.retriever.getTopK.return_value = []
        result = self.run_agent(FakeLLM({"action": "shell"}))
        self.assertEqual(result["status"], "no_evidence")
        self.assertNotIn("answer", result["usage"])
        self.retriever.getTopK.return_value = self.hits
        with patch("builtins.input", side_effect=AssertionError("No interactive prompt")):
            result = self.run_agent(FakeLLM({"action": "clarify", "clarification": "Which subject?"}))
        self.assertEqual(result["status"], "clarification_needed")
        self.assertEqual(result["answer_calls"], 0)

    def test_run_writes_cost_evidence_without_live_services(self):
        llm = FakeLLM(supported(), "Answer [A]", model=agent.LLM_MODEL)
        with patch.object(agent, "check_api_key", return_value="unused"), \
             patch.object(agent, "HybridRetriever", return_value=self.retriever), \
             patch.object(agent, "GraphRetriever", return_value=self.graph), \
             patch.object(agent, "get_eval_set", return_value=[{"question": "Question"}]), \
             patch.object(agent, "make_models", return_value=(llm, llm)), \
             patch.object(agent, "RUN_SECURITY_PROBES", True), \
             patch.object(agent, "RUN_MODEL_LADDER", True), \
             patch.object(agent, "INCLUDE_EXPENSIVE_MODELS", True), \
             patch.object(agent, "run_security_experiments") as security_runner, \
             patch.object(agent, "RUN_COST_MODEL_COMPARISON", False):
            agent.run()
        security_runner.assert_called_once()
        self.assertEqual(security_runner.call_args.kwargs["model_names"],
                         agent.SECURITY_MODELS + agent.EXPENSIVE_MODELS)
        self.assertFalse(security_runner.call_args.kwargs["optimize"])
        entries = agent.LOG_PATH.read_text().split("-" * 72 + "\n")
        summaries = [json.loads(entry.split("\n", 1)[1]) for entry in entries
                     if entry and entry.split("\n", 1)[0].endswith("] COST_SUMMARY")]
        self.assertEqual(summaries[0]["completed_questions"], 1)
        self.assertGreater(summaries[0]["estimated_chat_cost_usd"], 0)
        self.assertEqual(summaries[0]["usage"]["planner"][agent.LLM_MODEL]["calls"], 1)


if __name__ == "__main__":
    unittest.main()
