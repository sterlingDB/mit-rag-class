import ast
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from graph_retriever import GraphRetriever

# Load the functions without executing the script's live API evaluation.
script_path = Path(__file__).resolve().parents[1] / "capstone_checkpoint_5_1_agent_rag_starter.py"
script = ast.parse(script_path.read_text())
assert ast.unparse(script.body[-1]) == "run()"
script.body.pop()
agent = ModuleType("checkpoint_5_1")
agent.__file__ = str(script_path)
exec(compile(script, str(script_path), "exec"), agent.__dict__)


class FakeLLM:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.messages = []

    def invoke(self, messages):
        self.messages.append(messages)
        response = next(self.responses)
        content = json.dumps(response) if isinstance(response, dict) else response
        return SimpleNamespace(content=content)


class FakeRetriever:
    def __init__(self, results=None):
        self.results = results or {}
        self.calls = []

    def get_documents(self):
        return {name: f"Evidence for {name}" for name in "ABCDEFGHI"}

    def getTopK(self, query, k):
        self.calls.append((query, k))
        names = self.results.get(query, ["A"])
        return [(name, f"Evidence for {name}", 0.9, "bm25+vector") for name in names[:k]]

    def score_documents(self, query, doc_ids):
        return {name: 1.0 for name in doc_ids}


def supported_answer(reasoning="Enough evidence"):
    return {
        "action": "answer", "reasoning": reasoning,
        "requirements": [{"question": "First fact", "article_id": "A", "quote": "Evidence for A"}],
    }


class AgentTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        (self.root / "A.html").write_text('<a href="/wiki/B">B</a>')
        (self.root / "B.html").write_text('<a href="/wiki/C">C</a>')
        self.retriever = FakeRetriever()
        self.graph = GraphRetriever(self.retriever, self.root)
        self.graph.getTopK = Mock(wraps=self.graph.getTopK)
        self.enterContext(patch.object(agent, "LOG_PATH", self.root / "agent.log"))
        self.output = self.enterContext(redirect_stdout(io.StringIO()))

    def run_agent(self, llm, **kwargs):
        return agent.agentic_answer(llm, self.retriever, self.graph, "Question", **kwargs)

    def read_events(self, label):
        entries = agent.LOG_PATH.read_text().split("-" * 72 + "\n")
        return [json.loads(entry.split("\n", 1)[1]) for entry in entries
                if entry and entry.split("\n", 1)[0].endswith(f"] {label}")]

    def test_unsupported_five_draft_answer_triggers_another_search(self):
        question = "Compare Jeremiyah Love's draft spot with five older drafts for Arizona."
        hits = [
            ("2026_NFL_draft", "Arizona selected Love third overall in 2026.", 0.9, "bm25+vector"),
            ("2020_NFL_draft", "Arizona selected Simmons eighth overall in 2020.", 0.8, "bm25+vector"),
            ("National_Football_League", "The NFL holds an annual draft.", 0.7, "bm25+vector"),
        ]
        unsupported = {"action": "answer", "reasoning": "The articles cover five older drafts."}
        llm = FakeLLM(unsupported, unsupported, "Four older drafts are missing.")
        with patch.object(self.retriever, "getTopK", return_value=hits):
            result = agent.agentic_answer(llm, self.retriever, self.graph, question)
        self.assertGreater(result["search_calls"], 1)
        self.assertLessEqual(result["steps"], agent.MAX_STEPS)
        self.assertEqual(result["answer"], "Four older drafts are missing.")

    def test_missing_requirement_becomes_a_focused_search(self):
        decision = {
            "action": "answer",
            "requirements": [
                {"question": "First fact", "article_id": "A", "quote": "Evidence for A"},
                {"question": "Second fact", "article_id": "", "quote": ""},
            ],
        }
        result = agent.decide(FakeLLM(decision), "Compare two facts", {"A": "Evidence for A"}, ["Compare two facts"])
        self.assertEqual(result["action"], "search")
        self.assertEqual(result["queries"], ["Second fact"])

    def test_invented_quote_or_source_cannot_justify_early_answer(self):
        for article_id, quote in [("A", "Invented evidence"), ("Missing", "Evidence for A")]:
            with self.subTest(article_id=article_id, quote=quote):
                decision = {"action": "answer", "requirements": [
                    {"question": "Find the missing fact", "article_id": article_id, "quote": quote},
                ]}
                result = agent.decide(FakeLLM(decision), "Question", {"A": "Evidence for A"}, ["Question"])
                self.assertEqual(result["action"], "search")
                self.assertEqual(result["queries"], ["Find the missing fact"])

    def test_supported_checklist_accepts_whitespace_differences(self):
        result = agent.decide(FakeLLM(supported_answer()), "Question", {"A": "Evidence\n for  A"}, ["Question"])
        self.assertEqual(result["action"], "answer")
        self.assertEqual(result["missing_information"], [])

    def test_malformed_checklists_trigger_search_without_crashing(self):
        for requirements in [None, [], "text", [None], [{}], [{"question": []}],
                             [{"question": "Find a fact", "article_id": [], "quote": "Text"}]]:
            with self.subTest(requirements=requirements):
                decision = {"action": "answer", "requirements": requirements}
                result = agent.decide(FakeLLM(decision), "Question", {"A": "Evidence for A"}, ["Question"])
                self.assertEqual(result["action"], "search")

    def test_baseline_and_agent_use_same_cited_answer_prompt(self):
        baseline_llm = FakeLLM("Answer [A].")
        agent_llm = FakeLLM(supported_answer(), "Answer [A].")
        baseline = agent.baseline_answer(baseline_llm, self.retriever, "Question")
        result = self.run_agent(agent_llm)
        self.assertEqual(baseline_llm.messages[-1], agent_llm.messages[-1])
        self.assertIn("[article_id]", baseline_llm.messages[-1][0].content)
        self.assertIn("Retrieval role: primary: bm25+vector", baseline_llm.messages[-1][1].content)
        self.assertEqual(baseline["model_calls"], 1)
        self.assertEqual(result["model_calls"], 2)
        self.assertEqual(result["stop_reason"], "Enough evidence")
        self.assertEqual(self.read_events("AGENTIC_RESULT")[0]["answer"], "Answer [A].")

    def test_link_expansions_preserve_roles_and_deduplicate_sources(self):
        llm = FakeLLM(
            {"action": "follow_links", "article_ids": ["A"], "reasoning": "Need linked evidence"},
            {"action": "follow_links", "article_ids": ["B"]},
            "Answer [B] [C].",
        )
        result = self.run_agent(llm)
        self.assertEqual(self.retriever.calls, [("Question", 3)])
        self.assertEqual(result["link_calls"], 2)
        self.assertEqual(result["planner_calls"], 2)
        self.assertEqual(result["answer_calls"], 1)
        self.assertEqual(result["model_calls"], len(llm.messages))
        self.assertEqual(result["context_documents"], 3)
        self.assertEqual(result["context_characters"], sum(len(f"Evidence for {name}") for name in "ABC"))
        for name in "ABC":
            self.assertEqual(llm.messages[-1][1].content.count(f"[{name}]"), 1)
        self.assertIn("Retrieval role: context: linked from A; BM25", llm.messages[-1][1].content)
        steps = self.read_events("AGENT_STEP")
        self.assertEqual(steps[1]["reasoning"], "Need linked evidence")
        self.assertEqual(steps[1]["retrievals"][0]["seed_articles"], ["A"])
        self.assertEqual(steps[2]["collected_article_ids"], ["A", "B", "C"])

    def test_context_limit_records_all_hits_and_omitted_articles(self):
        self.retriever.results = {"Question": list("ABC"), "more": list("DEF"), "extra": list("GHI")}
        llm = FakeLLM({"action": "search", "queries": ["more", "extra"]}, "Answer [F].")
        result = self.run_agent(llm)
        self.assertEqual(result["context_documents"], 6)
        self.assertEqual(result["unique_documents_retrieved"], 9)
        self.assertEqual(result["search_calls"], 3)
        self.assertEqual(result["planner_calls"], 1)
        self.assertIn("context limit", result["stop_reason"])
        self.assertEqual(self.read_events("AGENT_STEP")[-1]["omitted_article_ids"], list("GHI"))
        self.assertNotIn("[G]", llm.messages[-1][1].content)
        self.assertEqual([source["article_id"] for source in result["sources"]], list("ABCDEF"))

    def test_repeated_queries_do_not_execute_again(self):
        self.retriever.results = {"more": ["A", "B"]}
        llm = FakeLLM(
            {"action": "search", "queries": [" QUESTION ", "more"]},
            {"action": "search", "queries": ["MORE"]},
            "Answer [B].",
        )
        result = self.run_agent(llm)
        self.assertEqual(self.retriever.calls, [("Question", 3), ("more", 3)])
        self.assertIn("no new queries remain", result["stop_reason"])
        self.assertEqual(result["context_documents"], 2)

    def test_repeated_links_do_not_execute_again(self):
        llm = FakeLLM(
            {"action": "follow_links", "article_ids": ["A"]},
            {"action": "follow_links", "article_ids": ["A"]},
            "Answer [B].",
        )
        result = self.run_agent(llm)
        self.assertEqual(self.graph.getTopK.call_count, 1)
        self.assertIn("no new link expansions remain", result["stop_reason"])

    def test_batch_clarification_records_result_without_input_or_answer_call(self):
        llm = FakeLLM({"action": "clarify", "clarification": "Which actor?"})
        with patch("builtins.input", side_effect=AssertionError("Unexpected input")):
            result = self.run_agent(llm)
        self.assertEqual(result["status"], "clarification_needed")
        self.assertEqual(result["answer"], "Clarification needed: Which actor?")
        self.assertEqual(result["model_calls"], 1)
        self.assertEqual(result["answer_calls"], 0)
        self.assertEqual(self.read_events("AGENTIC_RESULT")[0]["status"], "clarification_needed")

    def test_interactive_clarification_reaches_planner_answer_and_log(self):
        llm = FakeLLM(
            {"action": "clarify", "clarification": "Which actor?"},
            supported_answer(), "Answer [A].",
        )
        with patch("builtins.input", return_value="Ben Jones"):
            result = self.run_agent(llm, interactive=True)
        self.assertEqual(result["steps"], 3)
        self.assertIn("Ben Jones", llm.messages[1][1].content)
        self.assertIn("Ben Jones", llm.messages[-1][1].content)
        self.assertEqual(self.read_events("AGENT_STEP")[1]["user_response"], "Ben Jones")

    def test_step_limit_includes_clarifications(self):
        llm = FakeLLM(
            {"action": "clarify", "clarification": "Which actor?"},
            {"action": "clarify", "clarification": "Which role?"}, "Answer [A].",
        )
        with patch("builtins.input", side_effect=["Ben Jones", "Cooter"]):
            result = self.run_agent(llm, interactive=True)
        self.assertEqual(result["steps"], 3)
        self.assertEqual(result["planner_calls"], 2)
        self.assertIn("3 action rounds", result["stop_reason"])

    def test_empty_retrieval_skips_answer_model(self):
        self.retriever.results = {"Question": [], "Question supporting evidence": []}
        baseline = agent.baseline_answer(FakeLLM(), self.retriever, "Question")
        result = self.run_agent(FakeLLM({"action": "answer"}, {"action": "answer"}))
        self.assertEqual(baseline["model_calls"], 0)
        self.assertEqual(result["model_calls"], 2)
        self.assertEqual(result["search_calls"], 2)
        for value in [baseline, result]:
            self.assertEqual(value["status"], "no_evidence")
            self.assertEqual(value["answer_calls"], 0)
            self.assertEqual(value["context_documents"], 0)

    def test_invalid_decision_is_logged_and_does_not_run_a_tool(self):
        result = self.run_agent(FakeLLM("not JSON", "Answer [A]."))
        self.assertIn("Invalid planner decision", result["stop_reason"])
        self.assertEqual(result["search_calls"], 1)
        self.assertEqual(result["link_calls"], 0)
        self.assertIn("Invalid planner decision", self.read_events("AGENT_STEP")[-1]["reasoning"])

    def test_evaluation_uses_shared_grading_and_separates_judge_cost(self):
        llm = FakeLLM("Baseline [A].", "pass", supported_answer(), "Agent [A].", "fail")
        tasks = [{"question": "Question", "grading_notes": "Expected evidence A"}]
        with patch.object(agent, "make_llm", return_value=llm), \
             patch.object(agent, "HybridRetriever", return_value=self.retriever), \
             patch.object(agent, "GraphRetriever", return_value=self.graph), \
             patch.object(agent, "get_eval_set", return_value=tasks), \
             patch.object(agent, "RUN_JUDGE", True), \
             patch.object(agent, "perf_counter", side_effect=[0, 2, 100, 109, 200, 205, 300, 311]), \
             patch("builtins.input", side_effect=AssertionError("Unexpected input")):
            agent.run()
        baseline, result = self.read_events("EVALUATION")
        self.assertEqual(baseline["elapsed_seconds"], 2)
        self.assertEqual(result["elapsed_seconds"], 5)
        self.assertEqual(baseline["judge_elapsed_seconds"], 9)
        self.assertEqual(result["judge_elapsed_seconds"], 11)
        self.assertEqual(baseline["model_calls"], 1)
        self.assertEqual(result["model_calls"], 2)
        self.assertEqual(baseline["judge_calls"], 1)
        self.assertEqual(result["judge_calls"], 1)
        for call in [llm.messages[1], llm.messages[4]]:
            self.assertIn("Expected evidence A", call[1].content)
        summaries = self.read_events("SUMMARY")
        self.assertEqual([summary["passes"] for summary in summaries], [1, 0])
        self.assertIn("BASELINE pass rate: 1/1", self.output.getvalue())
        self.assertIn("AGENTIC pass rate: 0/1", self.output.getvalue())

    def test_disabled_judge_keeps_answers_without_grading_calls(self):
        llm = FakeLLM("Baseline [A].", supported_answer(), "Agent [A].")
        tasks = [{"question": "Question", "grading_notes": "Expected evidence A"}]
        with patch.object(agent, "make_llm", return_value=llm), \
             patch.object(agent, "HybridRetriever", return_value=self.retriever), \
             patch.object(agent, "GraphRetriever", return_value=self.graph), \
             patch.object(agent, "get_eval_set", return_value=tasks), \
             patch.object(agent, "RUN_JUDGE", False), \
             patch.object(agent, "judge", side_effect=AssertionError("Unexpected judge call")):
            agent.run()
        baseline, result = self.read_events("EVALUATION")
        self.assertEqual(baseline["answer"], "Baseline [A].")
        self.assertEqual(result["answer"], "Agent [A].")
        for record in [baseline, result]:
            self.assertEqual(record["verdict"], "skipped")
            self.assertEqual(record["judge_calls"], 0)
            self.assertEqual(record["judge_elapsed_seconds"], 0.0)
        for summary in self.read_events("SUMMARY"):
            self.assertIsNone(summary["passes"])
            self.assertEqual(summary["judge_calls"], 0)
        self.assertIn("grading skipped", self.output.getvalue())
        self.assertNotIn("pass rate", self.output.getvalue())


if __name__ == "__main__":
    unittest.main()
