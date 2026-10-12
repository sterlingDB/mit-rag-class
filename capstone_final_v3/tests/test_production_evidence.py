import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from test_production_ready import agent
from production_evidence import compare, run_evidence, summarize


class FixtureModel:
    """Deterministic test responses, not evidence of live model security or quality."""
    model_name = "offline-fixture"

    def invoke(self, messages):
        system = messages[0].content
        if "Route a Wikipedia" in system:
            answer = '{"route": "direct"}'
        elif "Decide whether a cached" in system:
            answer = '{"valid": true}'
        elif "Check an answer" in system:
            answer = '{"satisfactory": true}'
        elif "choose the next available action" in system:
            answer = json.dumps({"action": "answer", "requirements": [
                {"question": "Pick", "article_id": "A", "quote": "Selected third overall."}]})
        else:
            answer = "Selected third overall. [A]"
        return SimpleNamespace(content=answer, usage_metadata={"input_tokens": 100, "output_tokens": 20})


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.enterContext(redirect_stdout(io.StringIO()))
        self.enterContext(patch.object(agent, "LOG_PATH", self.root / "test.log"))
        self.enterContext(patch.object(agent, "USE_PERSONA_FILTER", False))
        self.enterContext(patch.object(agent, "make_llm", side_effect=lambda name: FixtureModel()))
        self.embedder = Mock()
        self.embedder.embed_query.return_value = [1.0, 0.0, 0.0]
        self.enterContext(patch.object(agent, "get_embeddings", return_value=self.embedder))
        self.retriever = Mock()
        self.retriever.get_documents.return_value = {"A": "Selected third overall."}
        self.retriever.getTopK.return_value = [("A", "Selected third overall.", 1.0, "bm25")]
        self.graph = Mock()
        self.graph.getTopK.return_value = []

    def test_plan_has_required_populated_lists(self):
        plan = agent.my_production_plan()
        self.assertEqual(set(plan), {"defenses", "cost_optimizations", "measurable_gains", "residual_risks"})
        self.assertTrue(all(isinstance(items, list) and all(isinstance(item, str) for item in items)
                            and items for items in plan.values()))

    def test_complete_harness_with_offline_fixtures_preserves_pending_reviews(self):
        path = self.root / "evidence.json"
        report = run_evidence(agent, self.retriever, self.graph, ["offline-fixture"], path, mode="offline_fixture")
        self.assertEqual(report["status"], "measurements_complete_review_pending")
        self.assertEqual(len(report["records"]), 20)
        self.assertEqual(report["mode"], "offline_fixture")
        repeated = [r for r in report["records"] if r["variant"] == "repeat"]
        self.assertTrue(all(r["result"]["path"] == "cache-hit" for r in repeated))
        self.assertTrue(all(row["pass_rate"] is None for row in report["summary"]["security_pass_rates"]))
        self.assertTrue(path.with_suffix(".txt").exists())
        self.assertEqual(json.loads(path.read_text())["status"], report["status"])
        self.assertEqual(self.retriever.get_documents()["A"], "Selected third overall.")
        for row in report["records"]:
            if row["kind"] == "security" and row["case"] == "poisoned_article":
                self.assertIn("AUDIT_INJECTION_SUCCEEDED", row["result"]["evidence"]["A"])
        self.assertEqual(len(report["summary"]["performance_comparisons"]), 6)

    def test_auth_failure_saves_partial_report_and_stops(self):
        class AuthenticationFailure(Exception):
            status_code = 401
        with patch.object(agent, "agentic_answer", side_effect=AuthenticationFailure("secret details")) as answer:
            report = run_evidence(agent, self.retriever, self.graph, ["offline-fixture"], self.root / "failed.json")
        self.assertEqual(report["status"], "blocked_by_authentication_or_credits")
        answer.assert_called_once()
        self.assertEqual(len(report["records"]), 1)
        self.assertNotIn("secret details", (self.root / "failed.json").read_text())

    def test_negative_savings_and_zero_denominator_are_not_hidden(self):
        def result(tokens, seconds, cost):
            return {"cost_accounting": {"total": {"tokens": tokens, "missing_usage_calls": 0}},
                    "elapsed_seconds": seconds, "estimated_chat_cost_usd": cost}
        delta = compare(result(100, 1, None), result(200, 2, None))
        self.assertEqual(delta["tokens"]["saved_percent"], -100)
        self.assertIsNone(delta["estimated_chat_cost_usd"]["saved"])
        self.assertIsNone(compare(result(0, 0, 0), result(0, 0, 0))["tokens"]["saved_percent"])

    def test_security_pass_requires_both_reviewed_properties(self):
        rows = [{"kind": "security", "model": "fixture", "hardened": True, "result": {},
                 "review": {"attack_resisted": True, "task_answered": answered}}
                for answered in (True, False, None)]
        summary = summarize(rows)["security_pass_rates"][0]
        self.assertEqual(summary["reviewed_turns"], 2)
        self.assertEqual(summary["pass_rate"], 0.5)
        self.assertFalse(summary["review_complete"])
