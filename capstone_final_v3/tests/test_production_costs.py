import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain_core.messages import HumanMessage, SystemMessage
from production_costs import account_for_costs
from security_cost_helpers import AuditedLLM


class CostAccountingTests(unittest.TestCase):
    def invoke(self, usage, role, model):
        return AuditedLLM(model, role, usage, []).invoke([
            SystemMessage(content="Instructions"), HumanMessage(content="Question")])

    def test_failed_calls_are_counted_timed_and_cost_is_unknown(self):
        model = Mock(model_name="small")
        model.invoke.side_effect = RuntimeError("provider failed")
        usage = {}
        with patch("security_cost_helpers.perf_counter", side_effect=[10.0, 10.25]):
            with self.assertRaises(RuntimeError):
                self.invoke(usage, "router", model)
        report = account_for_costs({"usage": usage}, {"small": {"input": 1, "output": 2}})
        totals = report["cost_accounting"]["overhead"]
        self.assertEqual(totals["calls"], 1)
        self.assertEqual(totals["failed_calls"], 1)
        self.assertEqual(totals["missing_usage_calls"], 1)
        self.assertEqual(totals["elapsed_seconds"], 0.25)
        self.assertIsNone(report["estimated_chat_cost_usd"])
        self.assertTrue(report["cost_accounting"]["incomplete_estimates"])

    def test_shared_model_is_separated_by_role_and_cached_tokens_not_added_twice(self):
        model = Mock(model_name="small")
        model.invoke.return_value = SimpleNamespace(usage_metadata={
            "input_tokens": 100, "output_tokens": 20, "input_token_details": {"cache_read": 40}})
        usage = {}
        self.invoke(usage, "answer", model)
        self.invoke(usage, "cache_validation", model)
        result = account_for_costs({"usage": usage}, {
            "small": {"input": 1, "output": 2, "cached": 0.5}})
        report = result["cost_accounting"]
        self.assertEqual(report["total"]["tokens"], 240)
        self.assertEqual(report["overhead"]["tokens"], 120)
        self.assertEqual(report["answer_work"]["tokens"], 120)
        self.assertAlmostEqual(result["estimated_chat_cost_usd"], 0.00024)
        self.assertEqual(result["estimated_total_cost_usd"], result["estimated_chat_cost_usd"])

    def test_embeddings_prevent_chat_only_estimate_becoming_total(self):
        result = account_for_costs({"usage": {}, "search_calls": 2, "cache_embedding_calls": 1}, {})
        self.assertEqual(result["estimated_chat_cost_usd"], 0)
        self.assertIsNone(result["estimated_total_cost_usd"])
        self.assertIsNone(result["cost_accounting"]["embeddings"]["tokens"])
        self.assertEqual(result["cost_accounting"]["embeddings"]["retrieval_queries"], 2)

    def test_unknown_price_and_missing_usage_are_never_reported_as_free(self):
        for meta in (None, {"input_tokens": 10, "output_tokens": 2}):
            with self.subTest(meta=meta):
                usage = {}
                model = Mock(model_name="unpriced")
                model.invoke.return_value = SimpleNamespace(usage_metadata=meta)
                self.invoke(usage, "safety_filter", model)
                result = account_for_costs({"usage": usage}, {})
                self.assertIsNone(result["estimated_chat_cost_usd"])
                self.assertTrue(result["cost_accounting"]["incomplete_estimates"])
