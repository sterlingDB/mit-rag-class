import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from test_production_ready import FakeLLM, agent
from semantic_answer_cache import SemanticAnswerCache, cache_namespace


SUPPORTED = {"action": "answer", "requirements": [
    {"question": "Fact", "article_id": "A", "quote": "Evidence A"}]}


class EfficientAgentTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.enterContext(patch.object(agent, "LOG_PATH", Path(directory.name) / "test.log"))
        self.enterContext(patch.object(agent, "USE_PERSONA_FILTER", False))
        self.enterContext(patch.object(agent, "USE_EXPENSIVE_FALLBACK", False))
        self.enterContext(redirect_stdout(io.StringIO()))
        self.retriever = Mock()
        self.retriever.getTopK.return_value = [("A", "Evidence A", 1.0, "bm25")]
        self.graph = Mock()
        self.cache = Mock()
        self.cache.embedding_calls = 0
        self.cache.embedding_seconds = 0.0
        self.cache.lookup.return_value = None

    def run_agent(self, model, judge, **kwargs):
        return agent.answer_efficient(model, self.retriever, self.graph, "Who won?",
                                      router_llm=judge, **kwargs)

    def test_direct_route_avoids_planner_and_stores_sources(self):
        result = self.run_agent(FakeLLM("Answer [A]"),
                                FakeLLM({"route": "direct"}, {"satisfactory": True}), cache=self.cache)
        self.assertEqual(result["path"], "direct")
        self.assertEqual(result["planner_calls"], 0)
        self.assertEqual(result["model_calls"], 3)
        self.cache.store.assert_called_once()
        self.assertEqual(result["evidence"], {"A": "Evidence A"})

    def test_failed_direct_answer_escalates_and_counts_both_attempts(self):
        model = FakeLLM("Partial answer", SUPPORTED, "Answer [A]")
        with patch.object(agent, "make_llm", side_effect=AssertionError("No expensive fallback")):
            result = self.run_agent(model, FakeLLM({"route": "direct"}, {"satisfactory": False}))
        self.assertEqual(result["path"], "direct-to-agent")
        self.assertEqual(result["search_calls"], 2)
        self.assertEqual(result["answer_calls"], 2)
        self.assertEqual(result["model_calls"], 5)
        self.assertEqual(result["usage"]["answer"]["test-model"]["calls"], 2)
        self.assertEqual(result["cost_accounting"]["overhead"]["calls"], 2)
        self.assertEqual(result["cost_accounting"]["answer_work"]["calls"], 3)

    def test_validated_cache_hit_skips_retrieval_but_counts_validation(self):
        candidate = {"question": "Who won?", "answer": "Answer [A]",
                     "sources": [{"article_id": "A"}], "evidence": {"A": "Evidence A"}}
        self.cache.lookup.return_value = candidate, "semantic"
        result = self.run_agent(FakeLLM(), FakeLLM({"valid": True}), cache=self.cache)
        self.assertEqual(result["path"], "cache-hit")
        self.assertEqual(result["model_calls"], 1)
        self.assertEqual(result["cache_match"], "semantic")
        self.retriever.getTopK.assert_not_called()
        self.assertEqual(result["sources"], candidate["sources"])
        self.assertEqual(result["cost_accounting"]["overhead"]["tokens"], 120)
        self.assertEqual(result["cost_accounting"]["answer_work"]["tokens"], 0)

    def test_router_outage_still_appears_in_final_accounting(self):
        judge = Mock(model_name="test-model")
        judge.invoke.side_effect = RuntimeError("offline")
        result = self.run_agent(FakeLLM(SUPPORTED, "Answer [A]"), judge)
        self.assertEqual(result["path"], "agent")
        self.assertEqual(result["model_calls"], 3)
        self.assertEqual(result["cost_accounting"]["overhead"]["failed_calls"], 1)
        self.assertIsNone(result["estimated_chat_cost_usd"])

    def test_direct_path_obeys_optimized_retrieval_budget(self):
        self.retriever.getTopK.return_value = [("A", "Evidence A " * 1000, 1.0, "bm25")]
        with patch.object(agent, "RUN_COST_OPTIMIZATIONS", True), \
             patch.object(agent, "OPTIMIZED_CONTEXT_CHARACTERS", 100):
            result = self.run_agent(FakeLLM("Answer [A]"),
                                    FakeLLM({"route": "direct"}, {"satisfactory": True}))
        self.retriever.getTopK.assert_called_once_with("Who won?", agent.OPTIMIZED_TOP_K)
        self.assertEqual(result["context_characters"], 100)

    def test_string_true_cannot_validate_cache_or_direct_answer(self):
        self.cache.lookup.return_value = ({"question": "Other", "answer": "Wrong [A]",
            "sources": [], "evidence": {"A": "Evidence A"}}, "semantic")
        model = FakeLLM("Bad [A]", SUPPORTED, "Answer [A]")
        judge = FakeLLM({"valid": "true"}, {"route": "direct"},
                        {"satisfactory": "true"}, {"satisfactory": False})
        result = self.run_agent(model, judge, cache=self.cache)
        self.assertEqual(result["path"], "direct-to-agent")
        self.cache.store.assert_not_called()

    def test_bad_router_json_defaults_to_agent(self):
        result = self.run_agent(FakeLLM(SUPPORTED, "Answer [A]"), FakeLLM("not json"))
        self.assertEqual(result["path"], "agent")

    def test_expensive_fallback_is_only_created_when_enabled(self):
        model = FakeLLM("Partial", SUPPORTED)
        fallback = FakeLLM("Answer [A]")
        with patch.object(agent, "USE_EXPENSIVE_FALLBACK", True), \
             patch.object(agent, "make_llm", return_value=fallback) as make:
            result = self.run_agent(model, FakeLLM({"route": "direct"}, {"satisfactory": False}))
        make.assert_called_once_with(agent.FALLBACK_MODEL)
        self.assertEqual(result["path"], "direct-to-agent")
        self.assertEqual(len(fallback.messages), 1)

    def test_disabled_router_and_cache_use_only_full_agent(self):
        result = self.run_agent(FakeLLM(SUPPORTED, "Answer [A]"), FakeLLM(), use_router=False)
        self.assertEqual(result["path"], "agent")
        self.assertEqual(result["model_calls"], 2)

    def test_history_bypasses_cache_and_router(self):
        result = self.run_agent(FakeLLM(SUPPORTED, "Answer [A]"), FakeLLM(), cache=self.cache,
                                conversation_history=[{"role": "user", "content": "Football"}])
        self.assertEqual(result["path"], "agent")
        self.cache.lookup.assert_not_called()
        self.cache.store.assert_not_called()

    def test_safety_filter_is_not_repeated_during_escalation(self):
        with patch.object(agent, "USE_PERSONA_FILTER", True):
            result = self.run_agent(FakeLLM("Partial", SUPPORTED, "Answer [A]"),
                FakeLLM({"route": "direct"}, {"satisfactory": False}),
                safety_filter_llm=FakeLLM("Who won?"))
        self.assertEqual(result["safety_filter_calls"], 1)
        self.assertEqual(result["model_calls"], 6)

    def test_cache_failure_does_not_discard_answer(self):
        self.cache.lookup.side_effect = RuntimeError("cache offline")
        self.cache.store.side_effect = RuntimeError("disk full")
        result = self.run_agent(FakeLLM("Answer [A]"),
            FakeLLM({"route": "direct"}, {"satisfactory": True}), cache=self.cache)
        self.assertEqual(result["answer"], "Answer [A]")


class SemanticCacheTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = directory.name
        self.embedder = Mock()
        self.embedder.embed_query.return_value = [1.0, 0.0, 0.0]
        self.cache = SemanticAnswerCache(self.path, self.embedder, "test_namespace", max_size=2)
        self.answer = {"answer": "Answer [A]", "sources": [{"article_id": "A"}],
                       "evidence": {"A": "Evidence A"}}

    def test_exact_semantic_and_distant_matches(self):
        self.cache.store("Who won?", self.answer)
        calls = self.cache.embedding_calls
        self.assertEqual(self.cache.lookup(" WHO  WON? ")[1], "exact")
        self.assertEqual(self.cache.embedding_calls, calls)
        self.assertEqual(self.cache.lookup("Who was the winner?")[1], "semantic")
        self.embedder.embed_query.return_value = [0.0, 1.0, 0.0]
        self.assertIsNone(self.cache.lookup("Unrelated"))

    def test_persistence_expiration_and_size_limit(self):
        with patch("semantic_answer_cache.time.time", return_value=100):
            self.cache.store("One", self.answer)
        with patch("semantic_answer_cache.time.time", return_value=101):
            self.cache.store("Two", self.answer)
        with patch("semantic_answer_cache.time.time", return_value=102):
            self.cache.store("Three", self.answer)
            self.assertEqual(self.cache.collection.count(), 2)
            self.assertFalse(self.cache.collection.get(ids=[self.cache.key("One")])["ids"])
            reopened = SemanticAnswerCache(self.path, self.embedder, "test_namespace")
            self.assertEqual(reopened.lookup("Three")[1], "exact")
        with patch("semantic_answer_cache.time.time", return_value=100000):
            self.assertIsNone(self.cache.lookup("Three"))

    def test_corpus_and_configuration_changes_invalidate_namespace(self):
        first = cache_namespace({"A": "Evidence"}, {"model": "one"})
        self.assertNotEqual(first, cache_namespace({"A": "Changed"}, {"model": "one"}))
        self.assertNotEqual(first, cache_namespace({"A": "Evidence"}, {"model": "two"}))
