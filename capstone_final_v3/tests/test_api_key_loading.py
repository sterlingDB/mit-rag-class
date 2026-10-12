import os
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chroma_helpers
from hybrid_retriever import HybridRetriever


class ApiKeyLoadingTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.object(chroma_helpers, "load_dotenv"))
        self.enterContext(patch.dict(os.environ, {}, clear=True))

    def test_embeddings_read_key_set_after_import(self):
        with patch.object(chroma_helpers, "API_KEY", "stale-test-key"), \
             patch.object(chroma_helpers, "OpenAIEmbeddings") as create:
            os.environ["OPENROUTER_API_KEY"] = "current-test-key"
            chroma_helpers.get_embeddings()
        self.assertEqual(create.call_args.kwargs["api_key"], "current-test-key")

    def test_missing_or_blank_key_fails_before_client_creation(self):
        for value in ("", "   "):
            with self.subTest(value=value), patch.object(chroma_helpers, "OpenAIEmbeddings") as create:
                os.environ["OPENROUTER_API_KEY"] = value
                with self.assertRaisesRegex(RuntimeError, "missing or empty"):
                    chroma_helpers.get_embeddings()
                create.assert_not_called()

    def test_retriever_reads_current_key(self):
        os.environ["OPENROUTER_API_KEY"] = "current-test-key"
        db = Mock()
        db.get.return_value = {"ids": ["A"], "documents": ["Sample article"], "metadatas": [{}]}
        with patch("hybrid_retriever.ChatOpenAI") as create, patch("builtins.print"):
            HybridRetriever(db=db)
        self.assertEqual(create.call_args.kwargs["api_key"], "current-test-key")


if __name__ == "__main__":
    unittest.main()
