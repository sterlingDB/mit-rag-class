import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hybrid_retriever import HybridRetriever


class EmptyIndexTests(unittest.TestCase):
    def test_empty_index_explains_recovery_instead_of_dividing_by_zero(self):
        db = Mock()
        db.get.return_value = {"ids": [], "documents": [], "metadatas": []}
        with patch("hybrid_retriever.get_api_key", return_value="offline-placeholder"), \
             patch("hybrid_retriever.ChatOpenAI"):
            with self.assertRaisesRegex(RuntimeError, "empty.*chroma_db"):
                HybridRetriever(db=db)


if __name__ == "__main__":
    unittest.main()
