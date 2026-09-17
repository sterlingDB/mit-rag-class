"""Run: .venv/bin/python -m unittest discover -s capstone/tests"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from graph_retriever import GraphRetriever, article_id_from_link


class FakeRetriever:
    def get_documents(self):
        return {name: f'Text for {name}' for name in ['A', 'B', 'C', 'D', 'E']}

    def getTopK(self, query, k):
        return [(name, f'Text for {name}', 0.8, 'bm25+vector') for name in ['A', 'B', 'C']][:k]

    def score_documents(self, query, doc_ids):
        return {name: (2.0 if query == 'relevant' else 0.0) for name in doc_ids}


class GraphTests(unittest.TestCase):
    def test_url_normalization(self):
        for url in ['/wiki/Ben_Jones%20(actor)#Life', './Ben_Jones_(actor)',
                    'Ben_Jones_(actor).html', 'https://en.wikipedia.org/wiki/Ben_Jones_(actor)']:
            self.assertEqual(article_id_from_link(url), 'Ben_Jones_(actor)')
        for url in ['#History', '/wiki/Category:Actors', 'https://example.com/wiki/A',
                    '/w/index.php?title=A', 'javascript:alert(1)']:
            self.assertIsNone(article_id_from_link(url))

    def test_expansion_budget_and_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'A.html').write_text(
                '<a href="/wiki/C">navigation</a><main><div id="mw-content-text">'
                '<a href="/wiki/D">D</a><a href="/wiki/D#Life">duplicate</a>'
                '<a href="/wiki/A">self</a><a href="/wiki/Missing">missing</a>'
                '</div></main><a href="/wiki/C">footer</a>')
            (path / 'D.html').write_text('<a href="/wiki/E">second hop</a>')
            graph = GraphRetriever(FakeRetriever(), path)
            self.assertEqual(set(graph.graph.successors('A')), {'D'})
            hits = graph.getTopK('relevant', 3)
            self.assertEqual([hit[0] for hit in hits], ['A', 'B', 'D'])
            self.assertIn('linked from A', hits[2][3])
            self.assertEqual(hits[2][1], 'Text for D')
            self.assertEqual([h[0] for h in graph.getTopK('unrelated', 3)], ['A', 'B', 'C'])
            self.assertEqual(graph.getTopK('relevant', 0), [])
            self.assertEqual(len(graph.getTopK('relevant', 1)), 1)
            self.assertEqual(graph.getTopK('relevant', 3, seed_hits=[]), [])
            seeds = FakeRetriever().getTopK('relevant', 3)
            self.assertEqual(graph.getTopK('relevant', 3, seed_hits=seeds + seeds), hits)


if __name__ == '__main__':
    unittest.main()
