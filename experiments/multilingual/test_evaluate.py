"""Check evaluation math against hand-computed retrieval examples."""

import unittest

import numpy as np
import torch
from evaluate import W8A8, metrics


class EvaluationTests(unittest.TestCase):
    def test_chunk_hits_are_ranked_once_per_document(self):
        # Both highest scoring chunks belong to document A; B must rank second.
        q = np.array([[1.0, 0.0]])
        d = np.array([[0.9, 0.0], [0.8, 0.0], [0.7, 0.0]])
        score, ranks = metrics(
            q, d, [{"_id": "q"}], [{"_id": "A"}, {"_id": "B"}], {"q": {"B"}}, [0, 0, 1]
        )
        self.assertEqual(ranks[0]["rank"], 2)
        self.assertEqual(score["hit@1"], 0.0)
        self.assertEqual(score["mrr"], 0.5)
        self.assertEqual(score["recall@5"], 1.0)
        self.assertAlmostEqual(score["ndcg@10"], 1 / np.log2(3))

    def test_multiple_positive_recall_uses_all_positives(self):
        q = np.array([[1.0]])
        d = np.arange(12, 0, -1).reshape(-1, 1)
        docs = [{"_id": str(i)} for i in range(12)]
        score, _ = metrics(q, d, [{"_id": "q"}], docs, {"q": {"0", "11"}})
        self.assertEqual(score["recall@5"], 0.5)
        self.assertEqual(score["recall@10"], 0.5)

    def test_w8a8_matches_explicit_integer_products_with_bias(self):
        torch.manual_seed(17)
        layer = torch.nn.Linear(16, 4, bias=True)
        x = torch.randn(3, 16)
        module = W8A8(layer)
        s = x.abs().amax(-1, keepdim=True) / 127
        q = (x / s).round().clamp(-127, 127).to(torch.int64)
        expected = (q @ module.w.to(torch.int64).T).float() * s * module.s + layer.bias
        torch.testing.assert_close(module(x), expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
