import unittest

import numpy as np
from answer_spans import contains, make_chunks, measure, sentence_bounds


class CharacterTokenizer:
    def num_special_tokens_to_add(self):
        return 2

    def encode(self, text):
        return [0] * (len(text) + 2)

    def __call__(self, text, **kwargs):
        return {"offset_mapping": [(i, i + 1) for i in range(len(text))]}


class AnswerSpanTests(unittest.TestCase):
    def test_answer_requires_complete_span_and_correct_parent(self):
        c = {"parent": 1, "start": 3, "end": 8}
        self.assertTrue(contains(c, 1, [(4, 8)]))
        self.assertFalse(contains(c, 1, [(4, 9)]))
        self.assertFalse(contains(c, 2, [(4, 8)]))

    def test_sentence_is_more_than_answer_presence(self):
        text = "最初。答えは東京です。最後。"
        at = text.index("東京")
        self.assertEqual(
            sentence_bounds(text, at, at + 2), (3, text.index("。", 3) + 1)
        )

    def test_splits_preserve_text_even_with_long_sentences(self):
        doc = {"title": "題", "context": "短文。" * 80 + "長" * 400 + "。終わり。"}
        for method in ["tokens", "sentence_end"]:
            cs = make_chunks(CharacterTokenizer(), [doc], 128, method)
            self.assertTrue(all(c["tokens"] <= 128 for c in cs))
            self.assertEqual(cs[-1]["end"], len(doc["title"] + "\n" + doc["context"]))

    def test_parent_hit_does_not_imply_answer_hit(self):
        qs = [
            {
                "id": "q",
                "type": "test",
                "parent": 0,
                "spans": [(5, 7)],
                "sentences": [(3, 9)],
            }
        ]
        cs = [
            {"parent": 0, "start": 0, "end": 4, "tokens": 128},
            {"parent": 1, "start": 0, "end": 10, "tokens": 128},
            {"parent": 0, "start": 4, "end": 8, "tokens": 128},
            {"parent": 0, "start": 3, "end": 9, "tokens": 128},
        ]
        scores = np.array([[1.0, 0.9, 0.8, 0.7]])
        r = measure(scores, np.eye(4), qs, cs, context_budget=256)["metrics"]
        self.assertEqual(r["chunk_parent@5"], 1)
        self.assertEqual(r["answer@1"], 0)
        self.assertEqual(r["answer@5"], 1)
        self.assertEqual(r["answer_budget256"], 0)
        self.assertEqual(r["sentence@5"], 1)


if __name__ == "__main__":
    unittest.main()
