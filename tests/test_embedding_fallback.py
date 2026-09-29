from __future__ import annotations

import unittest

from app.core.embedding_fallback import (
    CandidateText,
    TFIDF_ALGORITHM_VERSION,
    TfidfFallback,
    content_hash,
    deduplicate_candidates,
    deduplicate_with_vectors,
    tokenize,
)


class EmbeddingFallbackTests(unittest.TestCase):
    def test_tokenization_contains_chinese_ngrams_and_technical_terms(self):
        values = tokenize("中文 RAG + FastAPI")
        self.assertIn("cjk1:中", values)
        self.assertIn("cjk2:中文", values)
        self.assertIn("term:rag", values)
        self.assertIn("term:fastapi", values)

    def test_tfidf_is_deterministic_and_l2_normalized(self):
        vectorizer = TfidfFallback().fit(["智能简历 FastAPI", "天气查询"])
        left = vectorizer.transform(["智能简历 FastAPI"])[0]
        right = vectorizer.transform(["智能简历 FastAPI"])[0]
        self.assertEqual(left.values, right.values)
        self.assertAlmostEqual(sum(x * x for x in left.values.values()), 1.0)
        self.assertEqual(vectorizer.algorithm_version, TFIDF_ALGORITHM_VERSION)
        self.assertEqual(len(content_hash("a")), 64)

    def test_fallback_dedup_keeps_higher_match_score(self):
        candidates = [
            CandidateText("slot-1", "智能问答", "Python FastAPI", "RAG 检索", 70),
            CandidateText("slot-2", "知识库问答", "Python FastAPI", "RAG 检索", 90),
            CandidateText("slot-3", "旅游规划", "Vue", "地图", 80),
        ]
        result = deduplicate_candidates(candidates, threshold=0.2)
        self.assertIn("slot-2", {item.candidate_id for item in result.kept})
        self.assertIn("slot-1", {item.candidate_id for item in result.dropped})
        self.assertEqual(result.mode, "tfidf_fallback")

    def test_real_embedding_dedup_uses_independent_threshold(self):
        candidates = [CandidateText("a", "A", "", "", 1), CandidateText("b", "B", "", "", 2)]
        result = deduplicate_with_vectors(
            candidates,
            {"a": [1.0, 0.0], "b": [0.99, 0.01]},
            threshold=0.85,
        )
        self.assertEqual(len(result.kept), 1)
        self.assertEqual(result.threshold, 0.85)
        self.assertEqual(result.mode, "embedding")


if __name__ == "__main__":
    unittest.main()
