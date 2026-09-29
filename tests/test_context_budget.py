from __future__ import annotations

import unittest

from app.core.context_budget import (
    BudgetPolicy,
    ContextBuilder,
    ContextBudgetBlocked,
    ContextItem,
    make_request_key,
)


class ExactTokenizer:
    tokenizer_id = "test-tokenizer"
    tokenizer_version = "1"

    def count(self, rendered: str) -> int:
        return len(rendered)


class ContextBudgetTests(unittest.TestCase):
    def test_fallback_normalizes_and_prioritizes_atomic_items(self):
        builder = ContextBuilder(BudgetPolicy(context_window_tokens=900, max_output_tokens=100))
        snapshot = builder.build(
            system="系统\r\n约束",
            items=[
                ContextItem("history", "历史" * 200, "history"),
                ContextItem("fact", "用户确认事实", "user_confirmed", required=True),
            ],
        )
        self.assertEqual(snapshot.estimator_mode, "conservative_fallback")
        self.assertIn("fact", snapshot.included_object_ids)
        self.assertTrue(snapshot.omitted_object_ids or not snapshot.blocked)
        self.assertNotIn("\r", snapshot.rendered_text)

    def test_unknown_window_blocks_before_call(self):
        snapshot = ContextBuilder(BudgetPolicy(None, 100)).build(system="x")
        self.assertTrue(snapshot.blocked)
        self.assertEqual(snapshot.blocked_reason, "context_window_unknown")
        with self.assertRaises(ContextBudgetBlocked):
            ContextBuilder(BudgetPolicy(None, 100)).build_or_raise(system="x")

    def test_missing_required_item_blocks(self):
        snapshot = ContextBuilder(BudgetPolicy(2000, 100)).build(
            system="x", required_object_ids=["evidence-1"]
        )
        self.assertEqual(snapshot.blocked_reason, "required_item_missing:evidence-1")

    def test_exact_tokenizer_and_request_key_are_stable(self):
        builder = ContextBuilder(BudgetPolicy(2000, 100), tokenizer=ExactTokenizer())
        kwargs = {
            "system": "s",
            "messages": [{"role": "user", "content": "中文"}],
            "items": [ContextItem("e1", "证据", "evidence")],
        }
        one = builder.build(**kwargs)
        two = builder.build(**kwargs)
        self.assertEqual(one.estimator_mode, "exact")
        self.assertEqual(one.tokenizer_id, "test-tokenizer")
        self.assertEqual(one.context_manifest_hash, two.context_manifest_hash)
        self.assertEqual(
            make_request_key(
                thread_id="t",
                node_name="n",
                model_profile_version="m1",
                snapshot=one,
            ),
            make_request_key(
                thread_id="t",
                node_name="n",
                model_profile_version="m1",
                snapshot=two,
            ),
        )

    def test_policy_can_be_created_from_api_profile(self):
        class Profile:
            context_window_tokens = 4096
            generation_params = {"max_output_tokens": 256}

        policy = BudgetPolicy.from_profile(Profile())
        self.assertEqual(policy.max_output_tokens, 256)
        self.assertEqual(policy.usable_input, 3328)

    def test_invalid_json_context_is_blocked(self):
        snapshot = ContextBuilder(BudgetPolicy(2000, 100)).build(tool_schemas=[{"bad": float("nan")}])
        self.assertTrue(snapshot.blocked)
        self.assertEqual(snapshot.blocked_reason, "invalid_context")


if __name__ == "__main__":
    unittest.main()
