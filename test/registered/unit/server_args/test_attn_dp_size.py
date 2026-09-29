"""--attn-dp-size gives the attention data-parallel width directly; the
deprecated --enable-dp-attention spelling resolves to the same configuration."""

import json
import logging
import os
import shutil
import tempfile
import unittest

from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

# A real config.json, so resolution runs past the dummy-model early return.
_MINI_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_attention_heads": 8,
    "num_key_value_heads": 8,
    "num_hidden_layers": 2,
    "vocab_size": 128,
    "max_position_embeddings": 2048,
}


class TestAttnDpSize(CustomTestCase):
    def setUp(self):
        super().setUp()
        environment = dict(os.environ)

        def restore():
            os.environ.clear()
            os.environ.update(environment)

        self.addCleanup(restore)
        self.path = tempfile.mkdtemp(prefix="attn_dp_size_")
        self.addCleanup(shutil.rmtree, self.path, ignore_errors=True)
        with open(os.path.join(self.path, "config.json"), "w") as handle:
            json.dump(_MINI_CONFIG, handle)

    def resolve(self, **fields):
        server_args = ServerArgs(
            model_path=self.path, device="cuda", random_seed=42, **fields
        )
        server_args.resolve_once()
        return server_args

    def field(self, server_args, name):
        return resolution_result(server_args, name)

    def test_new_spelling_resolves_like_the_deprecated_one(self):
        for tp_size, dp_size in ((2, 2), (4, 4), (4, 2), (8, 4)):
            with self.subTest(tp_size=tp_size, dp_size=dp_size):
                legacy = self.resolve(
                    tp_size=tp_size, dp_size=dp_size, enable_dp_attention=True
                )
                new = self.resolve(tp_size=tp_size, attn_dp_size=dp_size)
                self.assertEqual(legacy.resolved_dict(), new.resolved_dict())
                self.assertEqual(self.field(new, "dp_size"), dp_size)
                self.assertEqual(self.field(new, "attn_dp_size"), dp_size)
                self.assertTrue(self.field(new, "enable_dp_attention"))

    def test_the_deprecated_spelling_warns_with_its_replacement(self):
        with self.assertLogs("sglang.srt.arg_groups.parallel_hook", "WARNING") as logs:
            self.resolve(tp_size=2, dp_size=2, enable_dp_attention=True)
        warnings = [line for line in logs.output if "--enable-dp-attention" in line]
        self.assertEqual(len(warnings), 1)
        self.assertIn("--attn-dp-size 2", warnings[0])

    def test_without_attention_dp_the_width_is_one(self):
        for fields in (
            {"tp_size": 2},
            {"tp_size": 2, "dp_size": 2},
            {"tp_size": 2, "dp_size": 2, "attn_dp_size": 1},
            # A single DP group turns the deprecated flag off, as before.
            {"tp_size": 2, "dp_size": 1, "enable_dp_attention": True},
        ):
            with self.subTest(**fields):
                server_args = self.resolve(**fields)
                self.assertEqual(self.field(server_args, "attn_dp_size"), 1)
                self.assertFalse(self.field(server_args, "enable_dp_attention"))
                self.assertEqual(
                    self.field(server_args, "dp_size"), fields.get("dp_size", 1)
                )

    def test_replica_and_attention_data_parallelism_cannot_combine(self):
        for fields in (
            {"tp_size": 4, "dp_size": 2, "attn_dp_size": 4},
            {
                "tp_size": 4,
                "dp_size": 2,
                "attn_dp_size": 4,
                "enable_dp_attention": True,
            },
            {
                "tp_size": 2,
                "dp_size": 2,
                "attn_dp_size": 1,
                "enable_dp_attention": True,
            },
            {"tp_size": 2, "attn_dp_size": 0},
        ):
            with self.subTest(**fields):
                with self.assertRaises(ValueError):
                    self.resolve(**fields)


if __name__ == "__main__":
    logging.basicConfig()
    unittest.main()
