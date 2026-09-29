import unittest

from sglang.srt.managers.prefill_delayer import PrefillDelayer
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, published_topology

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

TOPOLOGIES = {
    "tp": dict(tp_size=4),
    "attn_cp": dict(tp_size=4, attn_cp_size=2),
    "dp_attention": dict(tp_size=4, dp_size=2, enable_dp_attention=True),
    "dp_attention_attn_cp": dict(
        tp_size=8, dp_size=2, attn_cp_size=2, enable_dp_attention=True
    ),
}


class TestPrefillDelayerGatherShape(CustomTestCase):
    def test_one_slot_per_rank_of_the_tp_group(self):
        for name, topology in TOPOLOGIES.items():
            with self.subTest(name), published_topology(**topology):
                delayer = PrefillDelayer(
                    cpu_group=None, max_delay_passes=4, token_usage_low_watermark=None
                )
                buffer = delayer._global_info_buffer
                self.assertEqual(
                    buffer.numel(), get_parallel().tp_size * buffer.shape[-1]
                )


if __name__ == "__main__":
    unittest.main()
