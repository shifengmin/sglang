import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.layers.quantization.moe_wna16 import MoeWNA16Method
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, published_topology

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

HIDDEN = 16
INTERMEDIATE = 16
GROUP = 8
MOE_TP_SIZE = 2


def _layer(moe_tp_rank):
    quant_config = SimpleNamespace(
        has_zp=True, linear_quant_method="gptq", weight_bits=4
    )
    return SimpleNamespace(
        quant_config=quant_config,
        moe_tp_size=MOE_TP_SIZE,
        moe_tp_rank=moe_tp_rank,
        intermediate_size_per_partition=INTERMEDIATE // MOE_TP_SIZE,
        group_size_div_factor=1,
    )


def _load_w2_qzeros(layer):
    # GPTQ int4 down_proj qzeros: (intermediate // group, hidden // 8) int32.
    checkpoint = torch.arange(
        (INTERMEDIATE // GROUP) * (HIDDEN // 8), dtype=torch.int32
    ).view(INTERMEDIATE // GROUP, HIDDEN // 8)
    checkpoint = checkpoint * 0x01010101
    param = torch.nn.Parameter(
        torch.zeros(
            1,
            HIDDEN // 2,
            layer.intermediate_size_per_partition // GROUP,
            dtype=torch.uint8,
        ),
        requires_grad=False,
    )
    loader = MoeWNA16Method.get_weight_loader(layer, Mock())
    loader(param, checkpoint, "experts.w2_qzeros", "w2", 0)
    return param.data.clone()


@contextmanager
def _topology(*, tp_size, ep_size, world_rank):
    with published_topology(
        tp_size=tp_size, ep_size=ep_size, ranks=dict(world_rank=world_rank)
    ):
        device_group = SimpleNamespace(world_size=tp_size, device=torch.device("cpu"))
        with get_parallel().override(tp_group=device_group):
            yield


class TestMoeWNA16QzerosShardUnderEP(CustomTestCase):
    def setUp(self):
        # Without EP the TP rank and the MoE-TP rank coincide.
        with _topology(tp_size=MOE_TP_SIZE, ep_size=1, world_rank=1):
            self.expected = _load_w2_qzeros(_layer(moe_tp_rank=1))
            self.other_shard = _load_w2_qzeros(_layer(moe_tp_rank=0))

    def test_the_two_shards_differ(self):
        self.assertFalse(torch.equal(self.expected, self.other_shard))

    def test_ep_layout_loads_the_moe_tp_shard(self):
        # TP 4 with EP 2: TP rank 3 holds MoE-TP shard 1 of 2.
        with _topology(tp_size=4, ep_size=2, world_rank=3):
            self.assertEqual(get_parallel().tp_rank, 3)
            loaded = _load_w2_qzeros(_layer(moe_tp_rank=1))
        self.assertTrue(torch.equal(loaded, self.expected))


if __name__ == "__main__":
    unittest.main()
