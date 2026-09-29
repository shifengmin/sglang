"""A Sarvam dense layer leaves its MLP output's sum to the MLP's own
row-parallel projection, which follows the FFN exit's decision. The decoder
must not add a second, attention-TP-gated all-reduce: under attention DP the
dense MLP spans the full TP group, and a dense MLP on local rows owes no sum."""

import importlib
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers import layer_boundary as comm
from sglang.srt.layers.layer_boundary.residual.stream import ResidualStream
from sglang.test.boundary_fixtures import stub_plan
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def dense_layer(attn_tp_size):
    sarvam = importlib.import_module("sglang.srt.models.sarvam_moe")
    layer = sarvam.SarvamMoEMLADecoderLayer.__new__(sarvam.SarvamMoEMLADecoderLayer)
    torch.nn.Module.__init__(layer)
    complete = MagicMock(side_effect=lambda h, r: (h, r))
    plan = stub_plan()
    plan.path_for = lambda fb: SimpleNamespace(
        output=comm.OutputContract(comm.Layout(frozenset())),
        returns_over_dp=False,
        output_move_completes_sum=False,
    )
    # Neither flag is published, so the MLP's projection completes the sum.
    plan.output._decide = lambda fb, steps: comm.ExitDecision(
        defer_moe_finalize=False,
        fuse_mlp_allreduce=False,
        mlp_reduce_scatter=False,
        complete=complete,
    )
    object.__setattr__(
        layer,
        "attn_boundary",
        SimpleNamespace(prepare=lambda h, fb, **kwargs: h, finish=lambda h, fb: h),
    )
    object.__setattr__(
        layer,
        "ffn_boundary",
        SimpleNamespace(
            prepare=lambda h, fb: h,
            exit=lambda fb: plan.output.ffn_exit(fb, stream=fb.residual_stream),
        ),
    )
    object.__setattr__(layer, "self_attn", lambda **kwargs: kwargs["hidden_states"])
    object.__setattr__(layer, "mlp", lambda h, fb: h * 3)
    object.__setattr__(layer, "is_layer_sparse", False)
    object.__setattr__(layer, "attn_tp_size", attn_tp_size)
    return sarvam, layer, complete


class TestSarvamDenseMlpReduction(CustomTestCase):
    def test_decoder_hands_the_mlp_output_to_the_exit_unchanged(self):
        for attn_tp_size in (1, 2):
            with self.subTest(attn_tp_size=attn_tp_size):
                sarvam, layer, complete = dense_layer(attn_tp_size)
                all_reduce = MagicMock(side_effect=lambda x: x * 2)
                with patch.object(
                    sarvam, "tensor_model_parallel_all_reduce", all_reduce, create=True
                ):
                    layer.forward(
                        positions=None,
                        hidden_states=torch.ones(2, 4),
                        forward_batch=SimpleNamespace(
                            residual_stream=ResidualStream(torch.ones(2, 4))
                        ),
                    )
                all_reduce.assert_not_called()
                torch.testing.assert_close(
                    complete.call_args.args[0], torch.full((2, 4), 3.0)
                )


if __name__ == "__main__":
    unittest.main()
