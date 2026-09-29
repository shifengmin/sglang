"""An FFN output computed in parts (routed and shared experts) completes the
sum each part owes once, following the FFN exit's decision."""

import importlib
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers import layer_boundary as comm
from sglang.srt.layers.layer_boundary.residual.stream import ResidualStream
from sglang.srt.runtime_context import get_forward
from sglang.test.boundary_fixtures import stub_plan
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

SumGroup = comm.SumGroup


def groups(rank=0, attn_ranks=(0, 1)):
    made = {
        SumGroup.TP: SimpleNamespace(ranks=[0, 1], rank_in_group=rank),
        SumGroup.MOE_OUTPUT: SimpleNamespace(ranks=[0, 1], rank_in_group=rank),
        SumGroup.ATTN_TP: SimpleNamespace(ranks=list(attn_ranks), rank_in_group=rank),
    }
    for group in made.values():
        group.all_reduce = MagicMock(side_effect=lambda x: x * 2)
    return made


def ffn_exit(*, skips, declared=SumGroup.MOE_OUTPUT):
    complete = MagicMock(side_effect=lambda h, r: (h, r))
    plan = stub_plan()
    plan.path_for = lambda fb: SimpleNamespace(
        output=comm.OutputContract(comm.Layout(frozenset()), group=declared),
        returns_over_dp=False,
        output_move_completes_sum=False,
    )
    plan.output._decide = lambda fb, steps: comm.ExitDecision(
        defer_moe_finalize=False,
        fuse_mlp_allreduce=skips,
        mlp_reduce_scatter=False,
        complete=complete,
    )
    fb = SimpleNamespace(residual_stream=ResidualStream(torch.ones(2, 4)))
    return plan.output.ffn_exit(fb, stream=fb.residual_stream), complete


class TestCompleteParts(CustomTestCase):
    def setUp(self):
        self.routed = torch.full((2, 4), 2.0)
        self.shared = torch.full((2, 4), 3.0)
        self.extra = torch.full((2, 4), 5.0)

    def test_compute_completes_one_sum_per_group(self):
        made = groups()
        with patch.object(comm.exit, "_sum_group", made.__getitem__):
            scope, _ = ffn_exit(skips=False)
            total = scope.complete_parts(
                (self.routed, SumGroup.TP),
                (self.shared, SumGroup.TP),
                (self.extra, None),
            )
        torch.testing.assert_close(total, 2 * (self.routed + self.shared) + self.extra)
        made[SumGroup.TP].all_reduce.assert_called_once()

    def test_a_later_sum_counts_a_complete_part_once(self):
        for rank, expected in ((0, 10.0), (1, 5.0)):
            with self.subTest(rank=rank):
                made = groups(rank)
                with patch.object(comm.exit, "_sum_group", made.__getitem__):
                    scope, _ = ffn_exit(skips=True)
                    total = scope.complete_parts(
                        (self.routed, SumGroup.TP),
                        (self.shared, SumGroup.TP),
                        (self.extra, None),
                    )
                torch.testing.assert_close(total, torch.full((2, 4), expected))
                for group in made.values():
                    group.all_reduce.assert_not_called()

    def test_a_part_owing_another_group_is_rejected_when_the_sum_is_left(self):
        made = groups(attn_ranks=(0,))
        with patch.object(comm.exit, "_sum_group", made.__getitem__):
            scope, _ = ffn_exit(skips=True)
            with self.assertRaises(NotImplementedError):
                scope.complete_parts((self.routed, SumGroup.ATTN_TP))

    def test_separate_parts_skips_compute_reductions_inside_the_exit(self):
        scope, _ = ffn_exit(skips=False)
        with scope:
            self.assertFalse(get_forward().fuse_mlp_allreduce)
            with scope.separate_parts():
                self.assertTrue(get_forward().fuse_mlp_allreduce)
            self.assertFalse(get_forward().fuse_mlp_allreduce)


class TestStep3p5MoeParts(CustomTestCase):
    def layer(self, routed_owes):
        step3p5 = importlib.import_module("sglang.srt.models.step3p5")
        layer = step3p5.Step3p5DecoderLayer.__new__(step3p5.Step3p5DecoderLayer)
        torch.nn.Module.__init__(layer)
        seen = {}

        def moe(hidden, fb):
            # The MoE block's own reduction is skipped inside separate_parts().
            seen["skips"] = get_forward().fuse_mlp_allreduce
            return torch.full_like(hidden, 2.0)

        exit_scope, complete = ffn_exit(skips=False)
        object.__setattr__(
            layer,
            "attn_boundary",
            SimpleNamespace(prepare=lambda h, fb, **kw: h, finish=lambda h, fb: h),
        )
        object.__setattr__(
            layer,
            "ffn_boundary",
            SimpleNamespace(prepare=lambda h, fb: h, exit=lambda fb: exit_scope),
        )
        object.__setattr__(layer, "self_attn", lambda **kw: kw["hidden_states"])
        object.__setattr__(layer, "use_moe", True)
        object.__setattr__(layer, "moe", _Callable(moe, routed_owes))
        object.__setattr__(layer, "share_expert", lambda h: torch.full_like(h, 3.0))
        return step3p5, layer, complete, seen

    def test_routed_and_shared_parts_share_one_all_reduce(self):
        for routed_owes, expected in ((SumGroup.TP, 10.0), (None, 8.0)):
            with self.subTest(routed_owes=routed_owes):
                step3p5, layer, complete, seen = self.layer(routed_owes)
                made = groups()
                with (
                    patch.object(comm.exit, "_sum_group", made.__getitem__),
                    patch.object(
                        step3p5, "get_parallel", return_value=SimpleNamespace(tp_size=2)
                    ),
                ):
                    layer.forward(
                        positions=None,
                        hidden_states=torch.ones(2, 4),
                        forward_batch=None,
                    )
                self.assertTrue(seen["skips"])
                made[SumGroup.TP].all_reduce.assert_called_once()
                torch.testing.assert_close(
                    complete.call_args.args[0], torch.full((2, 4), expected)
                )


class _Callable:
    def __init__(self, fn, owes):
        self.fn, self.owes = fn, owes

    def __call__(self, *args):
        return self.fn(*args)

    def output_sum(self):
        return self.owes


if __name__ == "__main__":
    unittest.main()
