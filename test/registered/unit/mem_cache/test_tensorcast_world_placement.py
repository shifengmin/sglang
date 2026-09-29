import unittest

from sglang.srt.mem_cache.storage.tensorcast_store.host_allocator import (
    resolve_tensorcast_world_placement,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, enter_scope, published_topology

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestTensorcastWorldPlacement(CustomTestCase):
    def test_reads_this_process_placement_from_the_context(self):
        enter_scope(self, published_topology(tp_size=4, ranks=dict(world_rank=3)))

        self.assertEqual(resolve_tensorcast_world_placement(None, None), (3, 4))

    def test_explicit_placement_is_returned_unchanged(self):
        self.assertEqual(resolve_tensorcast_world_placement(1, 2), (1, 2))

    def test_rank_and_size_come_together(self):
        with self.assertRaisesRegex(ValueError, "provided together"):
            resolve_tensorcast_world_placement(0, None)


if __name__ == "__main__":
    unittest.main()
