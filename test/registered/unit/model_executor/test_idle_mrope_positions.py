"""Idle-batch ``mrope_positions`` materialization in ``ForwardBatch.init_new``.

The idle early-return in ``ForwardBatch.init_new`` used to leave
``mrope_positions`` at its ``None`` default while ``positions`` received an
empty tensor. Mrope models (the Qwen3-VL family etc.) replace ``positions``
with ``forward_batch.mrope_positions`` inside the model forward, so any
*eager* full-model forward on an idle batch crashed on
``mrope_positions.dim()``.

Two dp-attention paths turn an idle batch into a full eager model forward:
MLP-sync padding rewrites hybrid-SSM + spec idle batches from IDLE to
TARGET_VERIFY, and DFLASH runs eager lockstep idle verifies on idle DP ranks
(reproduced on H200 with Qwen3.8-27B-FP8 + DFlash2 draft, tp2/dp2: first
request crashes on the idle rank). Graph replay is unaffected -- load_batch
does not copy ``mrope_positions`` -- so the fix is to materialize the empty
``(3, 0)`` mrope layout for mrope models before the early return.

These tests drive the real ``init_new`` with mocked ScheduleBatch and
ModelRunner stand-ins (CPU only, no GPU) and lock the idle-branch contract:
empty ``positions`` for every model, and empty ``(3, 0)``
``mrope_positions`` exactly for mrope models.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.model_executor import forward_batch_info
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _idle_batch() -> SimpleNamespace:
    """Minimal ScheduleBatch stand-in for an idle dp-attention rank.

    Covers every attribute init_new touches before the idle early-return.
    """
    empty64 = torch.empty(0, dtype=torch.int64)
    return SimpleNamespace(
        forward_mode=ForwardMode.IDLE,
        # Core tensors, all empty on an idle rank.
        seq_lens=empty64,
        seq_lens_cpu=torch.empty(0, dtype=torch.int64),
        seq_lens_sum=0,
        input_ids=empty64,
        req_pool_indices=empty64,
        out_cache_loc=empty64,
        orig_seq_lens=torch.empty(0, dtype=torch.int32),
        # Optional compound fields.
        out_cache_loc_dsv4=None,
        engram_history=None,
        mamba_track_indices=None,
        mamba_track_mask=None,
        mamba_track_seqlens=None,
        mamba_prefill_track_mask_cpu=None,
        mamba_track_seqlens_cpu=None,
        mamba_cow_src_indices=None,
        mamba_cow_dst_indices=None,
        mamba_clear_indices=None,
        encoder_lens=None,
        encoder_out_cache_loc=None,
        input_embeds=None,
        replace_embeds=None,
        replace_positions=None,
        tbo_split_seq_index=None,
        # Flags and host-side metadata.
        return_logprob=False,
        is_extend_in_batch=False,
        can_run_decode_cuda_graph=False,
        can_run_dp_draft_cuda_graph=False,
        can_run_dp_prefill_cuda_graph=False,
        dp_prefill_cuda_graph_max_prefix_len=None,
        global_forward_mode=ForwardMode.IDLE,
        is_prefill_only=False,
        spec_algorithm=None,
        top_logprobs_nums=None,
        token_ids_logprobs=None,
        multimodal_inputs=None,
        encoder_cached=None,
        encoder_lens_cpu=None,
        reqs=[],
        sampling_info=None,
        spec_info=None,
        has_grammar=False,
        extend_input_logprob_token_ids=None,
        # init_mlp_sync_metadata inputs: None global counts -> early return.
        global_num_tokens=None,
        global_num_tokens_for_logprob=None,
        draft_global_num_tokens=None,
        draft_global_num_tokens_for_logprob=None,
        dp_spec_prefill_coordination_applied=False,
    )


def _model_runner(model_is_mrope: bool) -> SimpleNamespace:
    return SimpleNamespace(
        device=torch.device("cpu"),
        is_draft_worker=False,
        lora_manager=None,
        model_config=SimpleNamespace(
            model_is_mrope=model_is_mrope,
            requires_mm_token_modalities=False,
            is_matryoshka=False,
            hidden_size=8,
        ),
        kv_index_translator=SimpleNamespace(rebind_write_loc=lambda fb: None),
    )


class TestIdleMropePositions(CustomTestCase):
    def _init_idle(self, model_is_mrope: bool) -> ForwardBatch:
        # enable_num_token_non_padded() reads the published parallel config,
        # which a bare test process does not have; the idle branch must not
        # depend on it either way.
        with patch.object(
            forward_batch_info, "enable_num_token_non_padded", return_value=False
        ):
            return ForwardBatch.init_new(
                _idle_batch(),
                _model_runner(model_is_mrope),
                capture_hidden_mode=CaptureHiddenMode.FULL,
                return_hidden_states_before_norm=False,
            )

    def test_mrope_model_gets_empty_mrope_positions(self):
        fb = self._init_idle(model_is_mrope=True)
        # The crashing expression in the wild was positions.dim() on None.
        self.assertIsNotNone(fb.mrope_positions)
        self.assertEqual(fb.mrope_positions.dim(), 2)
        self.assertEqual(tuple(fb.mrope_positions.shape), (3, 0))
        self.assertEqual(fb.mrope_positions.dtype, torch.int64)

    def test_non_mrope_model_keeps_none(self):
        fb = self._init_idle(model_is_mrope=False)
        self.assertIsNone(fb.mrope_positions)

    def test_positions_contract_unchanged(self):
        for model_is_mrope in (True, False):
            fb = self._init_idle(model_is_mrope=model_is_mrope)
            self.assertTrue(fb.forward_mode.is_idle())
            self.assertEqual(tuple(fb.positions.shape), (0,))
            self.assertEqual(fb.positions.dtype, torch.int64)


if __name__ == "__main__":
    unittest.main()
