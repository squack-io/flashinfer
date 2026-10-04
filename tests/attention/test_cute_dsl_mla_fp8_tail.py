# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regression test: invalid tail rows of a partial MLA KV page must not
poison the P.V contraction of the monolithic CuTe FP8 MLA DCP decode kernel.

The kernel iterates ceil(K / rows_per_tile) K-tiles, so the last tile can be
partial. Its V smem stage historically contained the page's full physical
rows, i.e. bytes beyond the sequence's logical length K. Those rows have
P == 0 after score masking, but the P.V MMA had no tail predicate and
0 * NaN = NaN for an e4m3 NaN payload, so stale bytes left in a reused page
would poison every output row. This test pins the fix.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (10, 3),
    reason="Requires SM103 (Blackwell Ultra); reproducer validated on B300",
)

import flashinfer

DEV = "cuda:0"
B, S, H, DQK = 1, 8, 96, 576
LORA, ROPE = 512, 64
PAGE, NPAGES = 64, 4
WORLD = 8
NAN_BYTE = 0x7F  # e4m3 NaN payload
FIN_BYTE = 0x30  # e4m3 ~0.5, finite


def _local_len(g, r):
    return g // WORLD + (1 if r < g % WORLD else 0)


def _run(g, rank, region):
    L = _local_len(g, rank)
    kv = torch.zeros(NPAGES, 1, PAGE, DQK, dtype=torch.uint8, device=DEV)
    kv[0, 0, :L, :] = FIN_BYTE
    if region == "latent":
        kv[0, 0, L:, :LORA] = NAN_BYTE
    elif region == "rope":
        kv[0, 0, L:, LORA:] = NAN_BYTE
    elif region == "valid_row":
        kv[0, 0, L - 1, :LORA] = NAN_BYTE
    elif region == "other_page":
        kv[2, 0, :, :] = NAN_BYTE
    q = torch.full((B, S, H, DQK), FIN_BYTE, dtype=torch.uint8, device=DEV).view(
        torch.float8_e4m3fn
    )
    bt = torch.tensor([[0, -1]], dtype=torch.int32, device=DEV)
    sl = torch.tensor([L], dtype=torch.int32, device=DEV)
    gl = torch.tensor([g], dtype=torch.int32, device=DEV)
    ws = torch.zeros(128 * 1024 * 1024, dtype=torch.int8, device=DEV)
    out, _ = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
        query=q,
        kv_cache=kv.view(torch.float8_e4m3fn),
        workspace_buffer=ws,
        qk_nope_head_dim=128,
        kv_lora_rank=LORA,
        qk_rope_head_dim=ROPE,
        block_tables=bt,
        seq_lens=sl,
        max_seq_len=PAGE,
        bmm1_scale=float(DQK) ** -0.5,
        backend="cute-dsl",
        enable_dcp=True,
        cp_world=WORLD,
        cp_rank=rank,
        causal_seqlens_kv_global=gl,
        return_lse=True,
    )
    torch.cuda.synchronize()
    return out.float()


@pytest.mark.parametrize("region", ["latent", "rope", "other_page"])
@pytest.mark.parametrize("g", [42, 200])
def test_invalid_tail_rows_do_not_poison_output(g, region):
    """NaN bytes in a page region outside the logical sequence must yield
    finite output (zeros -> 0, NaN -> previously poisoned)."""
    out = _run(g, 0, region)
    assert not torch.isnan(out).any(), (
        f"g={g} region={region}: invalid/tail NaN leaked into output "
        f"({int(torch.isnan(out).sum())}/{out.numel()} NaN)"
    )


@pytest.mark.parametrize("rank", list(range(WORLD)))
def test_invalid_tail_latent_all_ranks_clean(rank):
    out = _run(42, rank, "latent")
    assert not torch.isnan(out).any()


def test_valid_row_nan_still_detected():
    """Sanity: a NaN in a *valid* row must still propagate (proves the test
    is not trivially masking everything)."""
    out = _run(42, 0, "valid_row")
    assert torch.isnan(out).any()
