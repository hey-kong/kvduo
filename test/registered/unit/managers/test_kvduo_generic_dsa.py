"""Generic DSA shares complete physical pages across HOT storage layers."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
from sglang.srt.mem_cache.allocator.hisparse import HiSparseTokenToKVPoolAllocator
from sglang.srt.mem_cache.hisparse_memory_pool import HiSparseDSATokenToKVPool
from sglang.test.ci.ci_register import register_cpu_ci


register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def make_pool(*, layers=3, dtype=torch.bfloat16, kvduo=True, start_layer=0):
    return HiSparseDSATokenToKVPool(
        size=1024,
        page_size=64,
        kv_lora_rank=12,
        dtype=dtype,
        qk_rope_head_dim=4,
        layer_num=layers,
        device="cpu",
        index_head_dim=128,
        enable_memory_saver=False,
        kv_cache_dim=22 if dtype == torch.float8_e4m3fn else 16,
        start_layer=start_layer,
        enable_kvduo=kvduo,
    )


def make_stack(monkeypatch, *, layers=3, top_k=64, start_layer=0, dtype=torch.bfloat16):
    import sglang.srt.managers.hisparse_coordinator as module

    pool = make_pool(layers=layers, start_layer=start_layer, dtype=dtype)
    allocator = HiSparseTokenToKVPoolAllocator(
        pool.size, pool.page_size, pool.dtype, "cpu", pool, False
    )
    # Only host allocation and CUDA streams are unavailable in this CPU test.
    # Pool, allocator, coordinator construction, growth, merge and rollback
    # all use their real implementations.
    monkeypatch.setattr(
        module,
        "MLATokenToKVPoolHost",
        lambda **kw: SimpleNamespace(size=2048, token_stride_size=32),
    )
    monkeypatch.setattr(module, "device_module", MagicMock())
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda **kw: 1)
    c = HiSparseCoordinator(
        SimpleNamespace(
            req_to_token=torch.zeros((2, 1024), dtype=torch.int32),
            max_context_len=1024,
        ),
        allocator,
        top_k=top_k,
        device_buffer_size=128,
        device="cpu",
        tp_group=None,
        tail_protected_pages=2,
    )
    c._active_kvduo_reqs = {
        req: SimpleNamespace(hisparse_staging=False) for req in range(2)
    }
    c._mixed_slots = [True, True]
    return c, pool, allocator


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_flat_backing_aliases_full_layer_views_and_preserves_model_writes(
    monkeypatch, dtype
):
    pool = make_pool(dtype=dtype, start_layer=7)
    backing = pool.get_flat_key_buffer()
    stride = pool.flat_layer_slot_stride
    assert stride == pool.size + pool.page_size
    assert backing.shape == (3 * stride, 1, pool.kv_cache_dim)
    assert pool.get_kv_size_bytes() == backing.nbytes + sum(
        buf.nbytes for buf in pool.index_k_with_scale_buffer
    )
    for layer in range(3):
        assert pool.kv_buffer[layer].data_ptr() == (
            backing.data_ptr() + layer * stride * pool.bytes_per_token
        )
        assert pool.get_key_buffer(layer + 7).dtype == dtype
    mapping = torch.zeros(2049, dtype=torch.int64)
    mapping[10:12] = torch.tensor([64, 65])
    pool.register_mapping(mapping)
    values = torch.ones((2, 1, pool.kv_cache_dim), dtype=dtype)
    if pool.dsa_kv_cache_store_fp8:
        # The FP8 packer is a GPU kernel; retain the actual mapped write adapter.
        monkeypatch.setattr(
            pool,
            "_write_mla_kv_buffer",
            lambda cache, loc, nope, rope: cache.index_copy_(
                0, loc, nope.view(pool.store_dtype)
            ),
        )
        pool.set_mla_kv_buffer(
            SimpleNamespace(layer_id=8), torch.tensor([10, 11]), values, None
        )
    else:
        pool.set_kv_buffer(
            SimpleNamespace(layer_id=8), torch.tensor([10, 11]), values, None
        )
    assert torch.equal(
        backing[stride + 64 : stride + 66], values.view(pool.store_dtype)
    )
    assert not torch.any(pool.kv_buffer[0])
    assert not torch.any(pool.kv_buffer[2])
    # Indexer storage remains layer-local and keeps logical token capacity.
    assert len(pool.index_k_with_scale_buffer) == 3
    assert pool.index_buf_size == 2 * pool.size


def test_plain_hisparse_keeps_separate_layer_buffers():
    pool = make_pool(kvduo=False)
    assert not hasattr(pool, "_flat_kv_buffer")
    assert len({buf.untyped_storage().data_ptr() for buf in pool.kv_buffer}) == 3


@pytest.mark.parametrize("layers,top_k", [(3, 64), (3, 193), (78, 2048)])
def test_real_coordinator_allocates_lp_slots_per_whole_hot_page(
    monkeypatch, layers, top_k
):
    c, pool, allocator = make_stack(monkeypatch, layers=layers, top_k=top_k)
    k = layers * pool.page_size
    minimum = (top_k + k - 1) // k
    assert c.hot_storage_layers == layers
    assert c.hot_page_size == k
    assert c.device_buffer_size == 8 * k
    assert c.min_hot_pages == minimum
    expected_dtype = torch.int32 if 8 * k > 32767 else torch.int16
    assert c.lru_slots.dtype == expected_dtype
    before = allocator.hisparse_attn_allocator.available_size()
    c._ensure_kvduo_hot_capacity_targets_batch({(layers - 1, 1): minimum * k})
    pages = c._kvduo_req_layer_pages[layers - 1, 1]
    assert len(pages) == minimum
    assert allocator.hisparse_attn_allocator.available_size() == before - minimum * 64
    expected = [
        page + storage_layer * pool.flat_layer_slot_stride + offset
        for page in pages
        for storage_layer in range(layers)
        for offset in range(64)
    ]
    assert (
        c.req_device_buffer_token_locs[layers - 1, 1, : minimum * k].tolist()
        == expected
    )
    assert c.lru_slots[layers - 1, 1, : minimum * k].tolist() == list(
        range(minimum * k)
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_generic_merge_moves_payload_across_storage_layers_before_whole_page_free(
    monkeypatch, dtype
):
    c, pool, allocator = make_stack(monkeypatch, dtype=dtype)
    k = c.hot_page_size
    owner = (1, 0)
    c._ensure_kvduo_hot_capacity_targets_batch({owner: 3 * k})
    pages = list(c._kvduo_req_layer_pages[owner])
    locations = c.req_device_buffer_token_locs[owner][: 3 * k].long()
    c.req_device_buffer_tokens[owner][: 3 * k] = torch.arange(3 * k)
    pool.get_flat_key_buffer()[locations] = (
        torch.arange(3 * k).remainder(251).to(pool.store_dtype)[:, None, None]
    )
    survivors = [0, k + 64, 2 * k + 64, 2 * k + 65]
    c.req_device_buffer_tokens[owner][: 3 * k] = -1
    c.req_device_buffer_tokens[owner][survivors] = torch.tensor(
        survivors, dtype=torch.int32
    )
    c.kvduo_capacity_steps[owner] = 65
    c.kvduo_capacity_ema[owner] = 0
    payloads = pool.get_flat_key_buffer()[locations[survivors]].clone()
    free = allocator.free_hisparse_indices

    def free_after_copy(indices):
        assert c._kvduo_req_layer_pages[owner] == [pages[2]]
        locs = c.req_device_buffer_token_locs[owner]
        tokens = c.req_device_buffer_tokens[owner]
        for slot, token in enumerate(survivors):
            mapped = locs[torch.where(tokens == token)[0][0]]
            assert torch.equal(pool.get_flat_key_buffer()[mapped], payloads[slot])
        assert indices.tolist() == [
            index for page in pages[:2] for index in range(page, page + 64)
        ]
        free(indices)

    allocator.free_hisparse_indices = MagicMock(side_effect=free_after_copy)
    before = allocator.hisparse_attn_allocator.available_size()
    assert c._merge_kvduo_hot_pages(*owner) == 128
    assert allocator.hisparse_attn_allocator.available_size() == before + 128
    assert (
        c.req_device_buffer_tokens[owner][c.lru_slots[owner][:k].long()][-4:].tolist()
        == survivors
    )
    assert c._kvduo_hot_base_pages[owner] == (pages[2],)
    allocator.free_hisparse_indices.assert_called_once()


def test_generic_resolver_receives_flat_backing_and_local_full_layer_offset(
    monkeypatch,
):
    import sglang.srt.managers.hisparse_coordinator as module

    c, pool, _ = make_stack(monkeypatch, start_layer=7)
    c.mem_pool_host.kv_buffer = torch.zeros((3, 128, 1, 16), dtype=torch.bfloat16)
    swap = MagicMock()
    monkeypatch.setattr(module, "load_cache_to_device_buffer_mla", swap)
    c.swap_in_selected_pages(
        torch.tensor([1]),
        torch.tensor([64]),
        torch.zeros((1, 64), dtype=torch.int32),
        8,
    )
    kwargs = swap.call_args.kwargs
    assert kwargs["device_buffer"] is pool.get_flat_key_buffer()
    assert kwargs["full_location_base"] == pool.flat_layer_slot_stride
    assert kwargs["page_size"] == 3 * 64
    assert kwargs["host_cache"].data_ptr() == c.mem_pool_host.kv_buffer[1].data_ptr()


def test_coordinator_initializes_workspace_once_and_reuses_it_across_requests(
    monkeypatch,
):
    import sglang.srt.managers.hisparse_coordinator as module
    from sglang.kernels.ops.kvcache.hisparse import hisparse_resolver_scratch_words

    allocate = MagicMock(
        side_effect=lambda requests, topk, capacity, block, device: torch.empty(
            (requests, hisparse_resolver_scratch_words(topk, capacity)),
            dtype=torch.int32,
            device=device,
        )
    )
    monkeypatch.setattr(module, "allocate_hisparse_resolver_scratch", allocate)
    c, _, _ = make_stack(monkeypatch, layers=78, top_k=2048)
    c.mem_pool_host.kv_buffer = torch.zeros((78, 1, 1, 16), dtype=torch.bfloat16)
    swap = MagicMock()
    monkeypatch.setattr(module, "load_cache_to_device_buffer_mla", swap)
    workspace = c.resolver_scratch
    for req in (1, 0):
        c.swap_in_selected_pages(
            torch.tensor([req]),
            torch.tensor([2048]),
            torch.zeros((1, 2048), dtype=torch.int32),
            77,
        )
        assert swap.call_args.kwargs["resolver_scratch"] is workspace
    allocate.assert_called_once_with(2, 2048, 39936, 960, torch.device("cpu"))


def test_kvduo_bf16_flashmla_kv_rejected_during_actual_backend_initialization(
    monkeypatch,
):
    import sglang.srt.layers.attention.dsa_backend as module

    pool = make_pool()
    hf = SimpleNamespace(architectures=["DeepseekV32ForCausalLM"], index_topk=64)
    hf.get_text_config = lambda: hf
    runner = SimpleNamespace(
        device="cpu",
        page_size=64,
        token_to_kv_pool=pool,
        model_config=SimpleNamespace(
            hf_config=hf,
            context_len=128,
            num_attention_heads=1,
            qk_nope_head_dim=12,
            kv_lora_rank=12,
            qk_rope_head_dim=4,
        ),
        req_to_token_pool=SimpleNamespace(
            req_to_token=torch.zeros((2, 128), dtype=torch.int32)
        ),
        hisparse_coordinator=SimpleNamespace(enable_mixed_residency=True),
        server_args=SimpleNamespace(
            enable_deterministic_inference=False,
            dsa_prefill_backend="flashmla_sparse",
            dsa_decode_backend="flashmla_kv",
        ),
    )
    monkeypatch.setattr(module, "get_parallel", lambda: SimpleNamespace(attn_tp_size=1))
    with pytest.raises(ValueError, match="KVDuo flashmla_kv decode requires.*fp8_e4m3"):
        module.DeepseekSparseAttnBackend(runner)


def test_actual_flashmla_kv_rejects_bf16_before_quantization_or_kernel_import(
    monkeypatch,
):
    import sglang.srt.layers.attention.dsa_backend as module

    backend = module.DeepseekSparseAttnBackend.__new__(module.DeepseekSparseAttnBackend)
    backend.hisparse_coordinator = SimpleNamespace(enable_mixed_residency=True)
    backend.dsa_decode_impl = "flashmla_kv"
    backend.dsa_kv_cache_store_fp8 = False
    quantize = MagicMock()
    monkeypatch.setattr(module, "quantize_k_cache", quantize)
    with pytest.raises(ValueError, match="For BF16 KV use.*flashmla_sparse"):
        backend._forward_flashmla_kv(None, None, 12, 1, None, None, None)
    quantize.assert_not_called()


def test_actual_flashmla_kv_uses_fp8_backing_without_quantizing_it(monkeypatch):
    import sglang.srt.layers.attention.dsa_backend as module

    pool = make_pool(dtype=torch.float8_e4m3fn)
    backend = module.DeepseekSparseAttnBackend.__new__(module.DeepseekSparseAttnBackend)
    backend.hisparse_coordinator = SimpleNamespace(enable_mixed_residency=True)
    backend.dsa_decode_impl = "flashmla_kv"
    backend.dsa_kv_cache_store_fp8 = True
    backend.real_page_size = 64
    backend.kv_cache_dim = pool.kv_cache_dim
    backend.dsa_index_topk = 2
    backend.flashmla_kv_num_q_heads = 64
    output = torch.ones((1, 1, 64, 12))
    kernel = MagicMock(return_value=(output, None))
    monkeypatch.setitem(
        sys.modules,
        "sgl_kernel.flash_mla",
        SimpleNamespace(flash_mla_with_kvcache=kernel),
    )
    quantize = MagicMock(
        side_effect=AssertionError("FP8 storage must never be quantized")
    )
    monkeypatch.setattr(module, "quantize_k_cache", quantize)
    result = backend._forward_flashmla_kv(
        torch.zeros((1, 1, 16)),
        pool.get_flat_key_buffer().view(pool.dtype),
        12,
        1,
        SimpleNamespace(tp_q_head_num=1, head_dim=16),
        SimpleNamespace(
            dsa_cache_seqlens_int32=torch.tensor([2]),
            flashmla_metadata=SimpleNamespace(flashmla_metadata=None, num_splits=None),
        ),
        torch.tensor([[64, pool.flat_layer_slot_stride + 65]], dtype=torch.int32),
    )
    assert torch.equal(result, output[:, :, :1, :])
    assert (
        kernel.call_args.kwargs["k_cache"].data_ptr()
        == pool.get_flat_key_buffer().data_ptr()
    )
    quantize.assert_not_called()


def test_large_generic_rollback_preserves_int32_lru_and_base_contents(monkeypatch):
    c, pool, allocator = make_stack(monkeypatch, layers=78, top_k=2048)
    k = c.hot_page_size
    owner = (77, 1)
    c._ensure_kvduo_hot_capacity_targets_batch({owner: k})
    c._merge_kvduo_hot_pages(*owner)
    base = c._kvduo_hot_base_pages[owner]
    c._ensure_kvduo_hot_capacity_targets_batch({owner: 8 * k})
    c.lru_slots[owner] = torch.arange(8 * k, dtype=torch.int32).flip(0)
    c.req_device_buffer_tokens[owner] = torch.arange(c.padded_buffer_size)
    backing = pool.get_flat_key_buffer().clone()
    before = allocator.hisparse_attn_allocator.available_size()
    c._rollback_kvduo_hot_batch((owner,))
    assert c._kvduo_req_layer_pages[owner] == list(base)
    assert allocator.hisparse_attn_allocator.available_size() == before + 7 * 64
    assert torch.equal(pool.get_flat_key_buffer(), backing)
    assert c.lru_slots[owner][:k].tolist() == list(reversed(range(k)))
    assert torch.all(c.lru_slots[owner][k:] == -1)
    assert c.req_device_buffer_tokens[owner][:k].tolist() == list(range(k))
    assert torch.all(c.req_device_buffer_tokens[owner][k:] == -1)


@pytest.mark.parametrize("kvduo", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_dsa_decode_attention_uses_resolver_address_space(monkeypatch, kvduo, dtype):
    import sglang.srt.layers.attention.dsa_backend as module

    pool = make_pool(kvduo=kvduo, dtype=dtype)
    backend = module.DeepseekSparseAttnBackend.__new__(module.DeepseekSparseAttnBackend)
    backend.token_to_kv_pool = pool
    backend.dsa_decode_impl = (
        "flashmla_kv" if dtype == torch.float8_e4m3fn else "flashmla_sparse"
    )
    backend.forward_metadata = SimpleNamespace(
        dsa_cache_seqlens_int32=torch.tensor([2])
    )
    locations = torch.tensor(
        [[pool.flat_layer_slot_stride + 64, 2 * pool.flat_layer_slot_stride + 65]]
    )
    backend.hisparse_coordinator = SimpleNamespace(
        enable_mixed_residency=kvduo,
        swap_in_selected_pages=MagicMock(return_value=locations),
    )
    observed = MagicMock(return_value=torch.tensor([123]))
    backend._forward_flashmla_sparse = observed
    backend._forward_flashmla_kv = observed
    monkeypatch.setattr(
        module,
        "concat_mla_absorb_q_general",
        lambda nope, rope: torch.cat((nope, rope), -1),
    )
    layer = SimpleNamespace(
        is_cross_attention=False,
        layer_id=1,
        tp_q_head_num=1,
        head_dim=16,
        v_head_dim=12,
        scaling=1,
    )
    result = backend.forward_decode(
        torch.zeros((1, 1, 16)),
        None,
        None,
        layer,
        SimpleNamespace(
            req_pool_indices=torch.tensor([0]), seq_lens=torch.tensor([64])
        ),
        topk_indices=torch.tensor([[1, 2]], dtype=torch.int32),
    )
    assert result.item() == 123
    cache = observed.call_args.kwargs["kv_cache"]
    expected = pool.get_flat_key_buffer() if kvduo else pool.get_key_buffer(1)
    assert cache.data_ptr() == expected.data_ptr()
    assert cache.shape == expected.shape
    assert cache.dtype == pool.dtype
    assert torch.equal(observed.call_args.kwargs["page_table_1"], locations)


@pytest.mark.parametrize("kvduo", [False, True])
def test_dsa_pool_factory_enables_flat_storage_only_for_kvduo(monkeypatch, kvduo):
    import sglang.srt.layers.cp.utils as cp_utils
    import sglang.srt.mem_cache.kv_cache_configurator as module

    monkeypatch.setattr(
        cp_utils, "get_glm_dsa_cp_layer_shard_info", lambda _: (None, None)
    )
    monkeypatch.setattr(
        module,
        "get_memory",
        lambda: SimpleNamespace(enable_hisparse=True, enable_kvduo=kvduo),
    )
    monkeypatch.setattr(
        module,
        "get_exec",
        lambda: SimpleNamespace(features=SimpleNamespace(enable_memory_saver=False)),
    )
    monkeypatch.setattr(module, "get_schedule", lambda: SimpleNamespace(page_size=64))
    hf_config = SimpleNamespace(
        architectures=["DeepseekV32ForCausalLM"], index_head_dim=128, index_topk=64
    )
    hf_config.get_text_config = lambda: hf_config
    factory = module.KVCacheConfigurator.__new__(module.KVCacheConfigurator)
    factory.server_args = SimpleNamespace(
        hisparse_config='{"host_to_device_ratio": 2}',
        dsa_prefill_backend="flashmla_sparse",
        dsa_decode_backend="flashmla_sparse",
        dcp_size=1,
    )
    factory.is_draft_worker = False
    factory.kv_cache_dtype = torch.bfloat16
    factory.device = "cpu"
    factory.model_config = SimpleNamespace(
        kv_lora_rank=12, qk_rope_head_dim=4, hf_config=hf_config
    )
    factory.layer_info = SimpleNamespace(
        num_effective_layers=3, start_layer=0, end_layer=2
    )

    pool = factory._build_dsa_kv_pool(max_total_num_tokens=256)

    assert pool.enable_kvduo is kvduo
    assert hasattr(pool, "_flat_kv_buffer") is kvduo
