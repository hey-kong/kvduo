# mapping on device memory, host memory and memory allocator

import logging
from contextlib import nullcontext
from typing import Optional

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
from sglang.srt.utils import is_cuda, is_hip

logger = logging.getLogger(__name__)

# sgl_kernel.kvcacheio is only available in CUDA/ROCm sgl-kernel builds (not XPU/MPS/NPU/CPU).
_is_cuda = is_cuda()
_is_hip = is_hip()
if _is_cuda or _is_hip:
    from sgl_kernel.kvcacheio import transfer_kv_all_layer_mla
else:

    def transfer_kv_all_layer_mla(*args, **kwargs):
        raise RuntimeError(
            "HiSparse device KV transfer requires sgl_kernel.kvcacheio (CUDA/ROCm). "
            "It is not available on this backend."
        )


class HiSparseDSATokenToKVPool(DSATokenToKVPool):
    def __init__(
        self,
        size: int,
        page_size: int,
        kv_lora_rank: int,
        dtype: torch.dtype,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        index_head_dim: int,
        enable_memory_saver: bool,
        kv_cache_dim: int,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        host_to_device_ratio: int = 2,
        enable_kvduo: bool = False,
        logical_size: Optional[int] = None,
    ):
        if logical_size is not None and (
            logical_size < size or logical_size % page_size
        ):
            raise ValueError(
                "logical_size must be page aligned and cover physical capacity"
            )
        self.enable_kvduo = enable_kvduo
        super().__init__(
            size=size,
            page_size=page_size,
            kv_lora_rank=kv_lora_rank,
            dtype=dtype,
            qk_rope_head_dim=qk_rope_head_dim,
            layer_num=layer_num,
            device=device,
            index_head_dim=index_head_dim,
            enable_memory_saver=enable_memory_saver,
            kv_cache_dim=kv_cache_dim,
            start_layer=start_layer,
            end_layer=end_layer,
            index_buf_size=(
                size * host_to_device_ratio if logical_size is None else logical_size
            ),
        )
        self.bytes_per_token = self.kv_cache_dim * self.dtype.itemsize

    def _create_buffers(self):
        if not self.enable_kvduo:
            return super()._create_buffers()
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.custom_mem_pool
                else nullcontext()
            ):
                # FULL retains ordinary layer views. HOT owns a whole page
                # number and can use its fragments across every storage layer.
                self._flat_kv_buffer = torch.zeros(
                    (
                        self.layer_num * (self.size + self.page_size),
                        1,
                        self.kv_cache_dim,
                    ),
                    dtype=self.store_dtype,
                    device=self.device,
                )
                self.kv_buffer = list(
                    self._flat_kv_buffer.view(
                        self.layer_num, self.size + self.page_size, 1, self.kv_cache_dim
                    )
                )

    def _clear_buffers(self):
        super()._clear_buffers()
        if self.enable_kvduo:
            del self._flat_kv_buffer

    def get_flat_key_buffer(self) -> torch.Tensor:
        """Raw cross-layer backing, with capture-stable token addresses."""
        return self._flat_kv_buffer

    @property
    def flat_layer_slot_stride(self) -> int:
        return self.size + self.page_size

    def register_mapping(self, full_to_hisparse_device_index_mapping: torch.Tensor):
        self.full_to_hisparse_device_index_mapping = (
            full_to_hisparse_device_index_mapping
        )

    def translate_loc_to_hisparse_device(self, compressed_indices: torch.Tensor):
        return self.full_to_hisparse_device_index_mapping[compressed_indices]

    def _translate_loc_to_hisparse_device(self, compressed_indices: torch.Tensor):
        return self.full_to_hisparse_device_index_mapping[compressed_indices]

    def translate_loc_from_full_to_hisparse_device(self, full_indices: torch.Tensor):
        return self._translate_loc_to_hisparse_device(full_indices)

    def translate_loc_from_full_to_compressed(self, full_indices: torch.Tensor):
        return full_indices

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        loc = self.translate_loc_to_hisparse_device(loc)
        super().set_kv_buffer(layer, loc, cache_k, cache_v)

    def set_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ):
        loc = self.translate_loc_to_hisparse_device(loc)
        super().set_mla_kv_buffer(layer, loc, cache_k_nope, cache_k_rope)

    def get_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        dst_dtype: Optional[torch.dtype] = None,
    ):
        loc = self.translate_loc_to_hisparse_device(loc)
        return super().get_mla_kv_buffer(layer, loc, dst_dtype)

    def transfer_values_on_device(self, dst_indices, src_indices):
        transfer_kv_all_layer_mla(
            src_layers=self.data_ptrs,
            dst_layers=self.data_ptrs,
            src_indices=src_indices,
            dst_indices=dst_indices,
            item_size=self.bytes_per_token,
            num_layers=self.layer_num,
        )

    def get_cpu_copy(self, indices, mamba_indices=None):
        raise NotImplementedError("HiSparseDevicePool does not support get_cpu_copy")

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        raise NotImplementedError("HiSparseDevicePool does not support load_cpu_copy")
