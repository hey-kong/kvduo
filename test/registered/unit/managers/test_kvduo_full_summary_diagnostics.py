from unittest.mock import MagicMock

import pytest
import torch
import tvm_ffi

from sglang.kernels.ops.kvcache.kvduo_full_reclaim import _call_full_summary
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def arguments():
    x = torch.arange(128, dtype=torch.int64)
    return (
        x.view(2, 64),
        x + 64,
        x,
        x,
        x,
        None,
        torch.empty((2, 64), dtype=torch.int64),
        torch.empty(2, dtype=torch.int64),
        torch.empty(2, dtype=torch.int32),
    )


def test_successful_ffi_call_with_optional_none_has_no_export_probe(monkeypatch):
    export = MagicMock(side_effect=AssertionError("diagnostic ran on success"))
    monkeypatch.setattr(torch.utils.dlpack, "to_dlpack", export)
    call = MagicMock()
    args = arguments()
    _call_full_summary(call, *args)
    call.assert_called_once_with(*args)
    export.assert_not_called()


def test_real_ffi_accepts_optional_none_and_inference_tensors():
    native = tvm_ffi.convert(lambda *args: None)
    with torch.inference_mode():
        _call_full_summary(native, *arguments())


@pytest.mark.parametrize(
    "position,name",
    [(0, "logical_pages"), (1, "mapping"), (6, "physical_pages"), (8, "valid_pages")],
)
def test_conversion_failure_names_the_bad_tensor_and_does_not_retry_kernel(
    monkeypatch, position, name
):
    args = arguments()
    original = torch.utils.dlpack.to_dlpack

    def export(tensor):
        assert tensor is not None
        if tensor is args[position]:
            raise RuntimeError("tensor does not have a device")
        return original(tensor)

    monkeypatch.setattr(torch.utils.dlpack, "to_dlpack", export)
    cause = RuntimeError("tensor does not have a device")
    call = MagicMock(side_effect=cause)
    with pytest.raises(RuntimeError, match=f"Failing arguments: {name}") as caught:
        _call_full_summary(call, *args)
    call.assert_called_once_with(*args)
    assert caught.value.__cause__ is cause
    assert "protected_pages: None (optional)" in str(caught.value)
    assert "device=cpu" in str(caught.value)


def test_passing_exports_do_not_falsely_attribute_the_failure_to_a_tensor():
    cause = RuntimeError("tensor does not have a device")
    with pytest.raises(RuntimeError, match="All standalone exports passed") as caught:
        _call_full_summary(MagicMock(side_effect=cause), *arguments())
    assert caught.value.__cause__ is cause


def test_other_errors_keep_original_exception_without_exporting_any_tensor(monkeypatch):
    export = MagicMock(
        side_effect=AssertionError("unsafe diagnostic after kernel failure")
    )
    monkeypatch.setattr(torch.utils.dlpack, "to_dlpack", export)
    cause = RuntimeError("CUDA error: an illegal memory access was encountered")
    with pytest.raises(RuntimeError) as caught:
        _call_full_summary(MagicMock(side_effect=cause), *arguments())
    assert caught.value is cause
    export.assert_not_called()
