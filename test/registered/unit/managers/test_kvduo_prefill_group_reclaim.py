"""Queue-wide FULL reclaim planning without speculative request allocation."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.managers import scheduler as scheduler_module
from sglang.srt.managers.schedule_policy import AddReqResult
from sglang.test.ci.ci_register import register_cpu_ci
from test_kvduo_prefill_admission import adder_and_req, ready_coordinator

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def group_fixture(count=4, *, logical_size=100, dsv4=False, swa_size=100):
    c, physical = ready_coordinator()
    adder, first = adder_and_req(
        c, logical_size=logical_size, dsv4=dsv4, swa_size=swa_size
    )
    c.token_to_kv_pool_allocator = adder.token_to_kv_pool_allocator
    first.origin_input_ids = list(range(16))
    first.finished = lambda: False
    queue = [first] + [
        SimpleNamespace(
            origin_input_ids=list(range(16)),
            output_ids=[],
            sampling_params=SimpleNamespace(max_new_tokens=0),
            finished=lambda: False,
        )
        for _ in range(count - 1)
    ]
    c.mem_pool_host.size = 1000
    c.mem_pool_host.available_size = lambda: 1000
    c._evict_host_prefix_for_slots = MagicMock()
    c.preview_kvduo_prefill_reclaim(first, 4, 16, adder.tree_cache)
    c.record_kvduo_prefill_queue(adder, queue, count)
    return c, physical, adder, queue


def test_group_accumulates_waiting_requests_without_mutating_them():
    c, _, adder, queue = group_fixture()
    before = [dict(vars(r)) for r in queue]
    # 8 first-request pages + 3 * 9 pages (16 input + 2 alignment tokens).
    assert c._kvduo_extend_prefill_reclaim_plan(queue[0], 8) == 35
    assert [vars(r) for r in queue] == before
    assert not c._kvduo_prefill_history
    assert not c._kvduo_prefill_reservations
    assert not c._kvduo_prefill_host_reservations
    assert not adder.can_run_list


def test_group_stops_at_cumulative_history_budget():
    c, _, _, queue = group_fixture(count=8)
    # 80 usable tokens: first prompt consumes 16, four more fit, sixth does not.
    assert c._kvduo_extend_prefill_reclaim_plan(queue[0], 8) == 44


def test_group_counts_existing_unallocated_history_claims():
    c, _, _, queue = group_fixture(count=8)
    c._kvduo_prefill_history["other"] = SimpleNamespace(
        prefix_indices=[], full_untruncated_fill_ids=list(range(32))
    )
    assert c._kvduo_extend_prefill_reclaim_plan(queue[0], 8) == 26


def test_group_stops_at_host_budget_without_retaining_claims():
    c, _, _, queue = group_fixture()
    c._kvduo_prefill_host_reservations["first"] = (queue[0], 16, ())
    c.mem_pool_host.available_size = lambda: 32
    assert c._kvduo_extend_prefill_reclaim_plan(queue[0], 8) == 17
    assert set(c._kvduo_prefill_host_reservations) == {"first"}


def test_group_respects_request_slot_limit_and_physical_capacity():
    c, physical, adder, queue = group_fixture()
    c.record_kvduo_prefill_queue(adder, queue, 2)
    assert c._kvduo_extend_prefill_reclaim_plan(queue[0], 8) == 17
    c.record_kvduo_prefill_queue(adder, queue, 4)
    physical.size = 36
    assert c._kvduo_extend_prefill_reclaim_plan(queue[0], 8) == 17


def test_group_respects_cumulative_swa_budget():
    c, _, adder, queue = group_fixture(dsv4=True, logical_size=1024, swa_size=32)
    # First and each later chunk need 16 SWA tokens with page/window=8.
    assert c._kvduo_extend_prefill_reclaim_plan(queue[0], 2) == 2
    assert not adder.can_run_list


def test_queue_planning_has_constant_collective_count():
    c, _, _, queue = group_fixture(count=30, logical_size=4096)
    c.mem_pool_host.available_size = lambda: 100000
    c._kvduo_common_available = MagicMock(side_effect=lambda value: value)
    assert c._kvduo_extend_prefill_reclaim_plan(queue[0], 8) == 269
    assert c._kvduo_common_available.call_count == 3


def test_scheduler_records_remaining_queue_after_physical_block():
    # Execute the actual queue loop without constructing model workers.
    tree = ast.parse(Path(scheduler_module.__file__).read_text())
    loop = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and isinstance(node.iter, ast.Attribute)
        and ast.unparse(node.iter) == "self.waiting_queue"
        and any(
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "add_one_req"
            for n in ast.walk(node)
        )
    )
    fn = ast.FunctionDef(
        name="run",
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg=x) for x in ("self", "adder", "running_batch")],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[loop],
        decorator_list=[],
    )
    namespace = {
        "AddReqResult": AddReqResult,
        "DisaggregationMode": SimpleNamespace(PREFILL="prefill"),
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),
            "scheduler_queue",
            "exec",
        ),
        namespace,
    )
    c, _, adder, queue = group_fixture()
    c._kvduo_prefill_queue_plan = None
    req = queue[0]
    req.init_next_round_input = MagicMock()
    req.mamba_pool_idx = None
    adder.kvduo_prefill_blocked = True
    adder.add_one_req = MagicMock(return_value=AddReqResult.OTHER)
    scheduler = SimpleNamespace(
        waiting_queue=queue,
        chunked_req=None,
        enable_lora=False,
        disaggregation_mode=None,
        enable_priority_preemption=False,
        enable_hicache_storage=False,
        truncation_align_size=None,
        hisparse_coordinator=c,
        get_num_allocatable_reqs=lambda _: 4,
        tree_cache=adder.tree_cache,
    )
    namespace["run"](scheduler, adder, SimpleNamespace(reqs=[], batch_is_full=False))
    assert c._kvduo_prefill_queue_plan == (adder, queue, 4)
    adder.add_one_req.assert_called_once()


def test_group_plan_reclaims_more_than_two_full_requests():
    from test_kvduo_full_reclaim import make_request_reclaim_coordinator

    c, physical, _ = make_request_reclaim_coordinator([4, 4, 4, 4])
    adder, first = adder_and_req(c)
    free = c.token_to_kv_pool_allocator.free_hisparse_indices
    c.token_to_kv_pool_allocator = adder.token_to_kv_pool_allocator
    c.token_to_kv_pool_allocator.free_hisparse_indices = free
    first.full_untruncated_fill_ids = first.origin_input_ids = list(range(4))
    first.prefix_indices = []
    queue = [first] + [
        SimpleNamespace(
            origin_input_ids=list(range(4)),
            output_ids=[],
            sampling_params=SimpleNamespace(max_new_tokens=0),
        )
        for _ in range(2)
    ]
    c.preview_kvduo_prefill_reclaim(first, 4, 4, adder.tree_cache)
    c.record_kvduo_prefill_queue(adder, queue, 3)
    c._reserve_kvduo_prefill_host = MagicMock(return_value=True)
    c.mem_pool_host.available_size = lambda: 1000
    c._evict_host_prefix_for_slots = MagicMock()
    # One decode page + 2 first-request pages + 3 pages per extra request.
    assert c.reclaim_kvduo_full_pages(c.page_size) == 12 * c.page_size
    assert c._mixed_slots == [True, True, True, False]
    assert physical.free_pages - c._kvduo_mandatory_hot_pages() == 9
    assert c._kvduo_next_prefill_pages == 8
    c.reclaim_kvduo_hot_pages.assert_not_called()


def test_group_uses_common_tp_request_limit():
    c, _, _, queue = group_fixture()
    values = iter([2, 50, 1000])
    c._kvduo_common_available = MagicMock(side_effect=lambda _: next(values))
    assert c._kvduo_extend_prefill_reclaim_plan(queue[0], 8) == 17


def test_queue_context_is_cleared_after_allocation():
    c, _, _, _ = group_fixture()
    assert c._kvduo_prefill_queue_plan is not None
    c.kvduo_prefill_batch_allocated()
    assert c._kvduo_prefill_queue_plan is None
    assert c._kvduo_full_prefill_hint is None
