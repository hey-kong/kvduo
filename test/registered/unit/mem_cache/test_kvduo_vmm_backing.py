import pytest
from sglang.srt.mem_cache.kvduo_vmm_backing import VMMBacking


class Backend:
    def __init__(self):
        self.handles = {}
        self.maps = {}
        self.next_handle = 0
        self.fail_map = None
        self.fail_unmap = None
        self.fail_create = False

    def create(self, size):
        if self.fail_create:
            raise MemoryError("driver OOM")
        self.next_handle += 1
        self.handles[self.next_handle] = bytearray(size)
        return self.next_handle

    def map(self, address, size, handle):
        if address == self.fail_map:
            raise RuntimeError("mapping failure")
        assert address not in self.maps
        assert len(self.handles[handle]) == size
        self.maps[address] = handle

    def unmap(self, address, size):
        if address == self.fail_unmap:
            raise RuntimeError("unmap failure")
        del self.maps[address]

    def release(self, handle):
        assert handle not in self.maps.values()
        del self.handles[handle]


def populate(pool, addresses):
    pool.prepare(pool.plan(destinations=addresses), accesses_complete=True).commit()


def setup_pool(blocks=3):
    backend = Backend()
    return backend, VMMBacking(backend, block_bytes=64, budget_bytes=blocks * 64)


def test_exchange_and_return_without_extra_physical_memory():
    backend, pool = setup_pool(2)
    populate(pool, (64, 128))
    donor = pool.mappings[64]
    backend.handles[donor][:] = b"x" * 64
    transaction = pool.prepare(
        pool.plan(donors=(64,), destinations=(256,)), accesses_complete=True
    )
    assert pool.mappings == {64: donor, 128: backend.maps[128]}
    assert backend.maps[64] == backend.maps[256]
    assert pool.allocated_bytes == 128
    transaction.commit()
    assert 64 not in backend.maps
    assert backend.handles[pool.mappings[256]] == b"x" * 64
    pool.prepare(
        pool.plan(donors=(256,), destinations=(64,)), accesses_complete=True
    ).commit()
    assert pool.mappings[64] == donor
    pool.close(accesses_complete=True)
    assert not backend.maps and not backend.handles


def test_tp_prepare_abort_preserves_donor_and_epoch():
    backend, pool = setup_pool(2)
    populate(pool, (64,))
    plan = pool.plan(donors=(64,), destinations=(128, 192))
    before = dict(pool.mappings)
    transaction = pool.prepare(plan, accesses_complete=True)
    transaction.abort()
    assert pool.mappings == before == backend.maps
    pool.prepare(plan, accesses_complete=True).commit()
    assert set(pool.mappings) == {128, 192}
    assert backend.next_handle == 2


@pytest.mark.parametrize("failure", [128, 192])
def test_map_failure_rolls_back_aliases(failure):
    backend, pool = setup_pool(2)
    populate(pool, (64,))
    before = dict(pool.mappings)
    backend.fail_map = failure
    with pytest.raises(RuntimeError, match="mapping failure"):
        pool.prepare(
            pool.plan(donors=(64,), destinations=(128, 192)), accesses_complete=True
        )
    assert pool.mappings == backend.maps == before
    backend.fail_map = None
    pool.prepare(
        pool.plan(donors=(64,), destinations=(128, 192)), accesses_complete=True
    ).commit()
    pool.close(accesses_complete=True)
    assert not backend.handles


def test_driver_oom_does_not_remove_donor():
    backend, pool = setup_pool(2)
    populate(pool, (64,))
    backend.fail_create = True
    with pytest.raises(MemoryError):
        pool.prepare(
            pool.plan(donors=(64,), destinations=(128, 192)), accesses_complete=True
        )
    assert pool.mappings == backend.maps
    assert set(backend.maps) == {64}


def test_complete_plan_must_fit_before_mapping_anything():
    backend, pool = setup_pool(2)
    populate(pool, (64, 128))
    with pytest.raises(MemoryError):
        pool.plan(donors=(64,), destinations=(192, 256))
    assert set(backend.maps) == {64, 128}


@pytest.mark.parametrize(
    "donors,destinations",
    [
        ((64, 64), (128,)),
        ((128,), (192,)),
        ((64,), (64,)),
        ((), (1,)),
        ((), (0,)),
        ((), (128, 128)),
    ],
)
def test_invalid_plan(donors, destinations):
    _, pool = setup_pool()
    populate(pool, (64,))
    with pytest.raises(ValueError):
        pool.plan(donors=donors, destinations=destinations)


def test_safety_boundary_and_stale_plan():
    _, pool = setup_pool()
    populate(pool, (64,))
    plan = pool.plan(donors=(64,), destinations=(128,))
    with pytest.raises(RuntimeError, match="not completed"):
        pool.prepare(plan, accesses_complete=False)
    populate(pool, (192,))
    with pytest.raises(RuntimeError, match="stale"):
        pool.prepare(plan, accesses_complete=True)
    with pytest.raises(RuntimeError, match="outstanding"):
        pool.close(accesses_complete=False)


def test_partial_commit_failure_prevents_further_use_and_can_clean_up():
    backend, pool = setup_pool(2)
    populate(pool, (64, 128))
    transaction = pool.prepare(
        pool.plan(donors=(64, 128), destinations=(192, 256)), accesses_complete=True
    )
    backend.fail_unmap = 128
    with pytest.raises(RuntimeError, match="unmap failure"):
        transaction.commit()
    with pytest.raises(RuntimeError, match="incomplete unmap"):
        pool.plan(destinations=(320,))
    backend.fail_unmap = None
    pool.close(accesses_complete=True)
    assert not backend.maps and not backend.handles


def test_free_backing_reused_and_simultaneous_transactions_rejected():
    backend, pool = setup_pool(2)
    populate(pool, (64, 128))
    transaction = pool.prepare(pool.plan(donors=(64,)), accesses_complete=True)
    with pytest.raises(RuntimeError, match="pending"):
        pool.plan(destinations=(192,))
    transaction.commit()
    populate(pool, (192,))
    assert backend.next_handle == 2
    pool.close(accesses_complete=True)
    pool.close(accesses_complete=True)


def test_abort_failure_is_not_reusable_until_teardown():
    backend, pool = setup_pool(1)
    populate(pool, (64,))
    transaction = pool.prepare(
        pool.plan(donors=(64,), destinations=(128,)), accesses_complete=True
    )
    backend.fail_unmap = 128
    with pytest.raises(RuntimeError):
        transaction.abort()
    with pytest.raises(RuntimeError):
        pool.plan()
    backend.fail_unmap = None
    pool.close(accesses_complete=True)
    assert not backend.maps and not backend.handles
