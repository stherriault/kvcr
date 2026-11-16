# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""G4 write side: spill mirroring G3, atomic publish, and the circuit breaker."""

import contextlib
import errno
import os
import re
import threading
import time

import pytest
from _g4_test_utils import (
    PAGE,
    digest,
    fetchable_g4,
    fetching_g4,
    find_files,
    new_world,
    pattern,
    tmp_files,
    wait_for,
)
from _kvcr_test_utils import _mem_descriptor

from kvcr.types import (
    BlockKey,
    CacheTier,
    PlacementAction,
    QueryStatus,
)

FLAVORS = ["object", "file"]
A = BlockKey(b"block-a")
B = BlockKey(b"block-b")
C = BlockKey(b"block-c")
D = BlockKey(b"block-d")
MISS = (QueryStatus.MISS, None)


def _present(w, key, rid="r"):
    assert w.await_check(key, rid) == fetchable_g4()


def _capacity_key(w):
    return w.local_dram._pools._capacity_eviction_key


def _is_open(w):
    """An open breaker answers MISS at once and starts no check."""
    return w.breaker_open()


# ---- Spill mirrors G3 ----


@pytest.mark.parametrize("flavor", FLAVORS)
def test_g4_spill_holds_g2_pending_until_the_write_ends(g4env, tmp_path, flavor):
    w = new_world(g4env, tmp_path, flavor=flavor)
    w.storage.outcome["WRITE"] = "HOLD"
    assert w.deposit(A, 0).success
    waiting = w.deposit(B, 1, wait=False)
    w.poll_until(lambda: len(w.storage.held("WRITE")) == 1)
    w.settle()
    assert waiting not in w.results
    assert _capacity_key(w) == A
    w.storage.finish_all("DONE")
    assert w.wait(waiting)[B].success
    assert w.stored(A)
    assert _capacity_key(w) is None


def test_g4_one_capacity_spill_in_flight_serializes_waiting_deposits(g4env, tmp_path):
    w = new_world(g4env, tmp_path, flavor="object", slots=2)
    w.storage.outcome["WRITE"] = "HOLD"
    for index, key in enumerate((A, B)):
        assert w.deposit(key, index).success
    extra = (C, D, BlockKey(b"block-e"))
    deposits = [w.deposit(key, 2 + i, wait=False) for i, key in enumerate(extra)]
    for round_ in range(3):
        w.poll_until(
            lambda: (
                len(w.storage.log("WRITE")) == round_ + 1
                and len(w.storage.held("WRITE")) == 1
            )
        )
        w.settle()
        assert all(handle not in w.results for handle in deposits[round_:])
        w.storage.finish_all("DONE")
        assert all(r.success for r in w.wait(deposits[round_]).values())
    assert w.stored(A) and w.stored(B)


def test_g4_waiting_deposit_fails_at_its_deadline_and_g2_stays_blocked(g4env, tmp_path):
    w = new_world(g4env, tmp_path, flavor="object", spill_timeout_ms=10_000)
    w.storage.outcome["WRITE"] = "HOLD"
    assert w.deposit(A, 0).success
    waiting = w.deposit(B, 1, wait=False)
    w.poll_until(lambda: len(w.storage.held("WRITE")) == 1)
    w.advance(2.0)
    assert not w.wait(waiting)[B].success
    assert _capacity_key(w) == A
    w.storage.finish_all("DONE")
    w.poll_until(lambda: _capacity_key(w) is None)


@pytest.mark.parametrize("flavor", FLAVORS)
def test_g4_failed_spill_retires_the_victim_and_the_deposit_lands(
    g4env, tmp_path, flavor
):
    w = new_world(g4env, tmp_path, flavor=flavor)
    w.storage.outcome["WRITE"] = "ERR"
    assert w.deposit(A, 0).success
    assert w.deposit(B, 1).success
    assert not w.stored(A)
    if flavor == "file":
        assert tmp_files(w.root) == []
    assert w.kvcr.query([B]) == [(QueryStatus.HIT, CacheTier.LOCAL_G2)]
    record = w.kvcr._core._block_record_map.get(A)
    assert record is None or record.local_dram is None
    [(meta, failure)] = w.policy.failures
    assert meta.block_key == A
    assert failure.attempted == (PlacementAction.MOVE_TO, CacheTier.G4)
    assert failure.source is CacheTier.LOCAL_G2


@pytest.mark.parametrize("flavor", FLAVORS)
@pytest.mark.parametrize("failure", ["RAISE_POST", "RAISE_POLL"])
def test_g4_spill_that_raises_retires_the_victim_and_the_deposit_lands(
    g4env, tmp_path, flavor, failure
):
    w = new_world(g4env, tmp_path, flavor=flavor)
    w.storage.outcome["WRITE"] = failure
    assert w.deposit(A, 0).success
    assert w.deposit(B, 1).success
    w.poll_until(lambda: not w.storage.unreleased())
    assert w.storage.violations == []
    assert not w.stored(A)
    if flavor == "file":
        assert tmp_files(w.root) == []
    assert w.kvcr.query([B]) == [(QueryStatus.HIT, CacheTier.LOCAL_G2)]
    [(meta, failure_info)] = w.policy.failures
    assert meta.block_key == A
    assert failure_info.reason == "transfer failed"


@pytest.mark.parametrize("flavor", FLAVORS)
def test_g4_spill_start_failure_retires_the_source_and_the_deposit_lands(
    g4env, tmp_path, flavor
):
    w = new_world(g4env, tmp_path, flavor=flavor, slots=2)
    assert w.deposit(A, 0).success and w.deposit(B, 1).success
    w.storage.init_error = lambda op: (
        RuntimeError("start failed") if op == "WRITE" else None
    )
    assert w.deposit(C, 2).success
    assert not w.stored(A)
    assert w.kvcr.query([B, C]) == [(QueryStatus.HIT, CacheTier.LOCAL_G2)] * 2
    assert len(w.policy.failures) == 1
    w.poll_until(lambda: not w.storage.live_regs)
    if flavor == "file":
        assert tmp_files(w.root) == []
    assert w.storage.unreleased() == []


def test_g4_resolve_eviction_keeps_a_pending_key_and_never_spills_it_twice(
    g4env, tmp_path
):
    w = new_world(g4env, tmp_path, flavor="object")
    w.storage.outcome["WRITE"] = "HOLD"
    assert w.deposit(A, 0).success
    core = w.kvcr._core
    record = core._block_record_map[A]
    size = core._local_dram._pools._size_bytes(record.local_dram)
    meta = core._block_meta(A, record, size)
    decision = (PlacementAction.MOVE_TO, CacheTier.G4)
    keep = (PlacementAction.KEEP, None)
    resolve = core._g4.resolve_eviction
    assert resolve(meta, CacheTier.LOCAL_G2, decision, 1.0) == (keep, True)
    assert resolve(meta, CacheTier.LOCAL_G2, decision, 1.0) == (keep, False)
    wait_for(lambda: len(w.storage.held("WRITE")) == 1)
    assert len(w.storage.log("WRITE")) == 1


# ---- Policy wiring ----


def test_g4_default_policy_spills_when_only_g4_is_configured(g4env, tmp_path):
    w = new_world(g4env, tmp_path, flavor="object", policy=None)
    assert w.deposit(A, 0).success and w.deposit(B, 1).success
    assert w.stored(A)


def test_g4_lru_policy_moves_local_g2_victims_to_g4():
    from kvcr.policy import G4LRUPolicy
    from kvcr.types import BlockMeta

    meta = BlockMeta(A, PAGE, 1, 0.0, frozenset({CacheTier.LOCAL_G2}))
    policy = G4LRUPolicy()
    assert policy.decide_eviction(meta, CacheTier.LOCAL_G2) == (
        PlacementAction.MOVE_TO,
        CacheTier.G4,
    )
    assert CacheTier.G4 in policy.required_tiers


def test_g4_default_policy_with_g3_stays_g3_lru_and_logs_the_missing_spill(
    g4env, tmp_path, caplog
):
    from kvcr.config import G3Options
    from kvcr.policy import G3LRUPolicy

    caplog.set_level("INFO")
    g3 = G3Options(
        paths=(tmp_path / "g3.data",), capacity_bytes_per_file=2 * PAGE, backend="MOCK"
    )
    w = new_world(g4env, tmp_path, flavor="object", policy=None, g3=g3)
    assert type(w.kvcr._core._policy._policy) is G3LRUPolicy
    assert any("G4" in record.getMessage() for record in caplog.records)


# ---- Atomic publish ----


def test_g4_file_spill_is_invisible_to_readers_until_rename(g4env, tmp_path):
    w = new_world(g4env, tmp_path, flavor="file")
    w.storage.outcome["WRITE"] = "HOLD"
    assert w.deposit(A, 0).success
    waiting = w.deposit(B, 1, wait=False)
    w.poll_until(lambda: len(w.storage.held("WRITE")) == 1)
    (tmp,) = tmp_files(w.root)
    assert re.fullmatch(rf"[0-9a-f]{{64}}\.tmp\..+\.{os.getpid()}\.[0-9a-f]+", tmp.name)
    assert find_files(w.root, A) == []
    reader = new_world(
        g4env, tmp_path, flavor="file", root=w.root, store=w.store, name="reader"
    )
    assert reader.await_check(A) == MISS
    w.storage.finish_all("DONE")
    assert w.wait(waiting)[B].success
    assert tmp_files(w.root) == []
    (final,) = find_files(w.root, A)
    assert final.read_bytes() == pattern(0, PAGE)


def test_g4_object_spill_is_one_whole_object_write(g4env, tmp_path):
    w = new_world(g4env, tmp_path, flavor="object", pages=2)
    w.spill(A, 0)
    (write,) = w.storage.log("WRITE")
    assert len(write.remote) == 1 and write.remote[0][1] == 2 * PAGE
    assert len(w.store.objects[w.store.find(A)]) == 2 * PAGE


def test_g4_overwriting_spill_leaves_a_complete_block(g4env, tmp_path):
    w = new_world(g4env, tmp_path, flavor="file")
    w.spill(A, 0)
    w.fill(0, b"\x42" * PAGE)
    assert w.wait(w.kvcr.deposit({A: [w.ref(0)]}))[A].success
    assert w.deposit(B, 1).success
    w.poll_until(lambda: not tmp_files(w.root) and not w.storage.unreleased())
    (final,) = find_files(w.root, A)
    assert final.read_bytes() == b"\x42" * PAGE
    assert tmp_files(w.root) == []


# ---- Circuit breaker ----


def _fail_reads(w, key, count, rid="r"):
    for _ in range(count):
        assert w.await_check(key, rid) == fetchable_g4()
        assert not w.wait(w.deliver(key, 4, rid))[key].success


def test_g4_head_ok_get_fails_opens_the_breaker_at_the_threshold(g4env, tmp_path):
    w = new_world(g4env, tmp_path, flavor="object", failure_threshold=3)
    w.spill(A, 0)
    w.storage.outcome["READ"] = "ERR"
    _fail_reads(w, A, 3)
    assert _is_open(w)


@pytest.mark.parametrize("flavor", FLAVORS)
@pytest.mark.parametrize("failure", ["RAISE_POST", "RAISE_POLL"])
def test_g4_read_that_raises_counts_toward_the_breaker(
    g4env, tmp_path, flavor, failure
):
    w = new_world(g4env, tmp_path, flavor=flavor, failure_threshold=1)
    w.spill(A, 0)
    w.storage.outcome["READ"] = failure
    _fail_reads(w, A, 1)
    w.poll_until(lambda: _is_open(w))


def test_g4_a_successful_read_resets_the_failure_count(g4env, tmp_path):
    w = new_world(g4env, tmp_path, flavor="object", failure_threshold=3)
    w.spill(A, 0)
    w.storage.sequence["READ"] = ["ERR", "ERR", "DONE", "ERR", "ERR"]
    for outcome_ok in (False, False, True, False, False):
        assert w.await_check(A) == fetchable_g4()
        assert w.wait(w.deliver(A, 4))[A].success is outcome_ok
    assert w.kvcr.query([BlockKey(b"fresh")], "r")[0] == fetching_g4()


def test_g4_answered_checks_never_count_toward_the_breaker(g4env, tmp_path):
    w = new_world(g4env, tmp_path, flavor="object", failure_threshold=2)
    for index in range(5):
        assert w.await_check(BlockKey(b"absent-%d" % index)) == MISS
    assert w.kvcr.query([BlockKey(b"next")], "r")[0] == fetching_g4()


def test_g4_open_breaker_drops_a_spill_without_io_or_stall(g4env, tmp_path):
    w = new_world(g4env, tmp_path, flavor="object", failure_threshold=2)
    w.storage.outcome["WRITE"] = "ERR"
    for index, key in enumerate((A, B, C)):
        assert w.deposit(key, index).success
    writes = len(w.storage.log("WRITE"))
    assert writes == 2
    assert w.deposit(D, 3).success
    assert len(w.storage.log("WRITE")) == writes
    assert not w.stored(C)


def test_g4_deadline_counts_once_per_operation(g4env, tmp_path):
    w = new_world(g4env, tmp_path, flavor="object", failure_threshold=2)
    w.spill(A, 0)
    assert w.await_check(A) == fetchable_g4()
    w.storage.outcome["READ"] = "HOLD"
    handle = w.deliver(A, 4)
    wait_for(lambda: len(w.storage.held("READ")) == 1)
    w.advance(2.0)
    w.settle()
    w.storage.finish_all("ERR")
    assert not w.wait(handle)[A].success
    calls = w.check_calls()
    assert w.await_check(BlockKey(b"fresh"), "r") == MISS
    assert w.check_calls() == calls + 1
    w.storage.outcome["READ"] = "ERR"
    w.advance(10.0)
    _fail_reads(w, A, 1)
    assert _is_open(w)


@pytest.mark.parametrize("trial", ["DONE", "ERR"])
def test_g4_half_open_trial_decides_the_next_state(g4env, tmp_path, trial):
    w = new_world(
        g4env, tmp_path, flavor="object", failure_threshold=1, backoff_ms=10_000
    )
    w.spill(A, 0)
    w.storage.outcome["READ"] = "ERR"
    _fail_reads(w, A, 1)
    assert _is_open(w)
    w.advance(10.5)
    assert w.await_check(A) == fetchable_g4()
    w.storage.outcome["READ"] = "HOLD"
    trial_op = w.deliver(A, 4)
    wait_for(lambda: len(w.storage.held("READ")) == 1)
    reads = len(w.storage.log("READ"))
    assert w.await_check(A) == fetchable_g4()
    assert not w.wait(w.deliver(A, 5))[A].success
    assert len(w.storage.log("READ")) == reads
    w.storage.finish_all(trial)
    assert w.wait(trial_op)[A].success is (trial == "DONE")
    if trial == "DONE":
        w.storage.outcome["READ"] = "DONE"
        assert w.await_check(A) == fetchable_g4()
        assert w.wait(w.deliver(A, 4))[A].success
    else:
        assert _is_open(w)
        w.advance(15.0)
        assert _is_open(w)
        w.advance(21.0)
        assert w.kvcr.query([BlockKey(b"later")], "r")[0] == fetching_g4()


def test_g4_spill_failure_counts_toward_the_breaker(g4env, tmp_path):
    w = new_world(g4env, tmp_path, flavor="object", failure_threshold=1)
    w.storage.outcome["WRITE"] = "ERR"
    assert w.deposit(A, 0).success and w.deposit(B, 1).success
    assert _is_open(w)


# ---- Spill deadline rules ----


@pytest.mark.parametrize("flavor", FLAVORS)
def test_g4_one_store_job_lands_floor_t_over_w_deposits(g4env, tmp_path, flavor):
    w = new_world(g4env, tmp_path, flavor=flavor, slots=4)
    victims = [BlockKey(b"v%d" % i) for i in range(4)]
    for index, victim in enumerate(victims):
        assert w.deposit(victim, index).success
    incoming = [BlockKey(b"d%d" % i) for i in range(4)]
    for index in range(4):
        w.fill(4 + index)
    w.storage.outcome["WRITE"] = "HOLD"
    job = w.kvcr.deposit({k: [_mem_descriptor(4 + i)] for i, k in enumerate(incoming)})

    def spill_running(count):
        return (
            len(w.storage.log("WRITE")) == count and len(w.storage.held("WRITE")) == 1
        )

    # T = 1.0 and w = 0.4: two spills finish in time, the third is still running.
    w.poll_until(lambda: spill_running(1))
    w.advance(0.4)
    w.storage.finish_all("DONE")
    w.poll_until(lambda: spill_running(2))
    w.advance(0.8)
    w.storage.finish_all("DONE")
    w.poll_until(lambda: spill_running(3))
    w.advance(1.0)
    result = w.wait(job)
    assert [result[key].success for key in incoming] == [True, True, False, False]
    # V3 keeps running past the job deadline and publishes.
    w.advance(1.2)
    w.storage.finish_all("DONE")
    w.poll_until(lambda: w.stored(victims[2]) and _capacity_key(w) is None)
    assert w.stored(victims[0]) and w.stored(victims[1])
    assert not w.stored(victims[3])


@pytest.mark.parametrize("flavor", FLAVORS)
def test_g4_slow_spill_is_published_and_leaves_the_breaker_closed(
    g4env, tmp_path, flavor
):
    w = new_world(
        g4env, tmp_path, flavor=flavor, failure_threshold=1, spill_timeout_ms=5000
    )
    w.storage.outcome["WRITE"] = "HOLD"
    assert w.deposit(A, 0).success
    waiting = w.deposit(B, 1, wait=False)
    w.poll_until(lambda: len(w.storage.held("WRITE")) == 1)
    w.advance(2.0)
    w.wait(waiting)
    w.storage.finish_all("DONE")
    w.poll_until(lambda: w.stored(A) and _capacity_key(w) is None)
    if flavor == "file":
        w.poll_until(lambda: tmp_files(w.root) == [])
    assert not _is_open(w)


@pytest.mark.parametrize("direct_io", [False, True])
def test_g4_file_spill_holds_g2_pending_in_both_direct_io_modes(
    g4env, tmp_path, direct_io
):
    w = new_world(g4env, tmp_path, flavor="file", direct_io=direct_io)
    w.storage.outcome["WRITE"] = "HOLD"
    assert w.deposit(A, 0).success
    waiting = w.deposit(B, 1, wait=False)
    w.poll_until(lambda: len(w.storage.held("WRITE")) == 1)
    assert _capacity_key(w) == A
    w.storage.finish_all("DONE")
    assert w.wait(waiting)[B].success
    assert w.stored(A)


# ---- Publish steps ----


@pytest.mark.parametrize("call", ["fsync", "rename"])
def test_g4_publish_error_after_done_is_a_spill_failure(
    g4env, tmp_path, monkeypatch, call
):
    w = new_world(g4env, tmp_path, flavor="file", failure_threshold=1)
    me = threading.get_ident()
    names = ["fsync"] if call == "fsync" else ["rename", "replace"]
    for name in names:
        real = getattr(os, name)

        def failing(*args, _real=real, **kwargs):
            if threading.get_ident() != me:
                raise OSError(errno.EIO, "publish failed")
            return _real(*args, **kwargs)

        monkeypatch.setattr(os, name, failing)
    assert w.deposit(A, 0).success
    assert w.deposit(B, 1).success
    assert not w.stored(A)
    w.poll_until(lambda: tmp_files(w.root) == [])
    assert len(w.policy.failures) == 1
    assert _is_open(w)


def test_g4_mkdir_race_on_a_new_fanout_directory_is_not_an_error(
    g4env, tmp_path, monkeypatch
):
    w = new_world(g4env, tmp_path, flavor="file")
    me = threading.get_ident()
    real = os.mkdir
    raced = set()

    def racing(path, *args, **kwargs):
        text = os.fspath(path)
        if threading.get_ident() != me and text.startswith(str(w.root)):
            if text not in raced:
                raced.add(text)
                with contextlib.suppress(FileExistsError):
                    real(path, *args, **kwargs)
                raise FileExistsError(errno.EEXIST, "lost the race", text)
        return real(path, *args, **kwargs)

    monkeypatch.setattr(os, "mkdir", racing)
    w.spill(A, 0)
    assert raced
    assert w.stored(A)


# ---- Breaker: absent never counts, one outcome per op ----


@pytest.mark.parametrize("flavor", FLAVORS)
@pytest.mark.parametrize("shape", ["one_op", "five_ops"])
def test_g4_removed_blocks_never_open_the_breaker(g4env, tmp_path, flavor, shape):
    w = new_world(g4env, tmp_path, flavor=flavor, failure_threshold=2)
    keys = [BlockKey(b"gone-%d" % i) for i in range(5)]
    for index, key in enumerate(keys):
        w.spill(key, index)
        _present(w, key)
    if flavor == "object":
        w.store.objects.clear()
    else:
        for key in keys:
            find_files(w.root, key)[0].unlink()
    if shape == "one_op":
        refs = {key: [_mem_descriptor(3 + i)] for i, key in enumerate(keys)}
        result = w.wait(w.kvcr.deliver(refs, "r"))
        assert not any(r.success for r in result.values())
    else:
        for key in keys:
            assert not w.wait(w.deliver(key, 4))[key].success
    assert not _is_open(w)


@pytest.mark.parametrize("flavor", FLAVORS)
def test_g4_a_failed_op_counts_once_however_many_keys(g4env, tmp_path, flavor):
    w = new_world(g4env, tmp_path, flavor=flavor, failure_threshold=2)
    keys = [BlockKey(b"k%d" % i) for i in range(5)]
    for index, key in enumerate(keys):
        w.spill(key, index)
        _present(w, key)
    w.storage.outcome["READ"] = "ERR"
    refs = {key: [_mem_descriptor(3 + i)] for i, key in enumerate(keys)}
    assert not any(r.success for r in w.wait(w.kvcr.deliver(refs, "r")).values())
    w.settle(30)
    assert not _is_open(w)
    _fail_reads(w, keys[0], 1)
    assert _is_open(w)


@pytest.mark.parametrize("flavor", FLAVORS)
def test_g4_open_breaker_fails_reads_with_results_without_io(g4env, tmp_path, flavor):
    w = new_world(g4env, tmp_path, flavor=flavor, failure_threshold=1)
    keys = [A, B, C]
    for index, key in enumerate(keys):
        w.spill(key, index)
    for key in keys:
        _present(w, key)
    w.storage.outcome["READ"] = "ERR"
    assert not w.wait(w.deliver(B, 4))[B].success
    assert _is_open(w)
    reads = len(w.storage.log("READ"))
    w.storage.outcome["READ"] = "DONE"
    assert not w.wait(w.deliver(A, 5))[A].success
    assert not w.wait(w.fetch(C))[C].success
    assert len(w.storage.log("READ")) == reads


# ---- File read-error re-stat probe ----

SENTINEL = BlockKey(b"after-the-probe")


def _file_read_error(w, change=None, *, gate=False):
    """Fail a file READ with ERR; `change` edits the block file while it is held.

    Returns the gate that holds the re-stat probe, or None without `gate`.
    """
    w.spill(A, 0)
    _present(w, A)
    w.storage.outcome["READ"] = "HOLD"
    handle = w.deliver(A, 4)
    wait_for(lambda: len(w.storage.held("READ")) == 1)
    (path,) = find_files(w.root, A)
    if change is not None:
        change(path)
    held = None
    if gate:
        held = threading.Event()
        w.env.gates.append(held)
        w.stat_gate = held
    w.stats_before = len(w.stat_log)
    w.storage.finish_all("ERR")
    assert not w.wait(handle)[A].success
    return held


def _probe_answered(w):
    """Wait for the probe stat, then for a later file job to pass behind it."""
    wait_for(lambda: len(w.stat_log) > w.stats_before)
    wait_for(lambda: w.blocked_checks() == 0)
    assert w.await_check(SENTINEL) == MISS


def _unlink(path):
    path.unlink()


def _rename_over(path):
    other = path.with_name("replacement")
    other.write_bytes(path.read_bytes())
    os.replace(other, path)


def test_g4_file_read_error_on_an_absent_file_does_not_count(g4env, tmp_path):
    w = new_world(g4env, tmp_path, flavor="file", failure_threshold=1)
    _file_read_error(w, _unlink)
    _probe_answered(w)
    assert not _is_open(w)


def test_g4_file_read_error_on_a_replaced_file_does_not_count(g4env, tmp_path):
    w = new_world(g4env, tmp_path, flavor="file", failure_threshold=1)
    _file_read_error(w, _rename_over)
    _probe_answered(w)
    assert not _is_open(w)


def test_g4_file_read_error_on_the_same_file_counts_once_the_probe_answers(
    g4env, tmp_path
):
    w = new_world(g4env, tmp_path, flavor="file", failure_threshold=1)
    _file_read_error(w)
    _probe_answered(w)
    w.poll_until(lambda: _is_open(w))


def test_g4_file_read_error_count_waits_for_the_probe_answer(g4env, tmp_path):
    w = new_world(g4env, tmp_path, flavor="file", failure_threshold=1)
    gate = _file_read_error(w, gate=True)
    wait_for(lambda: w.blocked_checks() == 1)
    assert not _is_open(w)
    gate.set()
    w.poll_until(lambda: _is_open(w))


def test_g4_file_read_error_counts_when_the_probe_stat_fails(g4env, tmp_path):
    w = new_world(g4env, tmp_path, flavor="file", failure_threshold=1)
    w.spill(A, 0)
    _present(w, A)
    w.storage.outcome["READ"] = "HOLD"
    handle = w.deliver(A, 4)
    wait_for(lambda: len(w.storage.held("READ")) == 1)
    stats = len(w.stat_log)
    w.stat_error = True
    w.storage.finish_all("ERR")
    assert not w.wait(handle)[A].success
    wait_for(lambda: len(w.stat_log) > stats)
    w.poll_until(lambda: _is_open(w))


# ---- Breaker and existence checks ----


@pytest.mark.parametrize("flavor", FLAVORS)
def test_g4_check_errors_count_toward_the_breaker(g4env, tmp_path, flavor):
    w = new_world(g4env, tmp_path, flavor=flavor, failure_threshold=2)
    w.fail_checks()
    for index in range(2):
        assert w.await_check(BlockKey(b"down-%d" % index)) == MISS
    assert _is_open(w)


@pytest.mark.parametrize("flavor", FLAVORS)
def test_g4_check_deadline_gives_miss_to_every_subscriber_and_drops_the_late_answer(
    g4env, tmp_path, flavor
):
    w = new_world(g4env, tmp_path, flavor=flavor, failure_threshold=2)
    w.spill(A, 0)
    gate = w.hold_checks()
    assert w.kvcr.query([A], "a") == [fetching_g4()]
    wait_for(lambda: w.blocked_checks() == 1)
    assert w.kvcr.query([A], "b") == [fetching_g4()]
    w.advance(1.5)
    assert w.kvcr.query([A], "a") == [MISS]
    assert w.kvcr.query([A], "b") == [MISS]
    gate.set()
    w.settle()
    # The late answer was dropped.
    assert w.kvcr.query([A], "a") == [fetching_g4()]
    # One timed-out check counted once, so the breaker is still closed.
    assert not _is_open(w)


@pytest.mark.parametrize("flavor", FLAVORS)
def test_g4_check_that_started_and_timed_out_counts_once(g4env, tmp_path, flavor):
    w = new_world(g4env, tmp_path, flavor=flavor, failure_threshold=2)
    gate = w.hold_checks()
    assert w.kvcr.query([A], "a") == [fetching_g4()]
    wait_for(lambda: w.blocked_checks() == 1)
    w.settle(2)
    assert w.kvcr.query([A], "b") == [fetching_g4()]
    w.advance(1.5)
    assert w.kvcr.query([A], "a") == [MISS]
    assert w.kvcr.query([A], "b") == [MISS]
    # The late answer is an error: dropped, never counted a second time.
    w.fail_checks()
    second = w.hold_checks()
    gate.set()
    w.settle()
    assert w.kvcr.query([B], "c") == [fetching_g4()]
    wait_for(lambda: w.blocked_checks() == 1)
    w.settle(2)
    w.advance(3.0)
    assert w.kvcr.query([B], "c") == [MISS]
    assert _is_open(w)
    second.set()


@pytest.mark.parametrize("flavor", FLAVORS)
def test_g4_check_still_queued_at_its_deadline_is_cancelled_and_does_not_count(
    g4env, tmp_path, flavor
):
    w = new_world(g4env, tmp_path, flavor=flavor, failure_threshold=2)
    gate = w.hold_checks()
    assert w.kvcr.query([A], "a") == [fetching_g4()]
    wait_for(lambda: w.blocked_checks() == 1)
    w.settle(2)
    assert w.kvcr.query([B], "b") == [fetching_g4()]
    w.advance(1.5)
    # A reached storage and passed its deadline, so it counts. B never left the
    # queue: the caller stopped waiting, which says nothing about storage.
    assert w.kvcr.query([A], "a") == [MISS]
    assert w.kvcr.query([B], "b") == [MISS]
    gate.set()
    w.settle()
    assert w.check_names() == [digest(A)]
    assert not _is_open(w)


@pytest.mark.parametrize("flavor", FLAVORS)
def test_g4_check_deadline_starts_when_storage_takes_the_job(g4env, tmp_path, flavor):
    w = new_world(g4env, tmp_path, flavor=flavor, failure_threshold=1)
    first = w.hold_checks()
    assert w.kvcr.query([A], "a") == [fetching_g4()]
    wait_for(lambda: w.blocked_checks() == 1)
    w.settle(2)
    assert w.kvcr.query([B], "b") == [fetching_g4()]
    w.advance(0.9)
    w.hold_checks()
    first.set()
    wait_for(lambda: len(w.check_names()) == 2 and w.blocked_checks() == 1)
    w.settle(2)
    assert w.kvcr.query([A], "a") == [MISS]
    # B waited 0.9 s for the worker; its own 1 s starts only now.
    w.advance(1.5)
    assert w.kvcr.query([B], "b") == [fetching_g4()]
    assert not _is_open(w)
    w.advance(2.0)
    assert w.kvcr.query([B], "b") == [MISS]
    assert _is_open(w)


def test_g4_slow_store_answers_every_batched_check_in_time(g4env, tmp_path):
    w = new_world(g4env, tmp_path, flavor="object", check_timeout_ms=1200)
    keys = [BlockKey(b"present-%d" % i) for i in range(64)]
    w.plant(keys)
    w.real_clock()
    # One OBJ query_memory call runs its HEADs concurrently: per-call latency.
    w.storage.query_latency = 0.8
    deadline = time.monotonic() + 10
    answers = {}
    while len(answers) < len(keys) and time.monotonic() < deadline:
        for key, status in zip(keys, w.kvcr.query(keys, "r"), strict=True):
            if status[0] is not QueryStatus.FETCHING:
                answers[key] = status
        w.poll()
        time.sleep(0.002)
    assert len(w.storage.query_calls) == 2
    assert [answers.get(key) for key in keys] == [fetchable_g4()] * 64
    assert not _is_open(w)


@pytest.mark.parametrize("flavor", FLAVORS)
@pytest.mark.parametrize(("check_ms", "expired"), [(None, True), (3000, False)])
def test_g4_check_timeout_defaults_to_the_operation_timeout(
    g4env, tmp_path, flavor, check_ms, expired
):
    options = {} if check_ms is None else {"check_timeout_ms": check_ms}
    w = new_world(g4env, tmp_path, flavor=flavor, **options)
    w.hold_checks()
    assert w.kvcr.query([A], "r") == [fetching_g4()]
    wait_for(lambda: w.blocked_checks() == 1)
    w.settle(2)
    w.advance(1.5)
    assert w.kvcr.query([A], "r") == [MISS if expired else fetching_g4()]
    w.advance(3.5)
    assert w.kvcr.query([A], "r") == [MISS]


def test_g4_one_error_in_an_obj_batch_misses_all_and_counts_once(g4env, tmp_path):
    w = new_world(g4env, tmp_path, flavor="object", failure_threshold=2)
    gate = w.hold_checks()
    keys = [BlockKey(b"k%d" % i) for i in range(4)]
    assert w.kvcr.query(keys[:1], "r") == [fetching_g4()]
    wait_for(lambda: w.blocked_checks() == 1)
    w.storage.query_error_if = lambda names: len(names) > 1
    assert w.kvcr.query(keys[1:], "r") == [fetching_g4()] * 3
    gate.set()
    for key in keys[1:]:
        assert w.await_check(key) == MISS
    assert not _is_open(w)


@pytest.mark.parametrize("flavor", FLAVORS)
def test_g4_opening_the_breaker_drops_checks_that_have_not_started(
    g4env, tmp_path, flavor
):
    w = new_world(g4env, tmp_path, flavor=flavor, failure_threshold=1)
    gate = w.hold_checks()
    assert w.kvcr.query([A], "r") == [fetching_g4()]
    wait_for(lambda: w.blocked_checks() == 1)
    w.advance(0.9)
    assert w.kvcr.query([B, C], "r") == [fetching_g4()] * 2
    # A's check deadline opens the breaker; B and C have not reached theirs.
    w.advance(1.1)
    assert w.kvcr.query([A], "r") == [MISS]
    assert w.kvcr.query([B, C], "r") == [MISS, MISS]
    gate.set()
    w.settle()
    assert w.check_names() == [digest(A)]
