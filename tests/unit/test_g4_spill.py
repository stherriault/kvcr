# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""G4 write side: spill mirroring G3, atomic publish, and the circuit breaker."""

import itertools
import os
import re

import pytest
from _g4_test_utils import (
    PAGE,
    fetchable_g4,
    fetching_g4,
    find_files,
    new_world,
    pattern,
    tmp_files,
    wait_for,
)

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


def _capacity_key(w):
    return w.local_dram._pools._capacity_eviction_key


_PROBES = itertools.count()


def _is_open(w):
    """An open breaker answers MISS at once and starts no check."""
    key = BlockKey(b"probe-%d" % next(_PROBES))
    calls = len(w.storage.query_calls)
    answer = w.kvcr.query([key], "probe")[0]
    w.settle(5)
    return answer == MISS and len(w.storage.query_calls) == calls


# ---- Spill mirrors G3 (AC4, AC5) ----


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
    w = new_world(g4env, tmp_path, flavor="object")
    w.storage.outcome["WRITE"] = "HOLD"
    assert w.deposit(A, 0).success
    waiting = w.deposit(B, 1, wait=False)
    w.poll_until(lambda: len(w.storage.held("WRITE")) == 1)
    w.advance(2.0)
    assert not w.wait(waiting)[B].success
    assert _capacity_key(w) == A
    w.storage.finish_all("DONE")
    w.poll_until(lambda: _capacity_key(w) is None)


def test_g4_file_spill_past_its_deadline_is_not_published(g4env, tmp_path):
    w = new_world(g4env, tmp_path, flavor="file")
    w.storage.outcome["WRITE"] = "HOLD"
    assert w.deposit(A, 0).success
    waiting = w.deposit(B, 1, wait=False)
    w.poll_until(lambda: len(w.storage.held("WRITE")) == 1)
    w.advance(2.0)
    w.wait(waiting)
    w.storage.finish_all("DONE")
    w.poll_until(lambda: not w.storage.unreleased() and not tmp_files(w.root))
    assert not w.stored(A)


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


# ---- Atomic publish (AC7) ----


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
    assert w.deposit(A, 0).success
    assert w.deposit(B, 1).success
    w.poll_until(lambda: not tmp_files(w.root) and not w.storage.unreleased())
    (final,) = find_files(w.root, A)
    assert final.read_bytes() == b"\x42" * PAGE
    assert tmp_files(w.root) == []


# ---- Circuit breaker (AC4) ----


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


def test_g4_check_errors_count_toward_the_breaker(g4env, tmp_path):
    w = new_world(g4env, tmp_path, flavor="object", failure_threshold=2)
    w.storage.query_error = RuntimeError("endpoint down")
    for index in range(2):
        assert w.await_check(BlockKey(b"down-%d" % index)) == MISS
    assert _is_open(w)


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
    assert w.kvcr.query([BlockKey(b"fresh")], "r")[0] == fetching_g4()
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
