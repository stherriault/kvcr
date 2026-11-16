# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""G4 read side: existence checks, fill, deliver, schema, size checks, threads."""

import ctypes
import hashlib
import os
import re
import threading
import time

import pytest
from _g4_test_utils import (
    PAGE,
    PREFIX,
    digest,
    fetchable_g4,
    fetching_g4,
    find_files,
    hybrid_block_bytes,
    hybrid_refs,
    new_hybrid_world,
    new_world,
    pattern,
    wait_for,
)
from _kvcr_test_utils import _mem_descriptor

from kvcr.types import BlockKey, CacheTier, QueryStatus

FLAVORS = ["file"]
A = BlockKey(b"block-a")
B = BlockKey(b"block-b")


def _present(w, key=A, rid="r"):
    assert w.await_check(key, rid) == fetchable_g4()


def _stored_path(w, key=A):
    (path,) = find_files(w.root, key)
    return path


# ---- Existence check flow ----


@pytest.mark.parametrize("flavor", FLAVORS)
def test_g4_query_absent_block_is_fetching_then_miss(g4env, tmp_path, flavor):
    w = new_world(g4env, tmp_path, flavor=flavor)
    assert w.kvcr.query([A], "r") == [fetching_g4()]
    assert w.await_check(A) == (QueryStatus.MISS, None)


@pytest.mark.parametrize("flavor", FLAVORS)
def test_g4_query_present_block_is_fetching_then_fetchable_g4(g4env, tmp_path, flavor):
    w = new_world(g4env, tmp_path, flavor=flavor)
    w.spill(A, 0)
    assert w.kvcr.query([A], "r") == [fetching_g4()]
    assert w.await_check(A) == fetchable_g4()


@pytest.mark.parametrize("flavor", FLAVORS)
def test_g4_check_result_is_consumed_by_deliver(g4env, tmp_path, flavor):
    w = new_world(g4env, tmp_path, flavor=flavor)
    w.spill(A, 0)
    _present(w)
    assert w.wait(w.deliver(A, 4))[A].success
    reads = len(w.storage.log("READ"))
    assert not w.wait(w.deliver(A, 5))[A].success
    assert len(w.storage.log("READ")) == reads


@pytest.mark.parametrize("flavor", FLAVORS)
def test_g4_deliver_without_a_check_result_fails_without_io(g4env, tmp_path, flavor):
    w = new_world(g4env, tmp_path, flavor=flavor)
    w.spill(A, 0)
    assert not w.wait(w.deliver(A, 4))[A].success
    assert w.storage.log("READ") == []


@pytest.mark.parametrize("flavor", FLAVORS)
def test_g4_block_removed_between_check_and_deliver_fails(g4env, tmp_path, flavor):
    w = new_world(g4env, tmp_path, flavor=flavor)
    w.spill(A, 0)
    _present(w)
    if flavor == "object":
        w.store.objects.clear()
    else:
        _stored_path(w).unlink()
    assert not w.wait(w.deliver(A, 4))[A].success


@pytest.mark.parametrize("flavor", FLAVORS)
def test_g4_block_removed_between_check_and_fetch_leaves_no_ready_slot(
    g4env, tmp_path, flavor
):
    w = new_world(g4env, tmp_path, flavor=flavor)
    w.spill(A, 0)
    _present(w)
    if flavor == "object":
        w.store.objects.clear()
    else:
        _stored_path(w).unlink()
    assert not w.wait(w.fetch(A))[A].success
    w.poll_until(lambda: not w.storage.unreleased())
    assert w.kvcr.query([A])[0][0] is not QueryStatus.HIT
    assert w.deposit(B, 1).success


# ---- Round trips ----


@pytest.mark.parametrize("flavor", FLAVORS)
def test_g4_spill_then_deliver_gives_identical_bytes_without_g2_residency(
    g4env, tmp_path, flavor
):
    w = new_world(g4env, tmp_path, flavor=flavor)
    w.spill(A, 0)
    w.clear(4)
    _present(w)
    assert w.wait(w.deliver(A, 4))[A].success
    assert w.element(4) == pattern(0, w.piece)
    assert w.kvcr.query([A])[0][0] is not QueryStatus.HIT


@pytest.mark.parametrize("flavor", FLAVORS)
def test_g4_spill_then_fetch_gives_identical_bytes_and_a_g2_hit(
    g4env, tmp_path, flavor
):
    w = new_world(g4env, tmp_path, flavor=flavor)
    w.spill(A, 0)
    _present(w)
    result = w.wait(w.fetch(A))[A]
    assert result.success and result.release_handle is not None
    assert bytes(w.local[: w.piece]) == pattern(0, w.piece)
    assert w.kvcr.query([A]) == [(QueryStatus.HIT, CacheTier.LOCAL_G2)]
    assert any(source is CacheTier.G4 for _, source in w.policy.ingested)
    w.kvcr.release((result.release_handle,))


@pytest.mark.parametrize("flavor", FLAVORS)
def test_g4_second_instance_loads_blocks_written_by_the_first(g4env, tmp_path, flavor):
    first = new_world(g4env, tmp_path, flavor=flavor)
    first.spill(A, 0)
    first.kvcr.close()
    second = new_world(
        g4env,
        tmp_path,
        flavor=flavor,
        store=first.store,
        root=first.root,
        name="host-b",
    )
    second.clear(4)
    _present(second)
    assert second.wait(second.deliver(A, 4))[A].success
    assert second.element(4) == pattern(0, second.piece)


# ---- Naming ----


@pytest.mark.parametrize("flavor", FLAVORS)
def test_g4_block_name_is_prefix_tag_fanout_digest(g4env, tmp_path, flavor):
    w = new_world(g4env, tmp_path, flavor=flavor)
    w.spill(A, 0)
    d = hashlib.sha256(bytes(A)).hexdigest()
    pattern_re = re.compile(rf"^{re.escape(PREFIX)}/[0-9a-f]{{8}}/{d[:2]}/{d}$")
    if flavor == "object":
        assert pattern_re.match(w.store.find(A))
    else:
        assert pattern_re.match(str(_stored_path(w).relative_to(w.root)))


# ---- Schema, file flavor ----


def _hybrid_spill(w, layout):
    keys = [BlockKey(b"h0"), BlockKey(b"h1")]
    original = hybrid_block_bytes(w, 0)
    for index, key in enumerate(keys):
        assert w.wait(w.kvcr.deposit({key: hybrid_refs(index, layout)}))[key].success
    assert w.stored(keys[0])
    return keys[0], original


@pytest.mark.parametrize("layout", [("full", "swa:a", "swa:b"), ("full", "swa", "swa")])
def test_g4_file_multi_piece_block_round_trips_by_deliver(g4env, tmp_path, layout):
    w = new_hybrid_world(g4env, tmp_path, key_layout=layout)
    key, original = _hybrid_spill(w, layout)
    _present(w, key)
    assert w.wait(w.kvcr.deliver({key: hybrid_refs(2, layout)}, "r"))[key].success
    assert hybrid_block_bytes(w, 2) == original


@pytest.mark.parametrize("layout", [("full", "swa:a", "swa:b"), ("full", "swa", "swa")])
def test_g4_file_multi_piece_block_round_trips_by_fetch(g4env, tmp_path, layout):
    w = new_hybrid_world(g4env, tmp_path, key_layout=layout)
    key, original = _hybrid_spill(w, layout)
    _present(w, key)
    handle = w.kvcr.fetch((key,), "r", expected_layout=list(layout))
    result = w.wait(handle)[key]
    assert result.success
    got = (
        bytes(w.local_bufs["full"][: 2 * PAGE])
        + bytes(w.local_bufs["swa"][:PAGE])
        + bytes(w.local_bufs["swa"][PAGE : 2 * PAGE])
    )
    assert got == original


def test_g4_file_named_layout_serves_partial_and_reordered_reads(g4env, tmp_path):
    layout = ("full", "swa:a", "swa:b")
    w = new_hybrid_world(g4env, tmp_path, key_layout=layout)
    key, original = _hybrid_spill(w, layout)
    piece_a, piece_b = original[2 * PAGE : 3 * PAGE], original[3 * PAGE :]
    swa = w.primary["swa"]

    _present(w, key)
    refs = [_mem_descriptor(4, "swa:b"), _mem_descriptor(5, "swa:a")]
    assert w.wait(w.kvcr.deliver({key: refs}, "r"))[key].success
    assert bytes(swa[4 * PAGE : 5 * PAGE]) == piece_b
    assert bytes(swa[5 * PAGE : 6 * PAGE]) == piece_a

    _present(w, key)
    assert w.wait(w.kvcr.deliver({key: [_mem_descriptor(3, "swa:b")]}, "r"))[
        key
    ].success
    assert bytes(swa[3 * PAGE : 4 * PAGE]) == piece_b


def test_g4_file_repeated_layout_ambiguous_partial_has_no_g4_source(g4env, tmp_path):
    layout = ("full", "swa", "swa")
    w = new_hybrid_world(g4env, tmp_path, key_layout=layout)
    key, _ = _hybrid_spill(w, layout)
    before = bytes(w.primary["swa"][4 * PAGE : 5 * PAGE])
    _present(w, key)
    deliver = w.kvcr.deliver({key: [_mem_descriptor(4, "swa")]}, "r")
    assert not w.wait(deliver)[key].success
    assert bytes(w.primary["swa"][4 * PAGE : 5 * PAGE]) == before
    assert w.storage.log("READ") == []
    _present(w, key)
    fetch = w.kvcr.fetch((key,), "r", expected_layout=["swa"])
    assert not w.wait(fetch)[key].success
    assert w.storage.log("READ") == []


def test_g4_key_with_another_layout_is_not_spilled(g4env, tmp_path):
    layout = ("full", "swa:a", "swa:b")
    w = new_hybrid_world(g4env, tmp_path, key_layout=layout)
    odd, full = BlockKey(b"odd"), BlockKey(b"full")
    only_full = [_mem_descriptor(0, "full")]
    assert w.wait(w.kvcr.deposit({odd: only_full}))[odd].success
    assert w.wait(w.kvcr.deposit({full: hybrid_refs(1, layout)}))[full].success
    assert not w.stored(odd)
    assert w.storage.log("WRITE") == []


# ---- One handle per key, OBJ devIds, registrations ----


@pytest.mark.parametrize("flavor", FLAVORS)
def test_g4_deliver_uses_one_handle_per_key(g4env, tmp_path, flavor):
    w = new_world(g4env, tmp_path, flavor=flavor)
    w.spill(A, 0)
    w.spill(B, 1)
    w.storage.outcome["READ"] = "HOLD"
    _present(w, A)
    _present(w, B)
    handle = w.kvcr.deliver({A: [_mem_descriptor(4)], B: [_mem_descriptor(5)]}, "r")
    wait_for(lambda: len(w.storage.held("READ")) == 2)
    reads = w.storage.log("READ")
    assert len(reads) == 2 and all(len(x.remote) == 1 for x in reads)
    assert len({x.remote[0][2] for x in reads}) == 2
    w.storage.finish_all("DONE")
    result = w.wait(handle)
    assert result[A].success and result[B].success


def test_g4_file_multi_piece_key_is_one_handle_with_a_descriptor_per_piece(
    g4env, tmp_path
):
    layout = ("full", "swa:a", "swa:b")
    w = new_hybrid_world(g4env, tmp_path, key_layout=layout)
    key, _ = _hybrid_spill(w, layout)
    _present(w, key)
    assert w.wait(w.kvcr.deliver({key: hybrid_refs(2, layout)}, "r"))[key].success
    (read,) = w.storage.log("READ")
    assert [(row[0], row[1]) for row in read.remote] == [
        (0, 2 * PAGE),
        (2 * PAGE, PAGE),
        (3 * PAGE, PAGE),
    ]
    assert len({row[2] for row in read.remote}) == 1


@pytest.mark.parametrize("flavor", FLAVORS)
def test_g4_registers_dram_regions_on_the_storage_agent_and_drops_them_at_close(
    g4env, tmp_path, flavor
):
    w = new_world(g4env, tmp_path, flavor=flavor)
    covered = [ctypes.addressof(w.engine), ctypes.addressof(w.local)]
    rows = [row for reg in w.storage.dram_regs for row in reg.rows]
    for addr in covered:
        assert any(row[0] <= addr < row[0] + row[1] for row in rows)
    w.kvcr.close()
    assert w.storage.live_dram == []


@pytest.mark.parametrize("flavor", FLAVORS)
def test_g4_storage_agent_calls_run_on_one_worker_thread(g4env, tmp_path, flavor):
    w = new_world(g4env, tmp_path, flavor=flavor)
    w.spill(A, 0)
    _present(w)
    assert w.wait(w.deliver(A, 4))[A].success
    _present(w)
    assert w.wait(w.fetch(A))[A].success
    threads = w.storage.threads()
    assert len(threads) == 1
    assert threading.get_ident() not in threads
    assert not threads & w.main.threads()


# ---- Caller thread never touches storage ----


@pytest.mark.parametrize("flavor", FLAVORS)
def test_g4_query_makes_no_storage_call_on_the_caller_thread(
    g4env, tmp_path, flavor, monkeypatch
):
    w = new_world(g4env, tmp_path, flavor=flavor)
    w.spill(A, 0)
    if flavor == "object":
        w.storage.query_gate = threading.Event()
    me = threading.get_ident()
    seen = []
    storage_calls = len(w.storage.calls)
    main_calls = len(w.main.calls)
    with monkeypatch.context() as scoped:
        for name in ("stat", "lstat", "open"):
            real = getattr(os, name)

            def spy(*args, _real=real, _name=name, **kwargs):
                if threading.get_ident() == me:
                    seen.append(_name)
                return _real(*args, **kwargs)

            scoped.setattr(os, name, spy)
        start = time.monotonic()
        assert w.kvcr.query([A], "r") == [fetching_g4()]
        elapsed = time.monotonic() - start
    assert seen == []
    assert elapsed < 1.0
    caller_storage = [c for c in w.storage.calls[storage_calls:] if c[1] == me]
    caller_main = [c for c in w.main.calls[main_calls:] if c[1] == me]
    assert caller_storage == [] and caller_main == []


def test_g4_file_checks_run_on_the_file_worker_not_the_storage_thread(g4env, tmp_path):
    w = new_world(g4env, tmp_path, flavor="file")
    w.spill(A, 0)
    _present(w)
    assert w.wait(w.deliver(A, 4))[A].success
    assert "query_memory" not in [name for name, _ in w.storage.calls]
    assert len(w.stat_threads) == 1
    assert w.stat_threads.isdisjoint(w.storage.threads())
    assert threading.get_ident() not in w.stat_threads


# ---- Failed transfers are never SUCCESS ----


@pytest.mark.parametrize("flavor", FLAVORS)
def test_g4_deliver_read_error_is_failed(g4env, tmp_path, flavor):
    w = new_world(g4env, tmp_path, flavor=flavor)
    w.spill(A, 0)
    _present(w)
    w.storage.outcome["READ"] = "ERR"
    assert not w.wait(w.deliver(A, 4))[A].success


@pytest.mark.parametrize("flavor", FLAVORS)
def test_g4_fetch_read_error_leaves_no_ready_slot(g4env, tmp_path, flavor):
    w = new_world(g4env, tmp_path, flavor=flavor)
    w.spill(A, 0)
    _present(w)
    w.storage.outcome["READ"] = "ERR"
    assert not w.wait(w.fetch(A))[A].success
    w.poll_until(lambda: not w.storage.unreleased())
    assert w.kvcr.query([A])[0][0] is not QueryStatus.HIT
    assert w.deposit(B, 1).success


# The real bindings raise from the post or the state poll; they do not return ERR.
RAISED = ["RAISE_POST", "RAISE_POLL"]


@pytest.mark.parametrize("flavor", FLAVORS)
@pytest.mark.parametrize("failure", RAISED)
def test_g4_deliver_transfer_that_raises_is_failed_and_released(
    g4env, tmp_path, flavor, failure
):
    w = new_world(g4env, tmp_path, flavor=flavor)
    w.spill(A, 0)
    _present(w)
    w.storage.outcome["READ"] = failure
    assert not w.wait(w.deliver(A, 4))[A].success
    w.poll_until(lambda: not w.storage.unreleased())
    assert w.storage.violations == []
    assert w.storage.live_dev == {} and w.storage.live_fds == set()


@pytest.mark.parametrize("flavor", FLAVORS)
@pytest.mark.parametrize("failure", RAISED)
def test_g4_fetch_transfer_that_raises_leaves_no_ready_slot(
    g4env, tmp_path, flavor, failure
):
    w = new_world(g4env, tmp_path, flavor=flavor)
    w.spill(A, 0)
    _present(w)
    w.storage.outcome["READ"] = failure
    assert not w.wait(w.fetch(A))[A].success
    w.poll_until(lambda: not w.storage.unreleased())
    assert w.storage.violations == []
    assert w.kvcr.query([A])[0][0] is not QueryStatus.HIT
    assert w.deposit(B, 1).success


# ---- File size checks and replace-after-open ----


@pytest.mark.parametrize("delta", [-1, 1])
def test_g4_file_stat_size_mismatch_is_a_miss(g4env, tmp_path, delta):
    w = new_world(g4env, tmp_path, flavor="file")
    w.spill(A, 0)
    os.truncate(_stored_path(w), w.piece + delta)
    assert w.await_check(A) == (QueryStatus.MISS, None)
    assert w.storage.log("READ") == []


def test_g4_file_size_change_after_check_is_failed_at_open(g4env, tmp_path):
    w = new_world(g4env, tmp_path, flavor="file")
    w.spill(A, 0)
    _present(w)
    os.truncate(_stored_path(w), w.piece // 2)
    assert not w.wait(w.deliver(A, 4))[A].success


def test_g4_file_short_read_is_failed(g4env, tmp_path):
    w = new_world(g4env, tmp_path, flavor="file")
    w.spill(A, 0)
    _present(w)
    path = _stored_path(w)
    w.storage.before_transfer = lambda x: (
        os.truncate(path, w.piece // 2) if x.op == "READ" else None
    )
    assert not w.wait(w.deliver(A, 4))[A].success


def test_g4_file_replaced_after_open_serves_the_opened_inode(g4env, tmp_path):
    w = new_world(g4env, tmp_path, flavor="file")
    w.spill(A, 0)
    w.clear(4)
    _present(w)
    w.storage.outcome["READ"] = "HOLD"
    handle = w.deliver(A, 4)
    wait_for(lambda: len(w.storage.held("READ")) == 1)
    path = _stored_path(w)
    replacement = path.with_name("replacement")
    replacement.write_bytes(b"\xee" * w.piece)
    os.rename(replacement, path)
    w.storage.finish_all("DONE")
    assert w.wait(handle)[A].success
    assert w.element(4) == pattern(0, w.piece)


# ---- Check table: sharing, bounds, queueing, caller pump ----


@pytest.mark.parametrize("flavor", FLAVORS)
def test_g4_query_without_request_id_is_miss_and_starts_no_check(
    g4env, tmp_path, flavor
):
    w = new_world(g4env, tmp_path, flavor=flavor)
    w.spill(A, 0)
    before = w.check_calls()
    assert w.kvcr.query([A]) == [(QueryStatus.MISS, None)]
    w.settle()
    assert w.check_calls() == before


@pytest.mark.parametrize("flavor", FLAVORS)
def test_g4_two_requests_share_one_check_and_read_into_own_destinations(
    g4env, tmp_path, flavor
):
    w = new_world(g4env, tmp_path, flavor=flavor)
    w.spill(A, 0)
    before = w.check_calls()
    gate = w.hold_checks()
    assert w.kvcr.query([A], "a") == [fetching_g4()]
    wait_for(lambda: w.blocked_checks() == 1)
    assert w.kvcr.query([A], "b") == [fetching_g4()]
    gate.set()
    _present(w, A, "a")
    _present(w, A, "b")
    assert w.check_calls() - before == 1

    w.storage.outcome["READ"] = "HOLD"
    for index in (4, 5):
        w.clear(index)
    first, second = w.deliver(A, 4, "a"), w.deliver(A, 5, "b")
    wait_for(lambda: len(w.storage.held("READ")) == 2)
    assert len({x.remote[0][2] for x in w.storage.held("READ")}) == 2
    w.storage.finish_all("DONE")
    assert w.wait(first)[A].success and w.wait(second)[A].success
    assert w.element(4) == w.element(5) == pattern(0, w.piece)


@pytest.mark.parametrize("flavor", FLAVORS)
def test_g4_full_check_table_queues_keys_instead_of_missing(g4env, tmp_path, flavor):
    w = new_world(g4env, tmp_path, flavor=flavor, max_inflight_checks=2)
    gate = w.hold_checks()
    keys = [BlockKey(f"k{i}".encode()) for i in range(5)]
    assert w.kvcr.query(keys, "r") == [fetching_g4()] * 5
    gate.set()
    for key in keys:
        assert w.await_check(key) == (QueryStatus.MISS, None)
    assert len(w.check_names()) == 5


@pytest.mark.parametrize("flavor", FLAVORS)
def test_g4_sixty_five_keys_from_two_requests_all_end_fetchable(
    g4env, tmp_path, flavor
):
    w = new_world(g4env, tmp_path, flavor=flavor)
    keys = [BlockKey(f"key-{i}".encode()) for i in range(65)]
    for index, key in enumerate(keys):
        w.spill(key, index % 4)
    gate = w.hold_checks()
    first, second = keys[:33], keys[33:]
    seen = w.kvcr.query(first, "a") + w.kvcr.query(second, "b")
    assert all(status == fetching_g4() for status in seen)
    gate.set()
    for key in first:
        assert w.await_check(key, "a") == fetchable_g4()
    for key in second:
        assert w.await_check(key, "b") == fetchable_g4()


@pytest.mark.parametrize("flavor", FLAVORS)
def test_g4_query_alone_sees_check_results_without_poll_completed(
    g4env, tmp_path, flavor
):
    w = new_world(g4env, tmp_path, flavor=flavor)
    w.spill(A, 0)
    assert w.kvcr.query([A], "r") == [fetching_g4()]
    assert w.query_only(A) == fetchable_g4()


# ---- OBJ check batching ----


def test_g4_file_checks_run_in_queue_order(g4env, tmp_path):
    w = new_world(g4env, tmp_path, flavor="file")
    gate = w.hold_checks()
    keys = [BlockKey(f"k{i}".encode()) for i in range(5)]
    assert w.kvcr.query(keys[:1], "r") == [fetching_g4()]
    wait_for(lambda: w.blocked_checks() == 1)
    assert w.kvcr.query(keys[1:], "r") == [fetching_g4()] * 4
    gate.set()
    for key in keys:
        w.await_check(key)
    assert w.check_names() == [digest(key) for key in keys]


# ---- Active-read eviction and direct_io modes ----


@pytest.mark.parametrize("flavor", FLAVORS)
def test_g4_block_deleted_during_an_active_read_completes_with_the_old_block(
    g4env, tmp_path, flavor
):
    w = new_world(g4env, tmp_path, flavor=flavor)
    w.spill(A, 0)
    w.clear(4)
    _present(w)
    w.storage.outcome["READ"] = "HOLD"
    handle = w.deliver(A, 4)
    wait_for(lambda: len(w.storage.held("READ")) == 1)
    if flavor == "object":
        w.store.objects.clear()
    else:
        _stored_path(w).unlink()
    w.storage.finish_all("DONE")
    assert w.wait(handle)[A].success
    assert w.element(4) == pattern(0, w.piece)


@pytest.mark.parametrize("direct_io", [False, True])
def test_g4_file_direct_io_mode_sets_o_direct_and_round_trips(
    g4env, tmp_path, direct_io
):
    w = new_world(g4env, tmp_path, flavor="file", direct_io=direct_io)
    w.spill(A, 0)
    assert w.storage.fd_direct
    assert all(flag is direct_io for flag in w.storage.fd_direct)
    w.clear(4)
    _present(w)
    assert w.wait(w.deliver(A, 4))[A].success
    assert w.element(4) == pattern(0, w.piece)
