# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""G4 read side: existence checks, fill, deliver, schema, size checks, threads."""

import ctypes
import hashlib
import os
import re
import stat
import threading
import time
from unittest.mock import patch

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

FLAVORS = ["object", "file"]
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


def test_g4_discard_hint_drops_an_unconsumed_result(g4env, tmp_path):
    w = new_world(g4env, tmp_path, flavor="object")
    w.spill(A, 0)
    _present(w, A, "a")
    w.kvcr.discard_hint("a")
    assert w.kvcr.query([A], "a") == [fetching_g4()]


def test_g4_start_check_keys_results_by_owner_id(g4env, tmp_path):
    w = new_world(g4env, tmp_path, flavor="object")
    w.spill(A, 0)
    w.kvcr._core._g4.start_check([A], "prefetch-1")
    assert w.await_check(A, "prefetch-1") == fetchable_g4()


def test_g4_unchecked_fill_does_not_evict_a_hot_g2_block(g4env, tmp_path):
    w = new_world(g4env, tmp_path, flavor="object")
    assert w.deposit(A, 0).success
    handle = w.fetch(B)
    assert not w.wait(handle)[B].success
    assert w.kvcr.query([A]) == [(QueryStatus.HIT, CacheTier.LOCAL_G2)]
    assert w.storage.log() == []


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


def test_g4_mixed_g2_and_g4_deliver_joins_into_one_result(g4env, tmp_path):
    w = new_world(g4env, tmp_path, flavor="object")
    w.spill(A, 0)
    resident = w.filler
    _present(w)
    handle = w.kvcr.deliver(
        {A: [_mem_descriptor(4)], resident: [_mem_descriptor(5)]}, "r"
    )
    result = w.wait(handle)
    assert result[A].success and result[resident].success
    assert w.element(4) == pattern(0, w.piece)
    assert w.element(5) == pattern(7, w.piece)


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


def test_g4_other_key_prefix_does_not_see_the_block(g4env, tmp_path):
    first = new_world(g4env, tmp_path, flavor="object")
    first.spill(A, 0)
    other = new_world(
        g4env,
        tmp_path,
        flavor="object",
        store=first.store,
        name="host-b",
        key_prefix=PREFIX + "/other",
    )
    assert other.await_check(A) == (QueryStatus.MISS, None)


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


def test_g4_layout_tag_depends_on_key_layout_and_block_size(g4env, tmp_path):
    def tag_of(world):
        return world.store.find(A).split("/")[-3]

    one = new_world(g4env, tmp_path, flavor="object", pages=1)
    one.spill(A, 0)
    two = new_world(g4env, tmp_path, flavor="object", pages=2, name="host-b")
    two.spill(A, 0)
    again = new_world(g4env, tmp_path, flavor="object", pages=1, name="host-c")
    again.spill(A, 0)
    assert tag_of(one) != tag_of(two)
    assert tag_of(one) == tag_of(again)

    (tmp_path / "n").mkdir()
    (tmp_path / "r").mkdir()
    named = new_hybrid_world(
        g4env, tmp_path / "n", key_layout=("full", "swa:a", "swa:b")
    )
    repeated = new_hybrid_world(
        g4env, tmp_path / "r", key_layout=("full", "swa", "swa")
    )
    tags = []
    for world, layout in ((named, named.layout), (repeated, repeated.layout)):
        for index in (0, 1):
            key = BlockKey(f"h{index}".encode())
            assert world.wait(world.kvcr.deposit({key: hybrid_refs(index, layout)}))[
                key
            ].success
        (path,) = find_files(world.root, BlockKey(b"h0"))
        tags.append(path.parent.parent.name)
    assert tags[0] != tags[1]


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


def test_g4_obj_devids_are_unique_for_overlapping_reads_and_spills(g4env, tmp_path):
    w = new_world(g4env, tmp_path, flavor="object")
    w.spill(A, 0)
    _present(w)
    w.storage.outcome.update(READ="HOLD", WRITE="HOLD")
    read = w.deliver(A, 4)
    wait_for(lambda: len(w.storage.held("READ")) == 1)
    spill = w.deposit(B, 1, wait=False)
    wait_for(lambda: len(w.storage.held("WRITE")) == 1)
    devs = [x.remote[0][2] for x in w.storage.held()]
    dram_devs = {row[2] for reg in w.storage.dram_regs for row in reg.rows}
    assert len(set(devs)) == len(devs) == 2
    assert not set(devs) & dram_devs
    w.storage.finish_all("DONE")
    assert w.wait(read)[A].success and w.wait(spill)[B].success


def test_g4_obj_registrations_follow_the_handle_and_devids_never_overlap(
    g4env, tmp_path
):
    w = new_world(g4env, tmp_path, flavor="object")
    for index, key in enumerate((A, B, BlockKey(b"c"))):
        w.spill(key, index)
    for outcome in ("DONE", "ERR", "DONE"):
        w.storage.outcome["READ"] = outcome
        _present(w, A)
        w.wait(w.deliver(A, 4))
    w.poll_until(lambda: not w.storage.live_dev and not w.storage.unreleased())
    assert w.storage.violations == []
    assert w.storage.devid_overwrites == []


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


@pytest.mark.parametrize("flavor", FLAVORS)
def test_g4_main_agent_transfers_are_not_delayed_by_a_hung_head(
    g4env, tmp_path, flavor
):
    w = new_world(g4env, tmp_path, flavor=flavor, slots=2)
    w.hold_checks()
    assert w.kvcr.query([A], "r") == [fetching_g4()]
    wait_for(lambda: w.blocked_checks() == 1)
    transfers = len(w.main.transfers)
    assert w.deposit(B, 1).success
    assert len(w.main.transfers) > transfers
    assert w.kvcr.query([B]) == [(QueryStatus.HIT, CacheTier.LOCAL_G2)]


@pytest.mark.parametrize("flavor", FLAVORS)
def test_g4_checks_probes_and_transfers_share_the_one_storage_agent(
    g4env, tmp_path, flavor
):
    w = new_world(g4env, tmp_path, flavor=flavor)
    w.spill(A, 0)
    _present(w)
    w.storage.outcome["READ"] = "ERR"
    assert not w.wait(w.deliver(A, 4))[A].success
    names = [name for name, _ in w.storage.calls]
    assert "query_memory" in names
    assert "initialize_xfer" in names
    assert w.storage.check_peer is None
    assert w.check_agent is w.storage


@pytest.mark.parametrize("flavor", FLAVORS)
def test_g4_every_storage_call_runs_on_one_thread(g4env, tmp_path, flavor):
    w = new_world(g4env, tmp_path, flavor=flavor)
    w.spill(A, 0)
    _present(w)
    w.storage.outcome["READ"] = "ERR"
    assert not w.wait(w.deliver(A, 4))[A].success
    assert w.kvcr.query([B], "b") == [fetching_g4()]
    wait_for(lambda: len(w.check_names()) >= 2)
    threads = w.storage.threads()
    assert len(threads) == 1
    assert threading.get_ident() not in threads
    assert threads.isdisjoint(w.main.threads())


def test_g4_file_syscalls_run_on_the_storage_thread(g4env, tmp_path):
    w = new_world(g4env, tmp_path, flavor="file")
    w.spill(A, 0)
    _present(w)
    w.storage.outcome["READ"] = "ERR"
    assert not w.wait(w.deliver(A, 4))[A].success
    wait_for(lambda: len(w.stat_log) > 0)
    assert w.stat_threads == w.storage.threads()


def test_g4_file_checks_ask_the_nixl_file_query_for_the_final_path(g4env, tmp_path):
    w = new_world(g4env, tmp_path, flavor="file")
    w.spill(A, 0)
    _present(w)
    (path,) = find_files(w.root, A)
    assert w.storage.query_rows == [((0, 0, 0, str(path)),)]


def test_g4_file_checks_use_the_file_memory_type_on_the_posix_backend(g4env, tmp_path):
    w = new_world(g4env, tmp_path, flavor="file")
    w.spill(A, 0)
    _present(w)
    assert w.storage.query_types == [("FILE", "POSIX")]


def test_g4_file_checks_do_not_stat_the_block_path(g4env, tmp_path):
    w = new_world(g4env, tmp_path, flavor="file")
    w.spill(A, 0)
    _present(w)
    assert w.await_check(BlockKey(b"never-stored"), "m") == (QueryStatus.MISS, None)
    assert w.stat_log == []


def test_g4_file_check_of_a_missing_file_is_a_miss_from_a_none_answer(g4env, tmp_path):
    w = new_world(g4env, tmp_path, flavor="file")
    assert w.await_check(A) == (QueryStatus.MISS, None)
    assert w.check_names() == [digest(A)]
    assert w.storage.log("READ") == []


@pytest.mark.parametrize("mode", [stat.S_IFDIR | 0o755, stat.S_IFIFO | 0o644])
def test_g4_file_check_of_a_non_regular_file_is_a_miss(g4env, tmp_path, mode):
    w = new_world(g4env, tmp_path, flavor="file")
    w.spill(A, 0)
    w.storage.file_mode = mode
    assert w.await_check(A) == (QueryStatus.MISS, None)
    assert w.check_names() == [digest(A)]
    assert w.storage.log("READ") == []


def test_g4_file_check_of_a_regular_file_with_permission_bits_is_fetchable(
    g4env, tmp_path
):
    w = new_world(g4env, tmp_path, flavor="file")
    w.spill(A, 0)
    w.storage.file_mode = stat.S_IFREG | 0o600
    assert w.await_check(A) == fetchable_g4()


@pytest.mark.parametrize("flavor", FLAVORS)
def test_g4_blocked_check_pauses_a_transfer_until_the_query_returns(
    g4env, tmp_path, flavor
):
    w = new_world(
        g4env, tmp_path, flavor=flavor, failure_threshold=2, check_timeout_ms=500
    )
    w.spill(A, 0)
    _present(w)
    gate = w.hold_checks()
    assert w.kvcr.query([B], "b") == [fetching_g4()]
    wait_for(lambda: w.blocked_checks() == 1)
    w.storage.outcome["READ"] = "HOLD"
    handle = w.deliver(A, 4)
    w.settle(20)
    # One thread serves queries and transfers: the read waits behind the query.
    assert w.storage.held("READ") == []
    w.advance(0.7)
    # The check still resolves at its deadline and counts once.
    assert w.kvcr.query([B], "b") == [(QueryStatus.MISS, None)]
    assert w.g4_state()["g4_breaker_failures"] == 1
    assert not w.breaker_open()
    gate.set()
    wait_for(lambda: len(w.storage.held("READ")) == 1)
    w.storage.finish_all("DONE")
    assert w.wait(handle)[A].success
    assert w.g4_state()["g4_breaker_failures"] == 1


@pytest.mark.parametrize("flavor", FLAVORS)
def test_g4_blocked_probe_pauses_the_next_transfer_until_it_returns(
    g4env, tmp_path, flavor
):
    w = new_world(g4env, tmp_path, flavor=flavor, failure_threshold=3)
    w.spill(A, 0)
    w.spill(B, 1)
    _present(w, A, "ra")
    _present(w, B, "rb")
    w.storage.sequence["READ"] = ["ERR", "DONE"]
    gate = w.hold_probes()
    assert not w.wait(w.deliver(A, 4, "ra"))[A].success
    wait_for(lambda: w.blocked_checks() == 1)
    handle = w.deliver(B, 5, "rb")
    w.settle(20)
    assert len(w.storage.log("READ")) == 1
    gate.set()
    assert w.wait(handle)[B].success
    assert len(w.storage.log("READ")) == 2
    assert not w.breaker_open()


# ---- Failed transfers are never SUCCESS ----


@pytest.mark.parametrize("flavor", FLAVORS)
def test_g4_deliver_read_error_is_failed(g4env, tmp_path, flavor):
    w = new_world(g4env, tmp_path, flavor=flavor)
    w.spill(A, 0)
    _present(w)
    w.storage.outcome["READ"] = "ERR"
    assert not w.wait(w.deliver(A, 4))[A].success


@pytest.mark.parametrize("flavor", FLAVORS)
def test_g4_read_error_completes_before_the_classification_probe(
    g4env, tmp_path, flavor
):
    w = new_world(g4env, tmp_path, flavor=flavor)
    w.spill(A, 0)
    _present(w)
    w.storage.outcome["READ"] = "ERR"
    gate = w.hold_probes()
    assert not w.wait(w.deliver(A, 4))[A].success
    gate.set()


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


def _success_flag(call):
    if "success" in call.kwargs:
        return call.kwargs["success"]
    return call.args[1]


def test_g4_fetch_deadline_then_late_done_completes_the_fill_once_as_failed(
    g4env, tmp_path
):
    w = new_world(g4env, tmp_path, flavor="object")
    w.spill(A, 0)
    _present(w)
    w.storage.outcome["READ"] = "HOLD"
    with patch.object(
        w.local_dram, "complete_fill", wraps=w.local_dram.complete_fill
    ) as spy:
        handle = w.fetch(A)
        # The fetch first spills the filler; its READ starts after that completion.
        w.poll_until(lambda: len(w.storage.held("READ")) == 1)
        w.advance(2.0)
        assert not w.wait(handle)[A].success
        assert spy.call_count == 0
        w.storage.finish_all("DONE")
        w.poll_until(lambda: spy.call_count >= 1)
        w.settle()
    assert spy.call_count == 1 and _success_flag(spy.call_args_list[0]) is False
    assert w.kvcr.query([A])[0][0] is not QueryStatus.HIT


def test_g4_deliver_deadline_then_late_done_is_failed(g4env, tmp_path):
    w = new_world(g4env, tmp_path, flavor="object")
    w.spill(A, 0)
    _present(w)
    w.storage.outcome["READ"] = "HOLD"
    handle = w.deliver(A, 4)
    wait_for(lambda: len(w.storage.held("READ")) == 1)
    w.advance(2.0)
    w.settle()
    assert handle not in w.results
    w.storage.finish_all("DONE")
    assert not w.wait(handle)[A].success
    assert w.events == []


def test_g4_deliver_past_abandon_timeout_reports_uncertain_then_quiesced(
    g4env, tmp_path
):
    w = new_world(g4env, tmp_path, flavor="object")
    w.spill(A, 0)
    _present(w)
    w.storage.outcome["READ"] = "HOLD"
    handle = w.deliver(A, 4)
    wait_for(lambda: len(w.storage.held("READ")) == 1)
    (native,) = w.storage.held("READ")
    w.advance(2.0)
    w.settle()
    assert w.events == [] and handle not in w.results
    w.advance(5.0)
    assert not w.wait(handle)[A].success
    assert [e.state for e in w.events] == ["uncertain"]
    assert w.events[0].destination_owner == "framework"
    assert [r.element_index for r in w.events[0].destination_regions] == [4]
    assert native.handle not in w.storage.released
    w.storage.finish(native.handle, "DONE")
    w.poll_until(lambda: len(w.events) == 2)
    assert [e.state for e in w.events] == ["uncertain", "quiesced"]
    assert w.events[0].op_handle == w.events[1].op_handle
    assert w.storage.released.count(native.handle) == 1


# ---- File size checks and replace-after-open ----


@pytest.mark.parametrize("delta", [-1, 1])
def test_g4_file_query_size_mismatch_is_a_miss(g4env, tmp_path, delta):
    w = new_world(g4env, tmp_path, flavor="file")
    w.spill(A, 0)
    os.truncate(_stored_path(w), w.piece + delta)
    assert w.await_check(A) == (QueryStatus.MISS, None)
    assert w.check_names() == [digest(A)]
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
def test_g4_discard_hint_detaches_a_waiting_subscription(g4env, tmp_path, flavor):
    w = new_world(g4env, tmp_path, flavor=flavor)
    w.spill(A, 0)
    before = w.check_calls()
    gate = w.hold_checks()
    assert w.kvcr.query([A], "a") == [fetching_g4()]
    wait_for(lambda: w.blocked_checks() == 1)
    assert w.kvcr.query([A], "b") == [fetching_g4()]
    w.kvcr.discard_hint("a")
    gate.set()
    _present(w, A, "b")
    # The late completion stored nothing for "a": it starts a fresh check.
    assert w.kvcr.query([A], "a") == [fetching_g4()]
    wait_for(lambda: w.check_calls() - before == 2)


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
def test_g4_max_check_results_misses_new_lookups(g4env, tmp_path, flavor):
    w = new_world(g4env, tmp_path, flavor=flavor, max_check_results=2)
    w.hold_checks()
    assert w.kvcr.query([A], "a") == [fetching_g4()]
    assert w.kvcr.query([A], "b") == [fetching_g4()]
    assert w.kvcr.query([A], "c") == [(QueryStatus.MISS, None)]
    w.kvcr.discard_hint("a")
    w.kvcr.discard_hint("b")
    assert w.kvcr.query([A], "c") == [fetching_g4()]


@pytest.mark.parametrize("flavor", FLAVORS)
def test_g4_query_alone_sees_check_results_without_poll_completed(
    g4env, tmp_path, flavor
):
    w = new_world(g4env, tmp_path, flavor=flavor)
    w.spill(A, 0)
    assert w.kvcr.query([A], "r") == [fetching_g4()]
    assert w.query_only(A) == fetchable_g4()


# ---- Check batching ----


@pytest.mark.parametrize("flavor", FLAVORS)
@pytest.mark.parametrize(("batch_max", "sizes"), [(32, [1, 5]), (3, [1, 3, 2])])
def test_g4_queued_checks_share_query_memory_calls(
    g4env, tmp_path, flavor, batch_max, sizes
):
    w = new_world(g4env, tmp_path, flavor=flavor, check_batch_max=batch_max)
    gate = w.hold_checks()
    keys = [BlockKey(f"k{i}".encode()) for i in range(6)]
    assert w.kvcr.query(keys[:1], "r") == [fetching_g4()]
    wait_for(lambda: w.blocked_checks() == 1)
    assert w.kvcr.query(keys[1:], "r") == [fetching_g4()] * 5
    gate.set()
    for key in keys:
        w.await_check(key)
    assert [len(call) for call in w.storage.query_calls] == sizes


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
