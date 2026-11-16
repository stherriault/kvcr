# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""G4 write side: spill mirroring G3, atomic publish, and the circuit breaker."""

import contextlib
import errno
import os
import re
import threading

import pytest
from _g4_test_utils import (
    PAGE,
    find_files,
    new_world,
    pattern,
    tmp_files,
)
from _kvcr_test_utils import _mem_descriptor

from kvcr.types import (
    BlockKey,
    CacheTier,
    PlacementAction,
    QueryStatus,
)

FLAVORS = ["file"]
A = BlockKey(b"block-a")
B = BlockKey(b"block-b")
C = BlockKey(b"block-c")
MISS = (QueryStatus.MISS, None)


def _capacity_key(w):
    return w.local_dram._pools._capacity_eviction_key


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
