# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""G4 read side: existence checks, fill, deliver, schema, size checks, threads."""

import ctypes

import pytest
from _g4_test_utils import (
    digest,
    fetching_g4,
    new_world,
    wait_for,
)

from kvcr.types import BlockKey, QueryStatus

FLAVORS = ["file"]
A = BlockKey(b"block-a")


# ---- Existence check flow ----


@pytest.mark.parametrize("flavor", FLAVORS)
def test_g4_query_absent_block_is_fetching_then_miss(g4env, tmp_path, flavor):
    w = new_world(g4env, tmp_path, flavor=flavor)
    assert w.kvcr.query([A], "r") == [fetching_g4()]
    assert w.await_check(A) == (QueryStatus.MISS, None)


# ---- One handle per key, OBJ devIds, registrations ----


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


# ---- Check table: sharing, bounds, queueing, caller pump ----


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
