# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""G4 read side: existence checks, fill, deliver, schema, size checks, threads."""

import ctypes

import pytest
from _g4_test_utils import (
    new_world,
)

FLAVORS = ["file"]


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
