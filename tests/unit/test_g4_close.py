# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""close() at every G4 operation stage leaves nothing behind; worker failure."""

import threading

from _g4_test_utils import (
    new_world,
)

# ---- File worker thread exists only for the file flavor ----


def _file_workers():
    return [
        t for t in threading.enumerate() if t.name == "kvcr-g4-file" and t.is_alive()
    ]


def test_g4_file_flavor_runs_one_file_worker_thread_until_close(g4env, tmp_path):
    threads = threading.active_count()
    w = new_world(g4env, tmp_path, flavor="file")
    assert len(_file_workers()) == 1
    w.kvcr.close()
    assert _file_workers() == []
    assert threading.active_count() == threads
