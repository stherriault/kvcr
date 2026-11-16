# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""G4 configuration, startup rejections and environment handling."""

import pytest
from _g4_test_utils import (
    g4_options,
    new_world,
)

from kvcr.types import CacheTier


def test_cache_tier_g4_exists():
    assert CacheTier.G4.value == "G4"


# ---- Startup rejections ----


def test_g4_file_flavor_requires_a_root(g4env, tmp_path):
    with pytest.raises(ValueError):
        new_world(g4env, tmp_path, flavor="file", options=g4_options("file", None))


def test_g4_file_key_layout_must_resolve_against_the_pools(g4env, tmp_path):
    with pytest.raises(ValueError):
        new_world(g4env, tmp_path, flavor="file", key_layout=("missing",))


def test_g4_file_root_must_be_a_directory(g4env, tmp_path):
    not_a_directory = tmp_path / "file"
    not_a_directory.write_text("x")
    with pytest.raises((ValueError, OSError, RuntimeError)):
        new_world(g4env, tmp_path, flavor="file", root=not_a_directory)
