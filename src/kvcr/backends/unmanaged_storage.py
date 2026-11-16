# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""G4: unmanaged remote storage (shared file system).

KVCR keeps no G4 index. Every block is one file named from its key under a
directory that encodes the ``key_layout`` schema.
"""

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

from ..config import G4Options
from ..types import BlockKey

if TYPE_CHECKING:
    from ..runtime.core import _KVCRCore

_FORMAT_VERSION = 1


class _G4:
    """G4 tier state owned by caller threads under the core state lock."""

    def __init__(self, kvcr: "_KVCRCore", options: G4Options) -> None:
        self._kvcr = kvcr
        if kvcr._local_dram is None:
            raise ValueError("G4 requires configured local DRAM")
        if options.flavor != "file":
            raise ValueError("G4 flavor must be 'file'")
        prefix = options.key_prefix
        if (
            not isinstance(prefix, str)
            or not prefix.strip("/")
            or any(part in ("", ".", "..") for part in prefix.strip("/").split("/"))
        ):
            raise ValueError("G4 key_prefix must be a non-empty relative path")
        self._prefix = prefix.strip("/")
        layout = tuple(options.key_layout)
        kvcr._validate_expected_layout(list(layout))
        self._layout = layout
        self._sizes = [kvcr._block_sizes[label.partition(":")[0]] for label in layout]
        schema = "|".join(
            [f"g4v{_FORMAT_VERSION}"]
            + [f"{label}={size}" for label, size in zip(layout, self._sizes)]
        )
        self._layout_tag = hashlib.sha256(schema.encode()).hexdigest()[:8]
        if options.root is None:
            raise ValueError("G4 file flavor requires a root directory")
        self._root = Path(options.root)
        if not self._root.is_dir():
            raise ValueError(f"G4 root {self._root} is not a directory")

    # Naming.

    def _name(self, key: BlockKey) -> str:
        digest = hashlib.sha256(bytes(key)).hexdigest()
        name = f"{self._prefix}/{self._layout_tag}/{digest[:2]}/{digest}"
        return str(self._root / name)
