# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""G4: unmanaged remote storage (object store or shared file system).

KVCR keeps no G4 index. ``query`` starts an existence check, ``fetch`` and
``deliver`` read one object per key straight into the destination, and a
LOCAL_G2 eviction writes the victim and publishes it atomically.

Threads: every storage-agent call runs on the G4 storage worker, every file
system call on the G4 file worker. Core state changes only on caller threads
under ``_state_lock``; workers report through completion queues that
``poll_completed`` (and ``query``, for checks) drain.

Lifetime rule: destinations, sources, registrations and fds are released only
after the storage worker released the transfer handle at DONE or ERR. This
proves quiescence because OBJ handles carry one descriptor and POSIX uses
linux AIO only.
"""

import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from ..config import G4Options
from ..types import (
    BlockKey,
)

if TYPE_CHECKING:
    from ..runtime.core import _KVCRCore

logger = logging.getLogger(__name__)

_FORMAT_VERSION = 1


class _G4:
    """G4 tier state owned by caller threads under the core state lock."""

    def __init__(self, kvcr: "_KVCRCore", options: G4Options) -> None:
        self._kvcr = kvcr
        self._options = options
        if kvcr._local_dram is None:
            raise ValueError("G4 requires configured local DRAM")
        if options.flavor not in ("object", "file"):
            raise ValueError("G4 flavor must be 'object' or 'file'")
        self._flavor = options.flavor
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
        if self._flavor == "object" and len(layout) != 1:
            raise ValueError("G4 object flavor key_layout must have exactly one label")
        self._layout = layout
        self._sizes = [kvcr._block_sizes[label.partition(":")[0]] for label in layout]
        self._offsets = [sum(self._sizes[:index]) for index in range(len(layout))]
        self._payload_len = sum(self._sizes)
        schema = "|".join(
            [f"g4v{_FORMAT_VERSION}"]
            + [f"{label}={size}" for label, size in zip(layout, self._sizes)]
        )
        self._layout_tag = hashlib.sha256(schema.encode()).hexdigest()[:8]
        if self._flavor == "object":
            if options.root is not None:
                raise ValueError("G4 object flavor does not use a root")
            self._root: Path | None = None
        else:
            if options.root is None:
                raise ValueError("G4 file flavor requires a root directory")
            self._root = Path(options.root)

    # Startup and naming.

    def _name(self, key: BlockKey) -> str:
        digest = hashlib.sha256(bytes(key)).hexdigest()
        name = f"{self._prefix}/{self._layout_tag}/{digest[:2]}/{digest}"
        return name if self._root is None else str(self._root / name)
