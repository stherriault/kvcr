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
import queue
import threading
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..config import G4Options
from ..runtime import progress as _progress
from ..runtime.nixl_module import load_nixl, nixl_module_name
from ..types import (
    BlockKey,
)

if TYPE_CHECKING:
    from ..runtime.core import _KVCRCore

logger = logging.getLogger(__name__)

_FORMAT_VERSION = 1
_IDLE_WAIT_SECONDS = 0.0005
_WAKE_SECONDS = 0.05
_STOP = object()


class _StorageWorker:
    """Owns the storage NIXL agent; the only thread that calls it."""

    def __init__(self, g4: "_G4") -> None:
        self._g4 = g4
        self.cond = threading.Condition()
        self.jobs: deque[object] = deque()
        self.stopping = False
        self.ready = threading.Event()
        self.error: BaseException | None = None
        self.failure: BaseException | None = None
        self.agent: Any = None
        self.agent_name = f"{g4._kvcr.nixl_agent_name}-g4"
        self._dram_regs: list[Any] = []
        self.thread = threading.Thread(
            target=self._run, daemon=True, name="kvcr-g4-storage"
        )

    def submit(self, job: object) -> None:
        with self.cond:
            self.jobs.append(job)
            self.cond.notify()

    def stop(self) -> None:
        with self.cond:
            self.stopping = True
            self.cond.notify()

    def _setup(self) -> None:
        g4 = self._g4
        module_name = nixl_module_name()
        try:
            # Same resolution as the main agent, so one NIXL build is loaded.
            module = load_nixl(module_name)
        except Exception as error:
            raise RuntimeError(f"G4 storage agent: {error}") from error
        try:
            agent = module.nixl_agent(
                self.agent_name, module.nixl_agent_config(backends=[])
            )
        except Exception as error:
            raise RuntimeError(
                f"G4 cannot create a storage agent with NIXL module "
                f"{module_name!r} (KVCR_NIXL_MODULE): {error}"
            ) from error
        self.agent = agent
        backend = g4._backend
        try:
            plugins = agent.get_plugin_list()
            if backend not in plugins:
                raise RuntimeError(f"plugin {backend} is not available")
            options = dict(g4._backend_params)
            agent.create_backend(backend, options)
        except Exception as error:
            raise RuntimeError(
                f"G4 cannot create NIXL backend {backend} with module "
                f"{module_name!r}; KVCR_NIXL_MODULE selects another wheel: {error}"
            ) from error
        for region in g4._dram_regions():
            extent = region.size + (region.count - 1) * (region.stride or region.size)
            self._dram_regs.append(
                agent.register_memory(
                    [(region.addr, extent, region.device_Id, "")],
                    mem_type="DRAM",
                    backends=[backend],
                )
            )

    def _run(self) -> None:
        try:
            self._setup()
        except BaseException as error:
            self.error = error
            self._deregister_dram()
            self.ready.set()
            return
        self.ready.set()
        try:
            while True:
                with self.cond:
                    if not (self.jobs or self.stopping):
                        self.cond.wait(_WAKE_SECONDS)
                    self.jobs.clear()
                    if self.stopping:
                        break
        except BaseException as error:
            self.failure = error
        try:
            self._deregister_dram()
        except BaseException as error:
            if self.failure is None:
                self.failure = error

    def _deregister_dram(self) -> None:
        agent = self.agent
        regs, self._dram_regs = self._dram_regs, []
        for reg in reversed(regs):
            agent.deregister_memory(reg, backends=[self._g4._backend])


class _FileWorker:
    """Runs every file system call of the file flavor, in queue order."""

    def __init__(self, g4: "_G4") -> None:
        self._g4 = g4
        self.queue: queue.SimpleQueue[object] = queue.SimpleQueue()
        self.stopping = False
        self.thread = threading.Thread(
            target=self._run, daemon=True, name="kvcr-g4-file"
        )

    def stop(self) -> None:
        self.stopping = True
        self.queue.put(_STOP)

    def _run(self) -> None:
        while True:
            job = self.queue.get()
            # Close drops every job still queued.
            if job is _STOP or self.stopping:
                return


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
        if not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in options.backend_options.items()
        ):
            raise TypeError("G4 backend options must map strings to strings")
        params = dict(options.backend_options)
        if self._flavor == "object":
            if options.root is not None:
                raise ValueError("G4 object flavor does not use a root")
            self._backend = options.backend or "OBJ"
            self._root: Path | None = None
        else:
            if options.root is None:
                raise ValueError("G4 file flavor requires a root directory")
            self._backend = options.backend or "POSIX"
            self._root = Path(options.root)
        self._backend_params = params
        self._closing = False
        self._storage = _StorageWorker(self)
        self._file = _FileWorker(self)

    # Startup and naming.

    def _name(self, key: BlockKey) -> str:
        digest = hashlib.sha256(bytes(key)).hexdigest()
        name = f"{self._prefix}/{self._layout_tag}/{digest[:2]}/{digest}"
        return name if self._root is None else str(self._root / name)

    def _dram_regions(self):
        return [
            region
            for regions in self._kvcr._memory_regions
            for region in regions.values()
            if region.mem_type == "DRAM"
        ]

    def start(self, timeout: float) -> None:
        self._storage.thread.start()
        self._file.thread.start()
        if not self._storage.ready.wait(timeout):
            raise RuntimeError(f"G4 storage worker did not start within {timeout:g}s")
        if self._storage.error is not None:
            raise self._storage.error

    # Close.

    def close_workers(self) -> None:
        """Drop unstarted jobs, let started transfers end, and join both workers."""
        self._closing = True
        self._storage.stop()
        self._file.stop()
        timeout = _progress._OP_CLEANUP_TIMEOUT_SECONDS
        for worker, bound in (
            (self._storage, timeout + _progress._JOIN_TIMEOUT_SECONDS),
            (self._file, timeout),
        ):
            if worker.thread.ident is not None:
                worker.thread.join(timeout=bound)
        if self._storage.thread.is_alive():
            raise RuntimeError("G4 storage worker did not stop")
        if self._file.thread.is_alive():
            raise RuntimeError("G4 file worker did not stop")
        if self._storage.failure is not None:
            raise self._storage.failure

    def is_quiescent(self) -> bool:
        return not self._storage.thread.is_alive() and not self._file.thread.is_alive()
