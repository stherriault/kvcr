# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""G4: unmanaged remote storage (shared file system).

KVCR keeps no G4 index. Every block is one file named from its key under a
directory that encodes the ``key_layout`` schema.

Threads: every storage agent call runs on the G4 storage worker, which owns
the storage agent. Core state changes only on caller threads under
``_state_lock``.
"""

import hashlib
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..config import G4Options
from ..runtime import progress as _progress
from ..runtime.nixl_module import load_nixl, nixl_module_name
from ..types import BlockKey

if TYPE_CHECKING:
    from ..runtime.core import _KVCRCore

_FORMAT_VERSION = 1


def _create_agent(g4: "_G4", name: str) -> Any:
    """Create a storage agent with the G4 backend; runs on its owner thread."""
    module_name = nixl_module_name()
    try:
        # Same resolution as the main agent, so one NIXL build is loaded.
        module = load_nixl(module_name)
    except Exception as error:
        raise RuntimeError(f"G4 storage agent: {error}") from error
    try:
        agent = module.nixl_agent(name, module.nixl_agent_config(backends=[]))
    except Exception as error:
        raise RuntimeError(
            f"G4 cannot create a storage agent with NIXL module "
            f"{module_name!r} (KVCR_NIXL_MODULE): {error}"
        ) from error
    backend = g4._backend
    try:
        plugins = agent.get_plugin_list()
        if backend not in plugins:
            raise RuntimeError(f"plugin {backend} is not available")
        agent.create_backend(backend, dict(g4._backend_params))
    except Exception as error:
        raise RuntimeError(
            f"G4 cannot create NIXL backend {backend} with module "
            f"{module_name!r}; KVCR_NIXL_MODULE selects another wheel: {error}"
        ) from error
    return agent


class _StorageWorker:
    """Owns the storage NIXL agent; the only thread that calls it."""

    def __init__(self, g4: "_G4") -> None:
        self._g4 = g4
        self.cond = threading.Condition()
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

    def stop(self) -> None:
        with self.cond:
            self.stopping = True
            self.cond.notify()

    def _setup(self) -> None:
        g4 = self._g4
        agent = self.agent = _create_agent(g4, self.agent_name)
        backend = g4._backend
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
            with self.cond:
                while not self.stopping:
                    self.cond.wait()
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
        if not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in options.backend_options.items()
        ):
            raise TypeError("G4 backend options must map strings to strings")
        params = dict(options.backend_options)
        if options.root is None:
            raise ValueError("G4 file flavor requires a root directory")
        self._backend = options.backend or "POSIX"
        self._root = Path(options.root)
        if not self._root.is_dir():
            raise ValueError(f"G4 root {self._root} is not a directory")
        self._backend_params = params
        self._storage = _StorageWorker(self)

    # Startup and naming.

    def _name(self, key: BlockKey) -> str:
        digest = hashlib.sha256(bytes(key)).hexdigest()
        name = f"{self._prefix}/{self._layout_tag}/{digest[:2]}/{digest}"
        return str(self._root / name)

    def _dram_regions(self):
        return [
            region
            for regions in self._kvcr._memory_regions
            for region in regions.values()
            if region.mem_type == "DRAM"
        ]

    def start(self, timeout: float) -> None:
        self._storage.thread.start()
        if not self._storage.ready.wait(timeout):
            raise RuntimeError(f"G4 storage worker did not start within {timeout:g}s")
        if self._storage.error is not None:
            raise self._storage.error

    # Close.

    def close_workers(self) -> None:
        """Stop the storage worker and join it."""
        storage = self._storage
        storage.stop()
        bound = _progress._OP_CLEANUP_TIMEOUT_SECONDS + _progress._JOIN_TIMEOUT_SECONDS
        if storage.thread.ident is not None:
            storage.thread.join(timeout=bound)
        if storage.thread.is_alive():
            raise RuntimeError("G4 storage worker did not stop")
        if storage.failure is not None:
            raise storage.failure

    def is_quiescent(self) -> bool:
        return not self._storage.thread.is_alive()
