# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""G4: unmanaged remote storage (object store or shared file system).

KVCR keeps no G4 index. ``query`` starts an existence check, ``fetch`` and
``deliver`` read one object per key straight into the destination, and a
LOCAL_G2 eviction writes the victim and publishes it atomically.

Threads: the G4 storage worker owns the one storage agent and makes every
storage call: transfers, existence checks, read-error probes and, for the
file flavor, every file system call. A slow call pauses the others; check
deadlines and the breaker bound what callers wait for. Core state changes
only on caller threads under ``_state_lock``; the worker reports through
completion queues that ``poll_completed`` (and ``query``, for checks) drain.

Lifetime rule: destinations, sources, registrations and fds are released only
after the storage worker released the transfer handle at DONE or ERR. This
proves quiescence because OBJ handles carry one descriptor and POSIX uses
linux AIO only.
"""

import atexit
import contextlib
import hashlib
import logging
import os
import queue
import secrets
import socket
import stat
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from ..config import G4Options
from ..runtime import progress as _progress
from ..runtime.nixl_module import load_nixl, nixl_module_name
from ..runtime.progress import (
    _FAILED_RESULT,
    _SUCCESS_RESULT,
    _Op,
    _OpId,
    _TransferRef,
)
from ..types import (
    BlockKey,
    BlockMeta,
    CacheTier,
    MemoryRef,
    OpEntryResult,
    OpHandle,
    PlacementAction,
    PlacementDecision,
    PlacementFailure,
    QueryStatus,
    TransferError,
)
from .local_dram_pools import _layout_indices, _LocalDramState

if TYPE_CHECKING:
    from ..runtime.core import _KVCRCore

logger = logging.getLogger(__name__)

_FORMAT_VERSION = 1
# OBJ devIds live above every DRAM device id so the two never collide.
_OBJ_DEVID_BASE = 1 << 32
_IDLE_WAIT_SECONDS = 0.0005
_WAKE_SECONDS = 0.05
_COUNTED_BATCHES = 1024
# Last join before close reports a worker as stuck; covers its exit path.
_FINAL_JOIN_SECONDS = 0.1
_FETCHING = (QueryStatus.FETCHING, CacheTier.G4)
_FETCHABLE = (QueryStatus.FETCHABLE, CacheTier.G4)
_MISS = (QueryStatus.MISS, None)
_FALSE_VALUES = ("", "0", "false", "no", "off")


@dataclass(eq=False)
class _CheckJob:
    """One existence check (or a post-ERR probe) for the storage worker."""

    key: BlockKey
    name: str
    probe_op: "_G4Op | None" = None
    file_id: tuple[int, int] | None = None
    started: bool = False
    started_at: float | None = None
    cancelled: bool = False
    epoch: int = 0
    batch: int | None = None
    result: Literal["", "present", "absent", "error"] = ""
    error: str = ""


@dataclass(eq=False)
class _Check:
    job: _CheckJob | None
    subscribers: set[str] = field(default_factory=set)
    # Deadline while the job is queued; a started job's runs from its start.
    deadline: float = 0.0


@dataclass(eq=False)
class _Xfer:
    """One key's transfer; fields are written by whichever stage owns it."""

    op: "_G4Op"
    key: BlockKey
    name: str
    local: list[tuple[int, int, int]]
    remote: list[tuple[int, int]]
    refs: list[_TransferRef]
    tmp: str | None = None
    fd: int | None = None
    fd_closed: bool = False
    file_id: tuple[int, int] | None = None
    reg: Any = None
    handle: Any = None
    # Set by the storage worker once the handle is released.
    released: bool = False
    start_failed: bool = False
    outcome: Literal["", "DONE", "ERR", "ABSENT", "PUBLISH_ERR"] = ""
    error: str = ""
    published: bool = False
    finished: bool = False
    started_at: float | None = None
    recorded: bool = False


@dataclass(eq=False)
class _G4Op(_Op):
    kind: Literal["fill", "deliver", "spill"] = "deliver"
    handle: OpHandle | None = None
    deadline: float = 0.0
    abandon_deadline: float | None = None
    xfers: list[_Xfer] = field(default_factory=list)
    results: dict[BlockKey, OpEntryResult] = field(default_factory=dict)
    probes: int = 0
    epoch: int = 0
    ended: bool = False
    latched: bool = False
    counted: bool = False
    reset: bool = False
    uncertain_refs: list[MemoryRef] | None = None
    lifecycle_handle: OpHandle | None = None

    @property
    def open_xfers(self) -> bool:
        return any(not xfer.finished for xfer in self.xfers)


class _Breaker:
    """Closed, open for ``backoff`` seconds, then half-open with one trial op.

    Check failures and data failures are separate streaks of consecutive
    failures; either one reaching ``threshold`` opens the breaker. Every state
    change is one WARNING. ``epoch`` counts openings: a failure of work
    started before the latest opening neither extends nor reopens it.
    """

    def __init__(self, threshold: int, backoff: float, clock, on_open) -> None:
        self._threshold = threshold
        self._backoff = backoff
        self._clock = clock
        self._on_open = on_open
        self._check_streak = 0
        self._data_streak = 0
        self._open_until: float | None = None
        self._trial: object | None = None
        self._half_logged = False
        self.epoch = 0

    def state(self) -> Literal["closed", "open", "half"]:
        if self._open_until is None:
            return "closed"
        if self._clock() < self._open_until:
            return "open"
        if not self._half_logged:
            self._half_logged = True
            logger.warning(
                "KVCR G4 breaker half-open: the next read or spill is a trial"
            )
        return "half"

    @property
    def failures(self) -> int:
        return max(self._check_streak, self._data_streak)

    def allows_checks(self) -> bool:
        return self.state() != "open"

    def allows_data(self) -> bool:
        state = self.state()
        return state == "closed" or (state == "half" and self._trial is None)

    def admit(self, token: object) -> bool:
        state = self.state()
        if state == "closed":
            return True
        if state == "open":
            return False
        if self._trial is None:
            self._trial = token
        return self._trial is token

    def release(self, token: object) -> None:
        if self._trial is token:
            self._trial = None

    def check_answered(self) -> None:
        """Storage answered a check; that does not decide a half-open trial."""
        self._check_streak = 0

    def data_success(self) -> None:
        state = self.state()
        if state == "open":
            return
        self._data_streak = 0
        if state == "half":
            self._open_until = None
            self._trial = None
            self._check_streak = 0
            logger.warning("KVCR G4 breaker closed: the trial transfer succeeded")

    def check_failure(self, reason: str, epoch: int) -> None:
        if self._counts(epoch):
            self._check_streak += 1
            self._failure(reason)

    def data_failure(self, reason: str, epoch: int) -> None:
        if self._counts(epoch):
            self._data_streak += 1
            self._failure(reason)

    def _counts(self, epoch: int) -> bool:
        state = self.state()
        return state == "closed" or (state == "half" and epoch == self.epoch)

    def _failure(self, reason: str) -> None:
        state = self.state()
        if state == "closed" and self.failures < self._threshold:
            return
        self._open_until = self._clock() + self._backoff
        self._trial = None
        self._half_logged = False
        self.epoch += 1
        if state == "half":
            logger.warning(
                "KVCR G4 breaker open again for %gs, the trial failed: %s",
                self._backoff,
                reason,
            )
        else:
            logger.warning(
                "KVCR G4 breaker open for %gs, failure streak %d: %s",
                self._backoff,
                self.failures,
                reason,
            )
        self._on_open()

    def gauge(self) -> int:
        return {"closed": 0, "open": 1, "half": 2}[self.state()]


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
    """Owns the storage NIXL agent and runs every G4 storage call in turn.

    Transfers, existence checks, read-error probes and every file system call
    of the file flavor run here, so a slow call pauses the others until it
    returns.
    """

    label = "storage"

    def __init__(self, g4: "_G4") -> None:
        self._g4 = g4
        self.cond = threading.Condition()
        self.jobs: deque[object] = deque()
        self.stopping = False
        self.ready = threading.Event()
        self.error: BaseException | None = None
        self.failure: BaseException | None = None
        self.unreleased: set[_Xfer] = set()
        self.agent: Any = None
        self.agent_name = f"{g4._kvcr.nixl_agent_name}-g4"
        self._dram_regs: list[Any] = []
        self._next_devid = _OBJ_DEVID_BASE
        self._next_batch = 0
        # Spill directories made this run; at most 256 per layout.
        self._made_dirs: set[str] = set()
        self.thread = threading.Thread(
            target=self._run, daemon=True, name="kvcr-g4-storage"
        )

    def submit(self, jobs: Collection[object]) -> None:
        with self.cond:
            self.jobs.extend(jobs)
            self.cond.notify()

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
        if g4._flavor == "object":
            self._check_bucket()

    def _check_bucket(self) -> None:
        # One HEAD at startup turns a wrong bucket, endpoint or credential
        # into a start error; absent and present both prove access.
        g4 = self._g4
        name = f"{g4._prefix}/{g4._layout_tag}/.startup-check"
        try:
            answers = list(
                self.agent.query_memory(
                    [(0, g4._payload_len, 0, name)], g4._backend, mem_type="OBJ"
                )
            )
            if len(answers) != 1:
                raise RuntimeError("query_memory answered the wrong number of rows")
        except Exception as error:
            bucket = g4._backend_params.get("bucket")
            raise RuntimeError(
                f"G4 cannot check objects in bucket {bucket!r}: {error}"
            ) from error

    def _run(self) -> None:
        try:
            self._setup()
        except BaseException as error:
            self.error = error
            self._deregister_dram()
            self.ready.set()
            return
        self.ready.set()
        g4 = self._g4
        active: list[_Xfer] = []
        checks: deque[_CheckJob] = deque()
        probes: deque[_CheckJob] = deque()
        close_deadline: float | None = None
        try:
            while True:
                with self.cond:
                    if not (self.jobs or active or checks or probes or self.stopping):
                        self.cond.wait(_WAKE_SECONDS)
                    jobs = list(self.jobs)
                    self.jobs.clear()
                    stopping = self.stopping
                if stopping:
                    # Close drops every job that has not started.
                    for job in (*jobs, *checks, *probes):
                        if isinstance(job, _CheckJob):
                            g4._try_cancel(job)
                    jobs = []
                    checks.clear()
                    probes.clear()
                    if close_deadline is None:
                        close_deadline = (
                            time.monotonic() + _progress._OP_CLEANUP_TIMEOUT_SECONDS
                        )
                for job in jobs:
                    if isinstance(job, _CheckJob):
                        (probes if job.probe_op is not None else checks).append(job)
                    else:
                        self._start(job, active)
                worked = self._poll(active)
                if stopping:
                    if not active:
                        break
                    if time.monotonic() >= close_deadline:
                        raise RuntimeError("G4 storage transfers did not finish")
                    time.sleep(_IDLE_WAIT_SECONDS)
                    continue
                if probes:
                    self._probe(probes.popleft())
                elif checks:
                    self._check_batch(checks)
                elif active and not worked:
                    time.sleep(_IDLE_WAIT_SECONDS)
        except BaseException as error:
            self.failure = error
        if not self.unreleased:
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

    # Transfers.

    def _start(self, xfer: _Xfer, active: list[_Xfer]) -> None:
        g4 = self._g4
        if g4._flavor == "file":
            try:
                if not self._open(xfer):
                    return
            except Exception as error:
                logger.exception("KVCR G4 block file job failed")
                self._fail(xfer, error)
                return
        agent = self.agent
        backend = g4._backend
        try:
            if g4._flavor == "object":
                dev = self._next_devid
                self._next_devid += 1
                kind = "OBJ"
                reg_rows = [(0, g4._payload_len, dev, xfer.name)]
            else:
                dev = xfer.fd
                kind = "FILE"
                reg_rows = [(0, g4._payload_len, dev, "")]
            xfer.reg = agent.register_memory(
                reg_rows, mem_type=kind, backends=[backend]
            )
            local = agent.get_xfer_descs(xfer.local, mem_type="DRAM")
            remote = agent.get_xfer_descs(
                [(offset, size, dev) for offset, size in xfer.remote], mem_type=kind
            )
            xfer.handle = agent.initialize_xfer(
                "WRITE" if xfer.op.kind == "spill" else "READ",
                local,
                remote,
                self.agent_name,
                backends=[backend],
            )
            if xfer.handle is None:
                raise RuntimeError("NIXL transfer creation returned None")
        except Exception:
            logger.warning("KVCR G4 transfer could not start", exc_info=True)
            self._deregister(xfer)
            xfer.released = True
            xfer.start_failed = True
            xfer.outcome = "ERR"
            self._ended(xfer)
            return
        self.unreleased.add(xfer)
        active.append(xfer)
        try:
            agent.transfer(xfer.handle)
        except Exception as error:
            # The handle's own state decides; it is released only when terminal.
            logger.debug("KVCR G4 transfer post raised: %s", error)

    def _deregister(self, xfer: _Xfer) -> None:
        reg, xfer.reg = xfer.reg, None
        if reg is None:
            return
        try:
            self.agent.deregister_memory(reg, backends=[self._g4._backend])
        except Exception:
            logger.warning("KVCR G4 deregistration failed", exc_info=True)

    def _poll(self, active: list[_Xfer]) -> bool:
        worked = False
        agent = self.agent
        for xfer in tuple(active):
            try:
                state = agent.check_xfer_state(xfer.handle)
            except Exception as error:
                # NIXL raises for a failed transfer instead of answering ERR.
                logger.warning("KVCR G4 transfer failed: %s", error)
                xfer.error = str(error)
                state = "ERR"
            if state in ("PROC", "PEND"):
                continue
            try:
                agent.release_xfer_handle(xfer.handle)
            except Exception:
                logger.warning("KVCR G4 handle release failed", exc_info=True)
                continue
            active.remove(xfer)
            self.unreleased.discard(xfer)
            xfer.released = True
            self._deregister(xfer)
            xfer.outcome = "DONE" if state == "DONE" else "ERR"
            self._ended(xfer)
            worked = True
        return worked

    def _ended(self, xfer: _Xfer) -> None:
        """The handle is released or never existed; close the file side."""
        if self._g4._flavor != "file":
            self._g4._op_done.put(xfer)
            return
        try:
            self._finish(xfer)
        except Exception as error:
            logger.exception("KVCR G4 block file job failed")
            self._fail(xfer, error)

    # File system calls of the file flavor.

    def _open(self, xfer: _Xfer) -> bool:
        """Open the block file; False when the transfer already ended."""
        g4 = self._g4
        direct = os.O_DIRECT if g4._options.direct_io else 0
        if xfer.op.kind == "spill":
            try:
                xfer.fd = self._open_temp(xfer.tmp, direct)
                os.ftruncate(xfer.fd, g4._payload_len)
            except OSError:
                logger.warning("KVCR G4 spill file could not open", exc_info=True)
                xfer.start_failed = True
                xfer.outcome = "ERR"
                self._finish(xfer)
                return False
            return True
        try:
            fd = os.open(xfer.name, os.O_RDONLY | os.O_CLOEXEC | direct)
        except FileNotFoundError:
            xfer.outcome = "ABSENT"
            g4._op_done.put(xfer)
            return False
        except OSError:
            logger.warning("KVCR G4 block file could not open", exc_info=True)
            xfer.outcome = "ERR"
            g4._op_done.put(xfer)
            return False
        xfer.fd = fd
        try:
            info = os.fstat(fd)
        except OSError:
            xfer.outcome = "ERR"
            self._finish(xfer)
            return False
        xfer.file_id = (info.st_dev, info.st_ino)
        if info.st_size != g4._payload_len:
            xfer.outcome = "ABSENT"
            self._finish(xfer)
            return False
        return True

    def _open_temp(self, tmp: str, direct: int) -> int:
        """Create a spill's temp file, making its directory once per run."""
        directory = os.path.dirname(tmp)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | direct
        if directory in self._made_dirs:
            try:
                return os.open(tmp, flags, 0o644)
            except OSError as error:
                # Storage-side cleanup may have removed the directory; NFS can
                # answer that with ESTALE, so the next spill makes it again.
                self._made_dirs.discard(directory)
                if not isinstance(error, FileNotFoundError):
                    raise
        self._g4._make_dirs(directory)
        fd = os.open(tmp, flags, 0o644)
        self._made_dirs.add(directory)
        return fd

    def _close_fd(self, xfer: _Xfer) -> None:
        if xfer.fd is not None and not xfer.fd_closed:
            xfer.fd_closed = True
            os.close(xfer.fd)

    def _finish(self, xfer: _Xfer) -> None:
        if xfer.op.kind == "spill" and xfer.tmp is not None:
            if xfer.outcome == "DONE":
                try:
                    os.fsync(xfer.fd)
                    self._close_fd(xfer)
                    os.rename(xfer.tmp, xfer.name)
                    xfer.published = True
                except OSError:
                    logger.warning("KVCR G4 spill publish failed", exc_info=True)
                    xfer.outcome = "PUBLISH_ERR"
            if not xfer.published:
                with contextlib.suppress(OSError):
                    self._close_fd(xfer)
                with contextlib.suppress(OSError):
                    os.unlink(xfer.tmp)
        else:
            try:
                self._close_fd(xfer)
            except OSError:
                logger.warning("KVCR G4 block file close failed", exc_info=True)
        self._g4._op_done.put(xfer)

    def _fail(self, xfer: _Xfer, error: Exception) -> None:
        """End a transfer whose file call raised, so its operation still completes."""
        with contextlib.suppress(OSError):
            self._close_fd(xfer)
        if xfer.tmp is not None and not xfer.published:
            with contextlib.suppress(OSError):
                os.unlink(xfer.tmp)
        if xfer.handle is None:
            xfer.start_failed = True
            xfer.outcome = "ERR"
        elif xfer.op.kind == "spill" and not xfer.published:
            xfer.outcome = "PUBLISH_ERR" if xfer.outcome == "DONE" else "ERR"
        xfer.error = str(error)
        self._g4._op_done.put(xfer)

    # Existence checks and read-error probes.

    def _check_batch(self, checks: deque[_CheckJob]) -> None:
        batch: list[_CheckJob] = []
        while checks and len(batch) < self._g4._options.check_batch_max:
            job = checks.popleft()
            if self._g4._try_start(job):
                batch.append(job)
        if batch:
            self._query(batch)

    def _probe(self, job: _CheckJob) -> None:
        if not self._g4._try_start(job):
            return
        if job.file_id is None:
            self._query((job,))
            return
        try:
            self._restat(job)
        except Exception as error:
            logger.exception("KVCR G4 block file re-check failed")
            job.result = "error"
            job.error = str(error)
        self._g4._check_done.put(job)

    def _restat(self, job: _CheckJob) -> None:
        """Answer whether the file a read used is still the one in place."""
        try:
            info = os.stat(job.name)
        except FileNotFoundError:
            job.result = "absent"
        except OSError as error:
            job.result = "error"
            job.error = str(error)
        else:
            same = (info.st_dev, info.st_ino) == job.file_id
            job.result = "present" if same else "absent"

    def _query(self, jobs: Collection[_CheckJob]) -> None:
        g4 = self._g4
        self._next_batch += 1
        for job in jobs:
            job.batch = self._next_batch
        if g4._flavor == "file":
            kind = "FILE"
            rows = [(0, 0, 0, job.name) for job in jobs]
        else:
            kind = "OBJ"
            rows = [(0, g4._payload_len, 0, job.name) for job in jobs]
        try:
            answers = list(self.agent.query_memory(rows, g4._backend, mem_type=kind))
            if len(answers) != len(rows):
                raise RuntimeError("query_memory answered the wrong number of rows")
            results = [self._result(answer) for answer in answers]
            error = "query_memory answered a file row without a usable size and mode"
        except Exception as caught:
            logger.debug("KVCR G4 existence check failed", exc_info=True)
            results = ["error"] * len(rows)
            error = str(caught)
        for job, result in zip(jobs, results, strict=True):
            job.result = result
            job.error = error if result == "error" else ""
            g4._check_done.put(job)

    def _result(self, answer: Any) -> str:
        if answer is None:
            return "absent"
        if self._g4._flavor == "object":
            return "present"
        # The POSIX query answers stat fields as decimal strings; anything
        # else means the query itself is broken, so it counts as an error.
        try:
            size = int(answer["size"])
            mode = int(answer["mode"])
        except (KeyError, TypeError, ValueError):
            return "error"
        if size == self._g4._payload_len and stat.S_ISREG(mode):
            return "present"
        return "absent"


class _G4:
    """G4 tier state owned by caller threads under the core state lock."""

    def __init__(self, kvcr: "_KVCRCore", options: G4Options) -> None:
        self._kvcr = kvcr
        self._options = options
        config = kvcr.config
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
        for name in (
            "max_inflight_checks",
            "max_check_results",
            "check_batch_max",
            "failure_threshold",
        ):
            if type(getattr(options, name)) is not int or getattr(options, name) < 1:
                raise ValueError(f"G4 {name} must be a positive integer")
        for name in ("check_timeout_ms", "spill_timeout_ms"):
            value = getattr(options, name)
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"G4 {name} must be a positive integer or None")
        for name in ("backoff_ms", "close_timeout_ms"):
            value = getattr(options, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"G4 {name} must be a non-negative integer")
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
            for name in ("use_uring", "use_posix_aio"):
                if params.get(name, "").lower() not in _FALSE_VALUES:
                    raise ValueError(f"G4 file flavor rejects POSIX {name}")
            if params.get("use_aio", "true").lower() in _FALSE_VALUES:
                raise ValueError("G4 file flavor requires POSIX use_aio")
            params["use_aio"] = "true"
            self._root = Path(options.root)
            if not self._root.is_dir():
                raise ValueError(f"G4 root {self._root} is not a directory")
            self._probe_root()
        self._backend_params = params
        self._check_timeout = (
            options.check_timeout_ms or config.operation_timeout_ms
        ) / 1000
        self._spill_timeout = (
            options.spill_timeout_ms or config.operation_timeout_ms
        ) / 1000
        self._abandon_grace = (
            config.abandon_timeout_ms - config.operation_timeout_ms
        ) / 1000

        self._breaker = _Breaker(
            options.failure_threshold,
            options.backoff_ms / 1000,
            lambda: self._kvcr._clock(),
            self._on_breaker_open,
        )
        self._job_lock = threading.Lock()
        self._check_done: queue.SimpleQueue[_CheckJob] = queue.SimpleQueue()
        self._op_done: queue.SimpleQueue[_Xfer] = queue.SimpleQueue()
        self._checks: dict[BlockKey, _Check] = {}
        self._pending_start: deque[BlockKey] = deque()
        self._outbox: list[_CheckJob] = []
        self._timed_out: dict[BlockKey, _CheckJob] = {}
        self._probes: set[_CheckJob] = set()
        self._pending_count = 0
        self._inflight = 0
        self._sub_count = 0
        self._results: OrderedDict[tuple[str, BlockKey], bool] = OrderedDict()
        self._owner_keys: dict[str, set[BlockKey]] = {}
        self._counted_batches: OrderedDict[int, None] = OrderedDict()
        self._ops: dict[_OpId, _G4Op] = {}
        self._pending_spills: set[BlockKey] = set()
        self._closing = False
        # Set when the storage worker dies: G4 takes no new work.
        self._off = False
        self.spill_late = 0
        self.spill_dropped = 0
        self._counters = dict.fromkeys(
            (
                "g4_check_present",
                "g4_check_absent",
                "g4_check_error",
                "g4_fill_ok",
                "g4_fill_failed",
                "g4_deliver_ok",
                "g4_deliver_failed",
                "g4_spill_published",
            ),
            0,
        )
        self._storage = _StorageWorker(self)

    # Startup and naming.

    def _probe_root(self) -> None:
        directory = self._root / self._prefix / self._layout_tag
        directory.mkdir(parents=True, exist_ok=True)
        tmp = directory / f".probe.tmp.{self._temp_suffix()}"
        final = directory / f".probe.{self._temp_suffix()}"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o644)
        try:
            try:
                os.write(fd, b"kvcr-g4")
                os.fsync(fd)
            finally:
                os.close(fd)
            os.rename(tmp, final)
            os.unlink(final)
        finally:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
        if self._options.direct_io:
            self._probe_direct_io(directory)

    def _probe_direct_io(self, directory: Path) -> None:
        # Opening a directory with O_DIRECT succeeds on file systems that
        # refuse it for files, so the check opens a file.
        path = directory / f".probe.direct.{self._temp_suffix()}"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_DIRECT
        try:
            os.close(os.open(path, flags, 0o644))
            os.close(os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECT))
        except OSError as error:
            raise ValueError(
                f"G4 root {self._root} does not accept O_DIRECT ({error}); "
                "set direct_io=False to use buffered I/O"
            ) from error
        finally:
            with contextlib.suppress(OSError):
                os.unlink(path)

    @staticmethod
    def _temp_suffix() -> str:
        return f"{socket.gethostname()}.{os.getpid()}.{secrets.randbits(64):016x}"

    def _name(self, key: BlockKey) -> str:
        digest = hashlib.sha256(bytes(key)).hexdigest()
        name = f"{self._prefix}/{self._layout_tag}/{digest[:2]}/{digest}"
        return name if self._root is None else str(self._root / name)

    def _make_dirs(self, path: str) -> None:
        try:
            os.mkdir(path)
        except FileExistsError:
            return
        except FileNotFoundError:
            self._make_dirs(os.path.dirname(path))
            with contextlib.suppress(FileExistsError):
                os.mkdir(path)

    def _dram_regions(self):
        return [
            region
            for regions in self._kvcr._memory_regions
            for region in regions.values()
            if region.mem_type == "DRAM"
        ]

    def check_dram_alignment(self) -> None:
        """Refuse DRAM regions that O_DIRECT transfers cannot use."""
        if self._flavor != "file" or not self._options.direct_io:
            return
        page = os.sysconf("SC_PAGE_SIZE")
        for region in self._dram_regions():
            fields = {"address": region.addr, "size": region.size}
            if region.stride:
                fields["stride"] = region.stride
            bad = [name for name, value in fields.items() if value % page]
            if bad:
                raise ValueError(
                    f"G4 direct_io needs page aligned DRAM regions: region "
                    f"{region.label!r} at {region.addr:#x} has {' and '.join(bad)} "
                    f"not a multiple of {page}; align it or set direct_io=False"
                )

    def start(self, timeout: float) -> None:
        worker = self._storage
        worker.thread.start()
        if not worker.ready.wait(timeout):
            raise RuntimeError(f"G4 storage worker did not start within {timeout:g}s")
        if worker.error is not None:
            raise worker.error

    # Worker hand-offs.

    def _try_start(self, job: _CheckJob) -> bool:
        with self._job_lock:
            if job.cancelled:
                return False
            job.started_at = self._kvcr._clock()
            job.started = True
            return True

    def _try_cancel(self, job: _CheckJob) -> bool:
        with self._job_lock:
            if job.started:
                return False
            job.cancelled = True
            return True

    def _launch(self, xfer: _Xfer) -> None:
        xfer.started_at = self._kvcr._timer()
        self._storage.submit((xfer,))

    # Existence checks.

    def refresh(self) -> None:
        """Apply finished checks and check deadlines; start queued checks."""
        if self._off:
            return
        while True:
            try:
                job = self._check_done.get_nowait()
            except queue.Empty:
                break
            self._apply_check(job)
        now = self._kvcr._clock()
        for key, check in tuple(self._checks.items()):
            job = check.job
            if self._checks.get(key) is not check or job is None:
                continue
            if job.started_at is None:
                if now < check.deadline:
                    continue
                if self._try_cancel(job):
                    # Queue wait says nothing about storage: MISS, no count.
                    self._resolve_check(key, check, False)
                    continue
            if now < job.started_at + self._check_timeout:
                continue
            self._resolve_check(key, check, False)
            self._counters["g4_check_error"] += 1
            # The job still runs; until its late answer is dropped the key
            # answers MISS instead of queueing behind it.
            self._timed_out[key] = job
            self._count_check_failure(
                job, f"existence check timed out after {self._check_timeout:g}s"
            )
        self._start_pending(now)
        self.submit_checks()

    def _apply_check(self, job: _CheckJob) -> None:
        op = job.probe_op
        if op is not None:
            self._probes.discard(job)
            op.probes -= 1
            if job.result == "absent":
                # Storage answered and the block is gone: not a failure.
                self._breaker.check_answered()
            elif job.result == "present":
                self._count_failure(op, "read failed on a block storage still has")
            elif not op.counted:
                op.counted = True
                self._breaker.check_failure(job.error, op.epoch)
            self._end_op_if_done(op)
            return
        if self._timed_out.get(job.key) is job:
            # Counted at its deadline; the late answer changes nothing.
            del self._timed_out[job.key]
            return
        check = self._checks.get(job.key)
        if check is None or check.job is not job:
            return
        self._resolve_check(job.key, check, job.result == "present")
        self._counters[f"g4_check_{job.result}"] += 1
        if job.result == "error":
            self._count_check_failure(job, job.error)
        else:
            self._breaker.check_answered()

    def _resolve_check(self, key: BlockKey, check: _Check, present: bool) -> None:
        del self._checks[key]
        if check.job is not None:
            self._inflight -= 1
        else:
            self._pending_count -= 1
        for owner in check.subscribers:
            self._sub_count -= 1
            self._results[(owner, key)] = present

    def _count_check_failure(self, job: _CheckJob, reason: str) -> None:
        batch = job.batch
        if batch is not None:
            if batch in self._counted_batches:
                return
            self._counted_batches[batch] = None
            if len(self._counted_batches) > _COUNTED_BATCHES:
                self._counted_batches.popitem(last=False)
        self._breaker.check_failure(reason, job.epoch)

    def _on_breaker_open(self) -> None:
        # Checks that have not reached storage are dropped without I/O.
        for key, check in tuple(self._checks.items()):
            if self._checks.get(key) is not check:
                continue
            if check.job is None or self._try_cancel(check.job):
                self._resolve_check(key, check, False)
        # Timed-out checks still queued behind a hung call would each block the
        # worker again while storage is down.
        for key, job in tuple(self._timed_out.items()):
            if self._try_cancel(job):
                del self._timed_out[key]
        # Probes not started yet would only confirm the outage.
        for job in tuple(self._probes):
            if self._try_cancel(job):
                self._probes.discard(job)
                job.probe_op.probes -= 1
                self._end_op_if_done(job.probe_op)

    def _start_pending(self, now: float) -> None:
        while (
            self._pending_start
            and self._inflight < self._options.max_inflight_checks
            and not self._closing
            and self._breaker.allows_checks()
        ):
            key = self._pending_start.popleft()
            check = self._checks.get(key)
            if check is not None and check.job is None:
                self._pending_count -= 1
                self._start_check_job(key, check, now)

    def _start_check_job(self, key: BlockKey, check: _Check, now: float) -> None:
        job = _CheckJob(key, self._name(key), epoch=self._breaker.epoch)
        check.job = job
        check.deadline = now + self._check_timeout
        self._inflight += 1
        self._outbox.append(job)

    def submit_checks(self) -> None:
        """Hand the checks started since the last call to the worker at once."""
        jobs, self._outbox = self._outbox, []
        if not jobs:
            return
        # One hand-off, so the worker batches them into full calls.
        self._storage.submit(jobs)

    def lookup(
        self, key: BlockKey, owner: str | None
    ) -> tuple[QueryStatus, CacheTier | None]:
        if owner is None or self._off:
            return _MISS
        present = self._results.get((owner, key))
        if present is not None:
            if present:
                return _FETCHABLE
            self._pop_result(key, owner)
            return _MISS
        check = self._checks.get(key)
        if check is not None and owner in check.subscribers:
            return _FETCHING
        if self._closing or key in self._timed_out or not self._breaker.allows_checks():
            return _MISS
        held = self._sub_count + self._pending_count + len(self._results)
        if held >= self._options.max_check_results:
            return _MISS
        if check is None:
            check = self._checks[key] = _Check(None)
            if self._inflight < self._options.max_inflight_checks:
                self._start_check_job(key, check, self._kvcr._clock())
            else:
                self._pending_count += 1
                self._pending_start.append(key)
        check.subscribers.add(owner)
        self._sub_count += 1
        self._owner_keys.setdefault(owner, set()).add(key)
        return _FETCHING

    def start_check(self, keys: Collection[BlockKey], owner: str) -> None:
        """Start or join checks for ``owner``; results are read through query."""
        with self._kvcr._state_lock:
            self.refresh()
            for key in keys:
                self.lookup(key, owner)
            self.submit_checks()

    def _pop_result(self, key: BlockKey, owner: str) -> bool | None:
        present = self._results.pop((owner, key), None)
        check = self._checks.get(key)
        if check is None or owner not in check.subscribers:
            keys = self._owner_keys.get(owner)
            if keys is not None:
                keys.discard(key)
                if not keys:
                    del self._owner_keys[owner]
        return present

    def discard_owner(self, owner: str) -> None:
        for key in self._owner_keys.pop(owner, ()):
            self._results.pop((owner, key), None)
            check = self._checks.get(key)
            if check is not None and owner in check.subscribers:
                check.subscribers.discard(owner)
                self._sub_count -= 1
                if not check.subscribers and check.job is None:
                    del self._checks[key]
                    self._pending_count -= 1

    # Reads.

    def take_read_source(
        self, key: BlockKey, owner: str | None, labels: list[str]
    ) -> bool:
        """Consume ``owner``'s present result when ``labels`` map to the schema."""
        if owner is None or self._off or not self._pop_result(key, owner):
            return False
        return _layout_indices(self._layout, labels) is not None

    def take_fill_source(
        self, key: BlockKey, owner: str | None, labels: list[str]
    ) -> bool:
        return self.take_read_source(key, owner, labels) and self._breaker.allows_data()

    def _make_read(
        self, op: _G4Op, key: BlockKey, refs: list[_TransferRef]
    ) -> _Xfer | None:
        indices = _layout_indices(self._layout, [ref.label for ref in refs])
        if indices is None:
            return None
        local = []
        for ref in refs:
            descriptor = self._kvcr._address_descriptor(ref)
            if descriptor.mem_type != "DRAM":
                return None
            local.append((descriptor.addr, descriptor.size, descriptor.device_Id))
        remote = [(self._offsets[index], self._sizes[index]) for index in indices]
        return _Xfer(op, key, self._name(key), local, remote, list(refs))

    def _start_reads(
        self,
        op: _G4Op,
        blocks: Mapping[BlockKey, list[_TransferRef]],
    ) -> list[BlockKey]:
        rejected = []
        for key, refs in blocks.items():
            xfer = None if self._closing else self._make_read(op, key, refs)
            if xfer is None or not self._breaker.admit(op):
                rejected.append(key)
                continue
            op.xfers.append(xfer)
        if op.xfers:
            self._ops[op.op_id] = op
            for xfer in op.xfers:
                self._launch(xfer)
        else:
            self._breaker.release(op)
        return rejected

    def start_fill(
        self,
        op_handle: OpHandle,
        blocks: Mapping[BlockKey, list[_TransferRef]],
        deadline: float,
    ) -> bool:
        op = _G4Op(
            op_id=("g4_fill", op_handle),
            keys=frozenset(blocks),
            kind="fill",
            handle=op_handle,
            deadline=deadline,
            epoch=self._breaker.epoch,
        )
        rejected = self._start_reads(op, blocks)
        if rejected:
            self._kvcr._local_dram.complete_fill(
                rejected, success=False, source=CacheTier.G4
            )
        return True

    def start_deliver(
        self,
        op_handle: OpHandle,
        blocks: Mapping[BlockKey, list[_TransferRef]],
        deadline: float,
    ) -> None:
        op = _G4Op(
            op_id=("g4_deliver", op_handle),
            keys=frozenset(blocks),
            kind="deliver",
            handle=op_handle,
            deadline=deadline,
            epoch=self._breaker.epoch,
            abandon_deadline=deadline + self._abandon_grace,
        )
        for key in self._start_reads(op, blocks):
            op.results[key] = _FAILED_RESULT
        if not op.xfers:
            self._kvcr._complete(op_handle, op.results)

    # Spills.

    def resolve_eviction(
        self,
        meta: BlockMeta,
        source: CacheTier,
        decision: PlacementDecision,
        deadline: float,
    ) -> tuple[PlacementDecision, bool]:
        """Start a LOCAL_G2 -> G4 spill; the deposit deadline does not bound it."""
        if source is not CacheTier.LOCAL_G2 or decision != (
            PlacementAction.MOVE_TO,
            CacheTier.G4,
        ):
            return decision, False
        core = self._kvcr
        key = meta.block_key
        record = core._block_record_map.get(key)
        residency = record.local_dram if record is not None else None
        if residency is None or key in self._pending_spills:
            return (PlacementAction.KEEP, None), False
        complete = (
            residency.state is _LocalDramState.READY
            and not residency.pending_labels
            and residency.key_layout == self._layout
            and residency.layout == self._layout
        )
        op_id = ("g4_spill", key)
        op = _G4Op(
            op_id=op_id,
            keys=frozenset((key,)),
            kind="spill",
            deadline=core._clock() + self._spill_timeout,
            epoch=self._breaker.epoch,
        )
        if self._closing or self._off or not complete or not self._breaker.admit(op):
            self.spill_dropped += 1
            return (PlacementAction.DROP, None), False
        sources = core._claim_local_dram_sources(op_id, (key,))
        if key not in sources:
            self._breaker.release(op)
            return (PlacementAction.KEEP, None), False
        refs = list(
            core._local_dram._pools.source_refs(
                sources[key], range(len(sources[key].slots))
            )
        )
        local = []
        for ref in refs:
            descriptor = core._address_descriptor(ref)
            local.append((descriptor.addr, descriptor.size, descriptor.device_Id))
        name = self._name(key)
        xfer = _Xfer(
            op,
            key,
            name,
            local,
            list(zip(self._offsets, self._sizes)),
            refs,
            tmp=f"{name}.tmp.{self._temp_suffix()}" if self._root is not None else None,
        )
        op.xfers.append(xfer)
        self._ops[op_id] = op
        self._pending_spills.add(key)
        core._add_block_dependencies(op, new_operation=True)
        self._launch(xfer)
        return (PlacementAction.KEEP, None), True

    def _retire_source(self, op: _G4Op) -> None:
        core = self._kvcr
        sources = core._local_dram_sources_by_op.get(op.op_id)
        if sources:
            core._local_dram._pools.retire_sources(sources)
            core._release_local_dram_sources(op.op_id)

    def _recover_spill_failure(self, op: _G4Op, key: BlockKey, reason: str) -> None:
        core = self._kvcr
        record = core._block_record_map.get(key)
        if record is not None and record.local_dram is not None:
            core._policy.decide_recovery(
                core._block_meta(
                    key,
                    record,
                    core._local_dram._pools._size_bytes(record.local_dram),
                ),
                PlacementFailure(
                    attempted=(PlacementAction.MOVE_TO, CacheTier.G4),
                    source=CacheTier.LOCAL_G2,
                    reason=reason,
                    failure_count=1,
                ),
            )
        self._retire_source(op)

    # Completion pump.

    def poll_main(self) -> None:
        if self._off:
            return
        if not self._closing and self._storage.failure is not None:
            self._disable()
            return
        self.refresh()
        while True:
            try:
                xfer = self._op_done.get_nowait()
            except queue.Empty:
                break
            self._finish_xfer(xfer)
        now = self._kvcr._clock()
        for op in tuple(self._ops.values()):
            if op.ended:
                continue
            if not op.latched and now >= op.deadline:
                op.latched = True
                self._count_failure(op, f"{op.kind} passed its deadline")
                if op.kind == "spill":
                    # The write may still land; reads and G2 admission stop
                    # waiting for it. Its source stays claimed until it ends.
                    self._breaker.release(op)
                    for key in op.keys:
                        self._kvcr._local_dram._pools.abandon_capacity_eviction(key)
            if (
                op.kind == "deliver"
                and op.latched
                and op.uncertain_refs is None
                and now >= op.abandon_deadline
            ):
                self._report_uncertain(op)

    def _clear_checks(self) -> None:
        self._checks.clear()
        self._timed_out.clear()
        self._pending_start.clear()
        self._outbox.clear()
        self._results.clear()
        self._owner_keys.clear()
        self._inflight = self._pending_count = self._sub_count = 0

    def _disable(self) -> None:
        """The storage worker died: fail every open G4 op, turn G4 off.

        Memory a posted, unreleased handle may still touch stays pinned until
        close; memory of transfers never posted is freed.
        """
        self._off = True
        storage = self._storage
        logger.warning(
            "KVCR G4 is off until restart: its storage worker failed: %r",
            storage.failure,
        )
        # The worker stops changing this set once it records its failure.
        unreleased = set(storage.unreleased)
        local = self._kvcr._local_dram
        for op in tuple(self._ops.values()):
            self._ops.pop(op.op_id)
            self._breaker.release(op)
            if op.ended:
                continue
            op.ended = True
            open_xfers = [xfer for xfer in op.xfers if not xfer.finished]
            for xfer in open_xfers:
                self._record(xfer, False)
            held = [xfer for xfer in open_xfers if xfer in unreleased]
            for xfer in op.xfers:
                if xfer not in held:
                    xfer.finished = True
            if op.kind == "spill":
                if held:
                    # Source claim, capacity reservation and dependencies stay.
                    continue
                self._pending_spills.difference_update(op.keys)
                reason = "storage worker failed"
                for key in op.keys:
                    self._recover_spill_failure(op, key, reason)
                self._kvcr._remove_block_dependencies(op)
            elif op.kind == "fill":
                for xfer in held:
                    # An extra claim keeps the slot from ever being reused.
                    record = self._kvcr._block_record_map[xfer.key]
                    local._pools._acquire_claim(xfer.key, record.local_dram)
                keys = [xfer.key for xfer in open_xfers]
                if keys:
                    local.complete_fill(keys, success=False, source=CacheTier.G4)
            elif op.uncertain_refs is None:
                if held:
                    # Storage may still write the destination: report it uncertain.
                    self._report_uncertain(op)
                else:
                    for xfer in open_xfers:
                        op.results[xfer.key] = _FAILED_RESULT
                    self._kvcr._complete(op.handle, op.results)
            for xfer in held:
                xfer.finished = True
        self._clear_checks()
        self._probes.clear()

    def _finish_xfer(self, xfer: _Xfer) -> None:
        op = xfer.op
        if xfer.finished or self._ops.get(op.op_id) is not op:
            return
        xfer.finished = True
        outcome = xfer.outcome
        if op.kind == "spill":
            self._finish_spill(op, xfer)
            return
        done = outcome == "DONE" and not op.latched
        self._record(xfer, done)
        # Recorded first: opening the breaker below can end this op.
        if op.kind == "fill":
            self._kvcr._local_dram.complete_fill(
                (xfer.key,), success=done, source=CacheTier.G4
            )
        elif op.uncertain_refs is None:
            op.results[xfer.key] = _SUCCESS_RESULT if done else _FAILED_RESULT
        if outcome == "DONE":
            if not op.latched:
                self._count_success(op)
        elif outcome == "ERR":
            # While the breaker is not closed a probe would only add load.
            probe = (
                not xfer.start_failed
                and not op.counted
                and not self._off
                and self._breaker.state() == "closed"
            )
            if probe and (self._flavor == "object" or xfer.file_id is not None):
                # A probe decides whether the block vanished or storage failed:
                # a query on OBJ, a re-stat of the file the read used on files.
                job = _CheckJob(xfer.key, xfer.name, probe_op=op, file_id=xfer.file_id)
                op.probes += 1
                self._probes.add(job)
                self._storage.submit((job,))
            else:
                self._count_failure(op, xfer.error or "read failed")
        self._end_op_if_done(op)

    def _end_op_if_done(self, op: _G4Op) -> None:
        if op.open_xfers or self._ops.get(op.op_id) is not op:
            return
        if not op.ended:
            op.ended = True
            if op.kind == "deliver":
                if op.uncertain_refs is not None:
                    self._report_memory(op, "quiesced")
                else:
                    self._kvcr._complete(op.handle, op.results)
        if op.probes:
            return
        self._ops.pop(op.op_id)
        self._breaker.release(op)

    def _finish_spill(self, op: _G4Op, xfer: _Xfer) -> None:
        core = self._kvcr
        self._ops.pop(op.op_id)
        self._pending_spills.discard(xfer.key)
        published = xfer.outcome == "DONE" and (self._root is None or xfer.published)
        self._record(xfer, published)
        try:
            if published:
                if op.latched:
                    self.spill_late += 1
                else:
                    self._count_success(op)
                self._retire_source(op)
            else:
                if xfer.outcome == "PUBLISH_ERR":
                    reason = "publish failed"
                elif xfer.start_failed:
                    reason = "start failed"
                else:
                    reason = "transfer failed"
                self._count_failure(
                    op, f"spill {reason}" + (f": {xfer.error}" if xfer.error else "")
                )
                self._recover_spill_failure(op, xfer.key, reason)
        finally:
            self._breaker.release(op)
            core._remove_block_dependencies(op)

    def _count_failure(self, op: _G4Op, reason: str) -> None:
        if not op.counted:
            op.counted = True
            self._breaker.data_failure(reason, op.epoch)

    def _count_success(self, op: _G4Op) -> None:
        if not op.counted and not op.reset:
            op.reset = True
            self._breaker.data_success()

    def _record(self, xfer: _Xfer, success: bool) -> None:
        """Count and record an ended transfer once; a spill succeeds when published."""
        if xfer.recorded:
            return
        xfer.recorded = True
        kind = xfer.op.kind
        if kind == "spill":
            if success:
                self._counters["g4_spill_published"] += 1
        else:
            self._counters[f"g4_{kind}_{'ok' if success else 'failed'}"] += 1
        self._kvcr._record_transfer(
            f"g4_{kind}",
            xfer.started_at,
            success,
            1,
            sum(size for _, size, _ in xfer.local),
        )

    def _report_uncertain(self, op: _G4Op) -> None:
        core = self._kvcr
        op.lifecycle_handle = core._next_fill_handle
        core._next_fill_handle -= 1
        op.uncertain_refs = [
            MemoryRef(
                end_point_name=ref.end_point_name,
                label=ref.label,
                element_index=ref.element_index,
            )
            for xfer in op.xfers
            if not xfer.finished
            for ref in xfer.refs
        ]
        self._report_memory(op, "uncertain")
        for xfer in op.xfers:
            op.results.setdefault(xfer.key, _FAILED_RESULT)
        core._complete(op.handle, op.results)

    def _report_memory(
        self, op: _G4Op, state: Literal["uncertain", "quiesced"]
    ) -> None:
        self._kvcr._notify_transfer_errors(
            [
                TransferError(
                    "KVCR G4 delivery memory",
                    op.lifecycle_handle,
                    state=state,
                    destination_regions=list(op.uncertain_refs or ()),
                    destination_owner="framework",
                )
            ]
        )

    # Close.

    def close_workers(self) -> None:
        """Drop unstarted jobs, let started transfers end, and join the worker."""
        self._closing = True
        worker = self._storage
        worker.stop()
        if worker.thread.ident is not None:
            # close_timeout_ms bounds the whole wait, a blocked call included.
            budget = self._options.close_timeout_ms / 1000
            final = min(_FINAL_JOIN_SECONDS, budget / 2)
            worker.thread.join(timeout=budget - final)
            if worker.thread.is_alive():
                # A worker leaving its loop just as the wait ended is not stuck.
                worker.thread.join(timeout=final)
        if worker.thread.is_alive():
            # Exiting while the agent's native threads still run aborts the
            # process; give the worker the cleanup budget again at exit.
            atexit.register(
                worker.thread.join,
                _progress._OP_CLEANUP_TIMEOUT_SECONDS + _progress._JOIN_TIMEOUT_SECONDS,
            )
            raise RuntimeError("G4 storage worker did not stop")
        if worker.failure is not None:
            raise worker.failure

    def is_quiescent(self) -> bool:
        return not self._storage.thread.is_alive() and not self._storage.unreleased

    def close_main(self) -> None:
        while True:
            try:
                xfer = self._op_done.get_nowait()
            except queue.Empty:
                break
            self._finish_xfer(xfer)
        local = self._kvcr._local_dram
        for op in tuple(self._ops.values()):
            self._ops.pop(op.op_id, None)
            if op.ended:
                continue
            open_keys = []
            for xfer in op.xfers:
                if xfer.finished:
                    continue
                xfer.finished = True
                open_keys.append(xfer.key)
                self._record(xfer, False)
                if xfer.fd is not None and not xfer.fd_closed:
                    xfer.fd_closed = True
                    with contextlib.suppress(OSError):
                        os.close(xfer.fd)
                if xfer.tmp is not None and not xfer.published:
                    with contextlib.suppress(OSError):
                        os.unlink(xfer.tmp)
            if op.kind == "spill":
                self._pending_spills.difference_update(op.keys)
                self._kvcr._release_local_dram_sources(op.op_id)
                for key in op.keys:
                    local._pools.abandon_capacity_eviction(key)
                self._kvcr._remove_block_dependencies(op)
            elif op.kind == "fill":
                if open_keys:
                    local.complete_fill(open_keys, success=False, source=CacheTier.G4)
            elif op.uncertain_refs is not None:
                self._report_memory(op, "quiesced")
            else:
                for key in open_keys:
                    op.results[key] = _FAILED_RESULT
                self._kvcr._complete(op.handle, op.results)
        self._checks.clear()
        self._timed_out.clear()
        self._pending_start.clear()
        self._outbox.clear()
        self._probes.clear()
        self._results.clear()
        self._owner_keys.clear()

    def telemetry_state(self) -> dict[str, int]:
        return {
            "g4_inflight_checks": self._inflight,
            "g4_pending_checks": self._pending_count,
            "g4_check_results": len(self._results),
            "g4_pending_spills": len(self._pending_spills),
            "g4_spill_late": self.spill_late,
            "g4_spill_dropped": self.spill_dropped,
            "g4_breaker_state": self._breaker.gauge(),
            "g4_breaker_failures": self._breaker.failures,
            **self._counters,
        }
