#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import ctypes
from contextlib import contextmanager
from logging import Logger
from sys import _current_frames
from threading import Timer, current_thread
from traceback import extract_stack
from typing import Iterable, Optional


logger = Logger("watchdog timer")


class WatchdogTimer:
    """This class guards against hangs or abnormally long completion times.

    It establishes a timer with a predefined timeout, and if it expires,
    the stacks for all threads in the process are logged and a TimeoutError is
    injected into the hanging thread.

    """

    def __init__(self):
        self._timer: Optional[Timer] = None
        self._tid: int = current_thread().ident

    @classmethod
    @contextmanager
    def create(cls, timeout_ms: int) -> Iterable["WatchdogTimer"]:
        """Creates a context manager watchdog

        If the context manager is not exited after timeout_ms elapse, a
        TimeoutError is generated in the thread that established the watchdog

        """
        timer: "WatchdogTimer" = WatchdogTimer()
        timer._set(timeout_ms)
        yield timer
        timer._cancel()

    def _on_watchdog_expired(self) -> None:
        logger.warning("Watchdog timer has expired;  dumping stacks.")
        for tid, stack in _current_frames().items():
            logger.warning(f"Thread {tid} stack:")
            for filename, line_number, function, line_text in extract_stack(stack):
                logger.warning(f"\t{filename}!{function}:{line_number}:")
                logger.warning(f"\t\t{line_text}:")

        logger.warning("Watchdog timer sending TimeoutError into hanging thread.")
        ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_long(self._tid), ctypes.py_object(TimeoutError)
        )

    def _set(self, timeout_ms: int) -> None:
        self._timer = Timer(timeout_ms / 1000.0, self._on_watchdog_expired)
        self._timer.start()

    def _cancel(self) -> None:
        if self._timer is not None:
            # pyre-fixme[16]: `Optional` has no attribute `cancel`.
            self._timer.cancel()
