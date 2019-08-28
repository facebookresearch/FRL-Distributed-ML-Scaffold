#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
from contextlib import ExitStack
from time import sleep
from unittest import TestCase
from unittest.mock import patch

from ..watchdog_timer import WatchdogTimer


class WatchdogTimerTest(TestCase):
    def test_timer_completes(self):
        with ExitStack() as exit_stack:
            set_mock = exit_stack.enter_context(patch.object(WatchdogTimer, "_set"))
            cancel_mock = exit_stack.enter_context(
                patch.object(WatchdogTimer, "_cancel")
            )

            LONG_TIMEOUT: int = 1000000
            with WatchdogTimer.create(LONG_TIMEOUT):
                call_list = set_mock.call_args_list
                self.assertEqual(len(call_list), 1)
                first_call = call_list[0]
                posit_args = first_call[0]
                first_arg = posit_args[0]
                self.assertEqual(first_arg, LONG_TIMEOUT)
            call_list = cancel_mock.call_args_list
            self.assertEqual(len(call_list), 1)

    def test_timer_expires_with_exception(self):
        with WatchdogTimer.create(0):
            self.assertRaises(TimeoutError, sleep, 1)  # 0 doesn't work on all platforms
