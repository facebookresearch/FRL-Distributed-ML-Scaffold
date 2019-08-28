#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import argparse
import itertools
import unittest
from enum import Enum
from typing import Dict, List, NamedTuple, Optional

from ..cli import (
    ScaffoldHelpFormatter,
    arguments_from_named_tuple,
    compose_type_arguments,
    compose_type_results,
    get_repro_args,
)


class TestEnum(Enum):
    NONE = "none"


class TestNestedArgs(NamedTuple):
    dict_field: Dict[str, int] = {}
    list_field: List[float] = []
    enum_field: TestEnum = TestEnum.NONE


class TestArgs(NamedTuple):
    int_field: int
    str_field: str
    bool_field: bool
    false_bool_field: bool
    nested_field: TestNestedArgs
    optional_int_field: int = 0
    optional_str_field: str = "test"
    optional_bool_field: bool = True
    optional_nested_field: Optional[TestNestedArgs] = None
    optional_nested_field2: Optional[TestNestedArgs] = None


TEST_CLI = [
    "--int_field",
    "3",
    "--str_field",
    "test",
    "--bool_field",
    "1",
    "--false_bool_field",
    "0",
    "--nested_field.dict_field",
    '{"a": 1, "b": 2}',
    "--nested_field.list_field",
    "[1, 2]",
    "--nested_field.enum_field",
    "none",
    "--optional_nested_field.enum_field",
    "none",
]
TEST_STRUCT = TestArgs(
    int_field=3,
    str_field="test",
    bool_field=True,
    false_bool_field=False,
    nested_field=TestNestedArgs(
        dict_field={"a": 1, "b": 2}, list_field=[1, 2], enum_field=TestEnum.NONE
    ),
    optional_nested_field=TestNestedArgs(enum_field=TestEnum.NONE),
)


class TestPosixIndexedDatasetReader(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = argparse.ArgumentParser(formatter_class=ScaffoldHelpFormatter)
        requiredGroup = self.parser.add_argument_group("required arguments")
        optionalGroup = self.parser.add_argument_group("optional arguments")
        compose_type_arguments(TestArgs, optionalGroup, requiredGroup)

    def test_deserialize(self):
        args = vars(self.parser.parse_args(TEST_CLI))
        parsed_test_struct = compose_type_results(TestArgs, args)
        self.assertEqual(parsed_test_struct, TEST_STRUCT)

    def test_reserialize(self):
        cli_args = arguments_from_named_tuple(TestArgs, TEST_STRUCT)
        serialized_args = list(
            itertools.chain(
                *[
                    (arg.name, arg.serializer(arg.value))
                    for arg in (cli_args.required + cli_args.optional)
                    if arg.value is not None
                ]
            )
        )
        args = vars(self.parser.parse_args(serialized_args))
        parsed_test_struct = compose_type_results(TestArgs, args)
        self.assertEqual(parsed_test_struct, TEST_STRUCT)

    def test_get_repro_args(self):
        repro_str = get_repro_args(TEST_STRUCT).strip()
        orig_words = []
        for word in TEST_CLI:
            if " " in word:
                if '"' in word:
                    orig_words.append(f"'{word}'")
                else:
                    orig_words.append(f'"{word}"')
            else:
                orig_words.append(word)
        orig_str = " ".join(orig_words).strip()
        repro_pieces = [x.strip() for x in repro_str.split("--")]
        orig_pieces = [x.strip() for x in orig_str.split("--")]
        for piece in orig_pieces:
            self.assertIn(piece, repro_pieces)
