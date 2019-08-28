#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import logging
from typing import Dict, Optional, Union


logger: logging.Logger = logging.getLogger("Config")


ConfigValue = Union[str, int]


class Config:
    _instance: Optional["Config"] = None

    def __init__(self) -> None:
        self._store: Dict[str, ConfigValue] = {}

    @classmethod
    def get(cls) -> "Config":
        if not cls._instance:
            cls._instance = Config()
        return cls._instance

    def configure(self, key: str, value: ConfigValue) -> None:
        logger.info(f"storing configuration: {key} : {value}")
        self._store[key] = value

    def read_key(self, key: str) -> Optional[ConfigValue]:
        if key in self._store:
            logger.info(f"reading configuration: {key} : {self._store[key]}")
            return self._store[key]
        else:
            return None
