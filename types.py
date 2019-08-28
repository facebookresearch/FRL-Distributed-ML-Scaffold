#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import warnings
from enum import Enum
from typing import NamedTuple, Optional, TypeVar

# isort:skip_file
# Keep this above the plotly import or it wont suppress the import side effect warnings
warnings.filterwarnings(
    "ignore",
    message="Looks like you don't have 'read-write' permission to your 'home'.+",
)

import numpy as np
import plotly.graph_objs as go


TNamedTuple = TypeVar("TNamedTuple")


def _setup_namedtuple_args(cls: TNamedTuple, *args, **kwargs):
    working_args = cls._field_defaults.copy()
    positional_keys = [
        k for k, _ in cls.__annotations__.items() if k not in kwargs.keys()
    ]
    working_args.update(dict(zip(positional_keys[: len(args)], args)))
    working_args.update(kwargs)
    return working_args


class ShuffleType(Enum):
    RANDPERM = "randperm"
    PER_NODE_RANDPERM = "per_node_randperm"


class Split(Enum):
    TRAIN = "training"
    TEST = "testing"
    HELDOUT = "heldOut"


class Device(Enum):
    CPU = "cpu"
    GPU = "cuda"


class Mode(Enum):
    EVAL = "eval"
    TRAIN = "train"


class OptAlgorithm(Enum):
    RMSPROP = "rmsprop"
    SGD = "sgd"
    ADAM = "adam"


class LossType(Enum):
    MSE = "mse"
    CrossEntropy = "crossentropy"


class SampleSummary(NamedTuple):
    image: Optional[np.ndarray] = None
    text: Optional[str] = None
    plot: Optional[go.Figure] = None
    source: Optional[str] = None


class LRSchedulerAlgorithm(Enum):
    DropEpochs = "drop"
    WarmupMultiStepLR = "multistep"


class LRSchedulerOpts(NamedTuple):
    algo: LRSchedulerAlgorithm = LRSchedulerAlgorithm.DropEpochs


# OptimOpts
class OptimOpts(NamedTuple):
    algo: OptAlgorithm
    lr: float = 0.001
    lr_scheduler: LRSchedulerOpts = LRSchedulerOpts()
    weightDecay: float = 0.00001
    momentum: float = 0.9
    epsilon: float = 1e-8
    amsgrad: bool = False
    gradientClip: float = 0.0


class OptimOptsBase(OptimOpts):
    def __new__(cls, *args, **kwargs):
        working_args = _setup_namedtuple_args(cls, *args, **kwargs)
        return super().__new__(cls, **working_args)


class RunOpts(NamedTuple):
    optim: OptimOpts
    batchSize: int
    cpuonly: bool = False
    nEpochs: int = 75
    # max number of images per epoch if > 0
    maxEpochImages: int = 0
    numThreads: int = 4
    numIOThreads: int = 5
    metricAmortizationSchedule: int = 10
    initialModelPath: Optional[str] = None
    mode: Mode = Mode.TRAIN
    numVisualizedSamples: int = 36
    singleThreaded: bool = False
    outputTTL: int = 0
    # frequency to log the loss to output, 0 to disable, n to log every n iterations
    lossLoggingFreq: int = 0
    debugGrad: bool = False
    shuffleType: ShuffleType = ShuffleType.RANDPERM
    minibatchTimeoutMs: int = 1000 * 60 * 60


class RunOptsBase(RunOpts):
    def __new__(cls, *args, **kwargs):
        working_args = _setup_namedtuple_args(cls, *args, **kwargs)
        return super().__new__(cls, **working_args)
