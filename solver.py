#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import io
import json
import logging
import math
import multiprocessing
import os
import pickle
import traceback
from collections import defaultdict
from contextlib import ExitStack
from itertools import chain
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from typing import (
    IO,
    Any,
    DefaultDict,
    Dict,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Union,
)

import cv2
import numpy as np
import plotly.graph_objs as go
import torch
import torch.nn as nn

from .criteria import BaseParallelCriterion
from .handlers import *  # noqa F403
from .lr_scheduler import DropEpochsScheduler, WarmupMultiStepLR
from .problem import Problem
from .solver_worker import (
    FractionalPerformanceSummary,
    SerializableSampleSummary,
    SolverWorker,
)
from .storage import StoragePath, storage_open, storage_os
from .storage_layers.dataset import SharedMemoryDatasetCache
from .types import (
    Device,
    LRSchedulerAlgorithm,
    Mode,
    OptAlgorithm,
    OptimOpts,
    RunOpts,
    SampleSummary,
    Split,
)
from .utils import import_config, mp_process


logging.basicConfig(
    level=logging.INFO, format="%(levelname)s (%(process)d) %(message)s"
)
logger = logging.getLogger(__name__)


CHECKPOINT_NAME = ".checkpoint.pth"


class SingleSampleSummary(NamedTuple):
    plot: go.Figure
    source: str


class SplitSampleSummaryGroup(NamedTuple):
    image: Optional[bytes]
    text: Optional[str]
    summaries: List[SingleSampleSummary]


class SplitSampleSummary(NamedTuple):
    random: SplitSampleSummaryGroup
    worst: SplitSampleSummaryGroup


class EpochSplitPerformanceSummary(NamedTuple):
    losses: Dict[str, float]  # loss name : loss value
    metrics: Dict[str, float]  # metric name : metric value
    sample_summary: SplitSampleSummary


class PerformanceSummary(NamedTuple):
    epoch: int
    performance: Dict[Split, EpochSplitPerformanceSummary]
    save_dir: str


class Checkpoint(NamedTuple):
    epoch: int
    modelState: Dict[Any, Any]
    optimizerState: Dict[Any, Any]


class SolverWorkerArgs(NamedTuple):
    run_opts: RunOpts
    problem: Problem
    save_dir: StoragePath
    run_device: Device
    node_idx: int
    node_count: int
    rank: int
    local_rank: int
    world_size: int
    group_name: Optional[str]
    init_method: str
    cache: SharedMemoryDatasetCache


def _load_model_state(model: nn.Module, model_path: str, strict: bool = True) -> None:
    with storage_open(StoragePath(model_path), "rb") as f:
        checkpoint = torch.load(f)
    print(checkpoint)
    new_state_dict = checkpoint["state_dict"]
    if strict:
        # requires the model's state_dict to be exactly same as the new_state_dict
        model.load_state_dict(new_state_dict)
    else:
        # only load parameters that are same size and type, and print warnings
        # for parameters that don't match
        own_state = model.state_dict()
        for name, param in new_state_dict.items():
            if name in own_state:
                if isinstance(param, torch.nn.Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                if own_state[name].size() == param.size():
                    own_state[name].copy_(param)
                else:
                    print(
                        "Warning: While copying the parameter named {}, "
                        "whose dimensions in the model are {} and "
                        "whose dimensions in the checkpoint are {}.".format(
                            name, own_state[name].size(), param.size()
                        )
                    )
            else:
                print(
                    "Warning: Parameter named {} is not used "
                    "by this model.".format(name)
                )
        for name, _ in own_state.items():
            if name not in new_state_dict:
                print(
                    "Warning: Parameter named {} in the model "
                    "is not initialized.".format(name)
                )


def _create_optimizer(parameters, optim_opts: OptimOpts) -> torch.optim.Optimizer:
    if optim_opts.algo == OptAlgorithm.RMSPROP:
        optimizer = torch.optim.RMSprop(
            parameters,
            lr=optim_opts.lr,
            momentum=optim_opts.momentum,
            weight_decay=optim_opts.weightDecay,
        )
    elif optim_opts.algo == OptAlgorithm.SGD:
        optimizer = torch.optim.SGD(
            parameters,
            lr=optim_opts.lr,
            momentum=optim_opts.momentum,
            weight_decay=optim_opts.weightDecay,
        )
    elif optim_opts.algo == OptAlgorithm.ADAM:
        optimizer = torch.optim.Adam(
            parameters,
            lr=optim_opts.lr,
            weight_decay=optim_opts.weightDecay,
            eps=optim_opts.epsilon,
            amsgrad=optim_opts.amsgrad,
        )
    else:
        # Not implemented
        raise ValueError("Unknown optimization algorithm type")
    return optimizer


def create_lr_scheduler(run_opts: RunOpts, optimizer, checkpoint_epoch=-1):
    lrs_opts = run_opts.optim.lr_scheduler
    if lrs_opts.algo == LRSchedulerAlgorithm.DropEpochs:
        # set the learning rate drop schedule
        drop_epochs = []
        if run_opts.nEpochs > 10:
            drop_epochs.append(np.floor(run_opts.nEpochs * 0.66667))
            drop_epochs.append(np.floor(run_opts.nEpochs * 0.9))
        scheduler = DropEpochsScheduler(
            optimizer, drop_epochs, last_epoch=checkpoint_epoch
        )
    elif lrs_opts.algo == LRSchedulerAlgorithm.WarmupMultiStepLR:
        steps_ratio = [0.33333, 0.66667, 0.9]
        steps = [np.floor(run_opts.nEpochs * x) for x in steps_ratio]
        scheduler = WarmupMultiStepLR(
            optimizer,
            steps,
            gamma=0.1,
            warmup_factor=1.0 / 1000,
            warmup_iters=5,
            warmup_method="linear",
            last_epoch=checkpoint_epoch,
        )
    else:
        # Not implemented
        raise ValueError("Unknown optimization algorithm type")

    return scheduler


def _aggregate_sample_summaries(
    sample_results: List[SampleSummary]
) -> SplitSampleSummaryGroup:
    image_samples = [s.image for s in sample_results if s.image is not None]
    info_samples = [s.text for s in sample_results if s.text is not None]
    sample_summaries = [
        # pyre-fixme[6]: Expected `str` for 2nd param but got `Optional[str]`.
        SingleSampleSummary(plot=s.plot, source=s.source)
        for s in sample_results
        if s.plot is not None and s.source is not None
    ]

    return SplitSampleSummaryGroup(
        image=_save_img(np.concatenate(image_samples)) if image_samples else None,
        # pyre-fixme[6]: Expected `Iterable[str]` for 1st param but got
        #  `List[Optional[str]]`.
        text="\n".join(info_samples) if info_samples else None,
        summaries=sample_summaries,
    )


def _save_img(img: Optional[np.ndarray]) -> Optional[bytes]:
    if img is not None:
        # Swap color order (we use RGB, OpenCV expects BGR)
        img = np.flip(img, axis=2)
        ret, frame_bytes = cv2.imencode(".png", img)
        return frame_bytes.tostring()
    return None


class Solver:
    @staticmethod
    def _load_checkpoint(checkpoint_file: IO) -> Checkpoint:
        logger.info("Loading from check point %s", str(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)

        epoch = checkpoint["epoch"]
        logger.info("Loaded from check point, starting from epoch %d", epoch)
        return Checkpoint(
            epoch=epoch,
            modelState=checkpoint["state_dict"],
            optimizerState=checkpoint["optimizer"],
        )

    @staticmethod
    def _init_distributed_model(
        model: torch.nn.Module,
        init_method: str,
        group_name: Optional[str],
        *,
        rank: int,
        local_rank: int,
        world_size: int,
    ) -> torch.nn.parallel.DistributedDataParallel:
        logger.info("Initializing process group with %s" % init_method)
        torch.distributed.init_process_group(
            backend="nccl",
            init_method=init_method,
            world_size=world_size,
            rank=rank,
            group_name=group_name,
        )
        logger.info(
            "Setting up distributed model (local_rank=%d, world_size=%d)"
            % (local_rank, world_size)
        )
        dist_model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank
        )
        logger.info(
            "Distributed model established (local_rank=%d, world_size=%d)"
            % (local_rank, world_size)
        )
        return dist_model

    @classmethod
    def _get_model(
        cls,
        solver_worker_args: SolverWorkerArgs,
        checkpoint: Optional[Checkpoint],
        device: torch.device,
    ) -> Union[torch.nn.Module, torch.nn.parallel.distributed.DistributedDataParallel]:
        run_opts = solver_worker_args.run_opts
        model = solver_worker_args.problem.get_model()
        model_path = run_opts.initialModelPath
        if model_path is not None:
            # If we are evaluating this model, it should be loaded with strict mode,
            # if we are using it as a pretrained model for initialization, we
            # allow to use partial of the layers's weights to initialize
            # our training model.
            if run_opts.mode == Mode.EVAL:
                _load_model_state(model, model_path)
            else:
                _load_model_state(model, model_path, strict=False)
        elif checkpoint:
            model.load_state_dict(checkpoint.modelState)

        model.to(device)
        if (
            solver_worker_args.run_device == Device.GPU
            and solver_worker_args.world_size > 1
        ):
            model = cls._init_distributed_model(
                model=model,
                init_method=solver_worker_args.init_method,
                group_name=solver_worker_args.group_name,
                rank=solver_worker_args.rank,
                local_rank=solver_worker_args.local_rank,
                world_size=solver_worker_args.world_size,
            )
        return model

    @classmethod
    def _run_solver_worker(
        cls, solver_worker_args: SolverWorkerArgs
    ) -> Iterator[FractionalPerformanceSummary]:
        """
        Sets up an instance of SolverWorker to train on a fraction of the dataset. There
        should be `world_size` total instances set up, with instances having `rank`
        from 0 to `world_size`. There may be multiple instances on one machine
        (one per GPU, for example), and there may be multiple machines too. `local_rank`
        is the `rank` but 0-indexed on a particular machine.
        """
        run_opts = solver_worker_args.run_opts
        problem = solver_worker_args.problem
        import_config(run_opts)

        # need to move model and criterion to device _before_ creating
        # the optimizer and/or load an existing model
        if (
            solver_worker_args.run_device == Device.GPU
            and solver_worker_args.world_size > 1
        ):
            torch.cuda.set_device(solver_worker_args.local_rank)

        devStr = solver_worker_args.run_device.value
        # pyre-fixme[19]: Expected 0 positional arguments.
        device = torch.device(devStr)
        logger.info("Using device %s" % devStr)

        checkpoint: Optional[Checkpoint] = None
        try:
            with storage_open(solver_worker_args.save_dir / CHECKPOINT_NAME, "rb") as f:
                checkpoint = cls._load_checkpoint(f)
        except FileNotFoundError:
            pass

        model = cls._get_model(solver_worker_args, checkpoint, device)

        criterion: BaseParallelCriterion = problem.get_criterion().to(device)

        # create optimizer
        optimizer = _create_optimizer(
            chain(model.parameters(), criterion.parameters()), run_opts.optim
        )
        if checkpoint:
            optimizer.load_state_dict(checkpoint.optimizerState)
            optimizer.zero_grad()

        solver_worker = SolverWorker(
            model,
            criterion,
            optimizer,
            device=device,
            run_opts=run_opts,
            cache=solver_worker_args.cache,
            local_rank=solver_worker_args.local_rank,
            node_idx=solver_worker_args.node_idx,
            node_count=solver_worker_args.node_count,
        )

        # set the learning rate drop schedule
        scheduler = create_lr_scheduler(
            run_opts, solver_worker.optimizer, checkpoint.epoch if checkpoint else -1
        )

        if run_opts.mode == Mode.TRAIN:
            yield from solver_worker.train(
                problem,
                startEpoch=checkpoint.epoch if checkpoint else 0,
                nEpochs=run_opts.nEpochs,
                batchSize=run_opts.batchSize,
                scheduler=scheduler,
            )

        elif run_opts.mode == Mode.EVAL:
            yield from solver_worker.eval(problem, batchSize=run_opts.batchSize)
        else:
            raise ValueError("unknown mode")

    @classmethod
    @mp_process
    def _solver_worker_process(
        cls,
        solver_worker_args: SolverWorkerArgs,
        comms_connection: Connection,
        # pyre-ignore
        cleanup_flag: multiprocessing.Event,  # noqa T484 (this does exist)
    ) -> None:
        try:
            for result in cls._run_solver_worker(solver_worker_args):
                # We do our own pickling here to ensure that we use the v4
                # pickle protocol for proper large-object support:
                # https://www.python.org/dev/peps/pep-3154/
                comms_connection.send_bytes(
                    pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
                )
            # `None` is a placeholder to signal that a worker has cleanly finished
            comms_connection.send(None)
        except Exception as e:
            logger.error(str(e))
            for l in traceback.format_exc().splitlines(False):
                logger.error(l)
            comms_connection.send(e)
            raise
        finally:
            # We need to wait to clean up to make sure the communication pipes
            # are fully processed/cleared.
            cleanup_flag.wait()  # noqa T484

    @staticmethod
    def _deserialize_sample_summaries(
        serialized_summaries: Sequence[SerializableSampleSummary]
    ) -> List[SampleSummary]:
        return [
            SampleSummary(
                image=s.image,
                text=s.text,
                # pyre-fixme[6]: Expected `Union[bytearray, bytes, str]` for 1st
                #  param but got `Optional[str]`.
                plot=go.Figure(**json.loads(s.plot)) if s.plot else None,
                source=s.source,
            )
            for s in serialized_summaries
        ]

    @classmethod
    def _aggregate_fractional_results(
        cls,
        run_opts: RunOpts,
        problem: Problem,
        fractional_results: List[FractionalPerformanceSummary],
    ) -> Dict[Split, EpochSplitPerformanceSummary]:
        # dataset splits (train, test, heldout)
        splits = [d.data_type for d in problem.datasets]
        metrics: Dict[Split, DefaultDict[str, float]] = {
            split: defaultdict(lambda: 0.0) for split in splits
        }
        losses: Dict[Split, DefaultDict[str, float]] = {
            split: defaultdict(lambda: 0.0) for split in splits
        }
        n_samples = {split: 0 for split in splits}

        samples_set: Dict[Split, List[SampleSummary]] = {split: [] for split in splits}
        worst_samples_set: Dict[Split, List[SampleSummary]] = {
            split: [] for split in splits
        }

        for fractional_result in fractional_results:
            for split, perf in fractional_result.performance.items():
                for metric_key in perf.metrics:
                    metrics[split][metric_key] += (
                        perf.metrics[metric_key] * perf.nSamples
                    )
                for loss_key in perf.losses:
                    losses[split][loss_key] += perf.losses[loss_key] * perf.nSamples
                n_samples[split] += perf.nSamples

                samples_set[split] += cls._deserialize_sample_summaries(perf.samples)
                worst_samples_set[split] += cls._deserialize_sample_summaries(
                    perf.worstSamples
                )

        def filter_metrics_RMSE(
            metrics_in: DefaultDict[str, float], split: Split
        ) -> DefaultDict[str, float]:
            # MSE metrics aggregate correctly but are difficult to interpret;
            # swap them out with RMSE here (after aggregating):
            metrics_out: DefaultDict[str, float] = defaultdict(lambda: 0.0)
            for (metric_name, metric_value) in metrics_in.items():
                metric_value = metric_value / n_samples[split]
                assert metric_name.find("RMSE") == -1
                if metric_name.find("MSE") != -1:
                    head, _, tail = metric_name.rpartition("MSE")
                    metric_name_new = head + "RMSE" + tail
                    # Some metrics may be negative due to being invalid.
                    metrics_out[metric_name_new] = (
                        math.sqrt(metric_value) if metric_value > 0 else metric_value
                    )
                else:
                    metrics_out[metric_name] = metric_value
            return metrics_out

        epoch_stats = {}
        for split in splits:
            random_summary = _aggregate_sample_summaries(samples_set[split])
            worst_summary = _aggregate_sample_summaries(worst_samples_set[split])

            epoch_stats[split] = EpochSplitPerformanceSummary(
                losses={k: v / n_samples[split] for k, v in losses[split].items()},
                metrics=filter_metrics_RMSE(metrics[split], split),
                sample_summary=SplitSampleSummary(
                    random=random_summary, worst=worst_summary
                ),
            )
        return epoch_stats

    @staticmethod
    def _save_epoch_group_summary(
        sample_summaries: SplitSampleSummaryGroup,
        save_dir: StoragePath,
        group_name: str,
        split: Split,
        epoch: int,
    ) -> None:

        if sample_summaries.image is not None:
            with storage_open(
                save_dir / (".%s_%s_%04d.png" % (split.value, group_name, epoch)), "wb"
            ) as f_img:
                f_img.write(sample_summaries.image)

        if sample_summaries.text is not None:
            with storage_open(
                save_dir / (".%s_%s_%04d.txt" % (split.value, group_name, epoch)), "wb"
            ) as f_txt:
                # pyre-fixme[16]: `Optional` has no attribute `encode`.
                f_txt.write(sample_summaries.text.encode("utf-8"))

    @classmethod
    def _save_epoch_summary(
        cls,
        epoch: int,
        save_dir: StoragePath,
        epoch_stats: Dict[Split, EpochSplitPerformanceSummary],
    ) -> None:
        for split, summary in epoch_stats.items():
            cls._save_epoch_group_summary(
                summary.sample_summary.random, save_dir, "random", split, epoch
            )
            cls._save_epoch_group_summary(
                summary.sample_summary.worst, save_dir, "worst", split, epoch
            )

    @classmethod
    def _save_checkpoint(
        cls,
        epoch: int,
        save_dir: StoragePath,
        run_opts: RunOpts,
        problem: Problem,
        fractional_results: List[FractionalPerformanceSummary],
        base_filename: str,
    ) -> None:
        # Because workers broadcast/share buffers with NCCL, the model and some
        # other things should be identical or equivalent across fractional results
        # from different workers. For those things we can select them from an
        # arbitrary fractional result.

        # Note we are trying to keep disk operations in the parent process so workers
        # can devote their time to training. Also note that anything requiring a GPU
        # must be done in a worker, since the parent process has no GPU context.
        arbitrary_result = fractional_results[0]
        split_priority = [Split.HELDOUT, Split.TEST, Split.TRAIN]
        test_io = None
        for split in split_priority:
            if split in arbitrary_result.performance:
                test_io = arbitrary_result.performance[split].testIO
                break
        if test_io is None:
            raise RuntimeError("No splits found")

        test_input = [
            torch.stack([sample.data[i] for sample in test_io])
            for i in range(len(test_io[0].data))
        ]

        test_output = [
            torch.stack([sample.output[i] for sample in test_io])
            for i in range(len(test_io[0].output))
        ]

        with io.BytesIO(arbitrary_result.modelBuffer) as model_buffer:
            model = torch.load(model_buffer, map_location="cpu")

        with io.BytesIO(
            arbitrary_result.optimizerStateBuffer
        ) as optimizer_state_buffer:
            optimizer_state_dict = torch.load(
                optimizer_state_buffer, map_location="cpu"
            )

        # =============================== Checkpointing ==================== #
        logger.info("==> saving checkpoint to %s", str(save_dir))
        # We write dot-prefixed checksum files to make blobsync skip
        # syncing them back to the isilons. Because these files are
        # frequently overwritten they don't play nice with blobsync in
        # its current state.
        with ExitStack() as stack:
            checkpoint_file = stack.enter_context(
                storage_open(save_dir / base_filename, "wb")
            )
            model_file = stack.enter_context(
                storage_open(save_dir / (base_filename + ".model"), "wb")
            )
            data_file = stack.enter_context(
                storage_open(save_dir / (base_filename + ".test_data"), "wb")
            )
            anno_param_file = stack.enter_context(
                storage_open(save_dir / (base_filename + ".annotate_param"), "wb")
            )
            torch.save(
                {
                    "epoch": epoch,
                    "optimizer": optimizer_state_dict,
                    "state_dict": model.state_dict(),
                },
                checkpoint_file,
            )

            # also try save the whole model so that one can load it directly without
            # specify the architecture before loading. This is useful for evaluation
            torch.save(model, model_file)
            torch.save(
                {"test_input": test_input, "test_output": test_output}, data_file
            )
            torch.save(
                # pyre-fixme[16]: `Optional` has no attribute `_asdict`.
                problem.anno_param._asdict() if problem.anno_param else {},
                anno_param_file,
            )

    @classmethod
    def _collect_direct_fractional_results(
        cls, solver_worker_args: SolverWorkerArgs
    ) -> Iterator[List[FractionalPerformanceSummary]]:
        """
        When in singleThreaded mode the results from the single instance of the
        SolverWorker class need to be wrapped in a list in order to mirror the data
        structure returned when we have multiple worker processes each handling a
        subset of the epoch data (a list of partial results)
        """
        for res in cls._run_solver_worker(solver_worker_args):
            yield [res]

    @classmethod
    def _collect_process_fractional_results(
        cls,
        processes: List[Process],
        parent_pipes: List[Connection],
        # pyre-ignore
        cleanup_flag: multiprocessing.Event,  # noqa T484 (this does exist)
    ) -> Iterator[List[FractionalPerformanceSummary]]:
        """
        When not in singleThreaded mode there will be multiple processes each
        generating results for a subset of data in an epoch. They send these
        partial results back via a set of pipes and this method will listen to
        those pipes collecting all of the partial results for the current epoch.
        """
        aggregations: DefaultDict[
            int, List[FractionalPerformanceSummary]
        ] = defaultdict(list)
        try:
            cleanlyExitedProcesses: List[Connection] = []

            while len(cleanlyExitedProcesses) < len(processes):
                ready_pipes = multiprocessing.connection.wait(parent_pipes, 5)
                for parent_pipe in ready_pipes:
                    assert isinstance(parent_pipe, Connection)
                    try:
                        message = parent_pipe.recv()
                        if message is None:
                            logger.info("Worker process ready to exit")
                            cleanlyExitedProcesses.append(parent_pipe)
                        elif isinstance(message, Exception):
                            logger.info("Received exception from child process")
                            raise message
                        else:
                            epoch = message.epoch
                            aggregations[epoch].append(message)
                            if len(aggregations[epoch]) == len(processes):
                                # We received all the FractionalPerformanceSummary
                                # messages from each local worker on this machine,
                                # now roll them up into one PerformanceSummary
                                res = aggregations[epoch]
                                del aggregations[epoch]
                                yield res
                    except EOFError:
                        if parent_pipe not in cleanlyExitedProcesses:
                            raise Exception("Child process failed to exit cleanly.")
                        logger.info("Worker process has exited")

        except Exception:
            logger.exception("Unexpected error in solve")
            cleanup_flag.set()  # noqa T484 (this does exist)
            logger.info("Sending SIGTERM to all child processes.")
            for p in processes:
                logger.info("Killing process with pid " + str(p.pid))
                p.terminate()
            raise

        finally:
            cleanup_flag.set()  # noqa T484 (this does exist)
            logger.info("Joining worker processes")
            for p in processes:
                p.join()

    @classmethod
    def solve(
        cls,
        run_opts: RunOpts,
        problem: Problem,
        *,
        group_name: Optional[str],
        init_method: str,
        node_idx: int = 0,
        node_count: int = 1,
        memory_quota: int = 0,
    ) -> Iterator[PerformanceSummary]:
        run_device = (
            Device.GPU
            if not run_opts.cpuonly and torch.cuda.is_available()
            else Device.CPU
        )

        if run_device == Device.GPU and not run_opts.singleThreaded:
            device_count = torch.cuda.device_count()
            world_size = device_count * node_count
        else:
            device_count = 1
            world_size = 1
        logger.info("World size %d, device count %d" % (world_size, device_count))

        num_workers = min(device_count, world_size)
        canonical_dataset = max(problem.datasets, key=lambda x: len(x))
        cache = SharedMemoryDatasetCache(
            dataset=canonical_dataset,
            local_worker_count=num_workers,
            total_worker_count=world_size,
            memory_quota=memory_quota,
        )

        if node_count > 1:
            save_dir = StoragePath(problem.save_dir + str(node_idx))
        else:
            save_dir = StoragePath(problem.save_dir)
        storage_os.makedirs(save_dir, exist_ok=True, ttl=run_opts.outputTTL)

        logger.info("Parent process has pid " + str(os.getpid()))
        processes: List[Process] = []
        cleanup_flag = multiprocessing.Event()
        parent_pipes: List[Connection] = []
        solver_worker_args: Optional[SolverWorkerArgs] = None
        for local_rank in range(num_workers):
            solver_worker_args = SolverWorkerArgs(
                run_opts=run_opts,
                problem=problem,
                save_dir=save_dir,
                run_device=run_device,
                local_rank=local_rank,
                node_idx=node_idx,
                node_count=node_count,
                rank=node_idx * device_count + local_rank,
                world_size=world_size,
                group_name=group_name,
                init_method=init_method,
                cache=cache,
            )
            if not run_opts.singleThreaded:
                parent_conn, child_conn = Pipe(duplex=False)
                parent_pipes.append(parent_conn)
                p = Process(
                    target=cls._solver_worker_process,
                    kwargs={
                        "solver_worker_args": solver_worker_args,
                        "comms_connection": child_conn,
                        "cleanup_flag": cleanup_flag,
                    },
                )
                p.start()
                processes.append(p)
                logger.info("Started worker with rank: %d pid: %d", local_rank, p.pid)
        assert solver_worker_args is not None
        if run_opts.singleThreaded:
            assert num_workers == 1, "Single threaded run cannot use multiple workers"
            fractional_result_generator = cls._collect_direct_fractional_results(
                solver_worker_args
            )
        else:
            fractional_result_generator = cls._collect_process_fractional_results(
                processes, parent_pipes, cleanup_flag
            )

        # Checkpoint frequency: if the data set size is big, then save after
        # every epoch so we don't lose much when training fails; otherwise
        # save every 5 epochs
        assert len(problem.datasets) > 0, "datasets cannot be empty"
        save_every = 1 if len(problem.datasets[0]) > 300_000 else 5

        for res in fractional_result_generator:
            epoch = res[0].epoch
            epoch_stats = cls._aggregate_fractional_results(run_opts, problem, res)

            if run_opts.mode == Mode.TRAIN:
                is_final_epoch = epoch == run_opts.nEpochs
                if epoch % save_every == 0 or is_final_epoch:
                    cls._save_epoch_summary(epoch, save_dir, epoch_stats)
                    base_filename = (
                        "final_model.pth" if is_final_epoch else CHECKPOINT_NAME
                    )
                    cls._save_checkpoint(
                        epoch, save_dir, run_opts, problem, res, base_filename
                    )

            yield PerformanceSummary(
                epoch=epoch, performance=epoch_stats, save_dir=str(save_dir)
            )
