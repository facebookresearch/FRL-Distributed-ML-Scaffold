#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import heapq
import io
import itertools
import json
import logging
import random
import time
import warnings
from collections import defaultdict
from math import ceil
from typing import (
    Any,
    DefaultDict,
    Dict,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import _LRScheduler

from . import utils
from .config import Config
from .configurables import CONFIG_MINIBATCH_TIMEOUT_MS
from .criteria import BaseParallelCriterion, GradNormWeightedCriterion
from .model import MultiTaskModel
from .problem import BatchMetrics, Ordering, Problem
from .sampler import ScaffoldSampler
from .storage_layers.dataset import (
    CachedDatasetAccessor,
    MultifieldDataset,
    SharedMemoryDatasetCache,
    SubsetMultifieldDataset,
)
from .types import Device, Mode, RunOpts, SampleSummary, Split
from .watchdog_timer import WatchdogTimer


logging.basicConfig(level=logging.INFO, format="%(levelname)s(%(process)d) %(message)s")
logger = logging.getLogger(__name__)

# This is innocuous and there's no way to fix the underlying issue in fbcode.
# https://discuss.pytorch.org/t/got-warning-couldnt-retrieve-source-code-for-container/7689/7 # noqa: B950
warnings.filterwarnings(
    "ignore",
    message="Couldn't retrieve source code for container of type [^ ]+ "
    + "It won't be checked for correctness upon loading\\.",
)


# We need some set of samples for caffe2 to replay to sample the network
# architecture, and also to use for validating the exported network's
# input/output mapping
MODEL_CONVERSION_TEST_SAMPLE_CT: int = 2


class SingleSample(NamedTuple):
    data: List[torch.Tensor]
    target: List[Tuple[torch.Tensor, ...]]
    meta: Dict[str, Any]
    output: List[torch.Tensor]
    metric: Dict[str, float]

    # https://stackoverflow.com/questions/42236820/adding-numpy-array-to-a-heap-queue
    def __lt__(self, b: Any) -> bool:
        return False

    def __gt__(self, b: Any) -> bool:
        return False

    def __eq__(self, b: Any) -> bool:
        return False

    def __ne__(self, b: Any) -> bool:
        return True


JSON = str


class SerializableSampleSummary(NamedTuple):
    """
    SampleSummary includes plotly plots which are not picklable. In order to send
    a SampleSummary from an individual worker process to the main coordinator process
    it must be pickled. Plotly plots *are* json serializable so we convert a
    SampleSummary into this class with the plotly plots pre-serialized to json, making
    this class picklable.
    """

    image: Optional[np.ndarray]
    text: Optional[str]
    plot: Optional[JSON] = None
    source: Optional[str] = None


class FractionalEpochSplitPerformanceSummary(NamedTuple):
    nSamples: int
    losses: Dict[str, float]  # loss name : loss value
    metrics: Dict[str, float]  # metric name : metric value
    samples: List[SerializableSampleSummary]
    worstSamples: List[SerializableSampleSummary]
    testIO: List[SingleSample]


class FractionalPerformanceSummary(NamedTuple):
    epoch: int
    modelBuffer: bytes
    optimizerStateBuffer: bytes
    performance: Dict[Split, FractionalEpochSplitPerformanceSummary]


def stack_recursive(
    items: List[Union[torch.Tensor, Sequence[torch.Tensor]]]
) -> Union[torch.Tensor, List[torch.Tensor]]:
    assert len(items) > 0
    canonical_item = items[0]
    if isinstance(canonical_item, Sequence):
        result = []
        for i in range(len(canonical_item)):
            result.append(torch.stack([item[i] for item in items]))
        return result
    else:
        return torch.stack(cast(List[torch.Tensor], items))


def stat_to_str(stats: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> str:
    """
    key of the element of stats is the metric name
    element value is two arrays, while the first array is the number of samples within
    each bin; the second array is the bin values
    """
    s = ""
    for metric_name, hist_data in stats.items():
        s += "\n" + metric_name + "\n"
        # compute the cumulative number
        bin_cumuls = np.cumsum(hist_data[0])
        bin_values = hist_data[1]
        bin_percentage = [
            b / bin_cumuls[-1] if bin_cumuls[-1] != 0 else 0.0 for b in bin_cumuls
        ]
        s += "".join(
            "{:.3f}: {:.3f}\t".format(bin_values[i], bin_percentage[i])
            for i in range(0, len(bin_cumuls))
        )
    s += "\n"
    return s


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self) -> None:
        self.val: float = 0.0
        self.avg: float = 0.0
        self.sum: float = 0.0
        self.count: int = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EpochTimer:
    def __init__(self) -> None:
        self.batch = AverageMeter()
        self.epoch = AverageMeter()


RawMetas = Dict[str, Union[torch.Tensor, List[Union[str, float]]]]


class SamplerState:
    def __init__(
        self,
        problem: Problem,
        loader: torch.utils.data.DataLoader,
        dataset: MultifieldDataset,
        device: torch.device,
        n_vis: int,
    ) -> None:
        self._problem = problem
        self._device = device
        self._n_samples = len(loader.sampler)

        dataset_fraction = self._n_samples / len(dataset)
        self._n_vis = min(self._n_samples, ceil(n_vis * dataset_fraction))

        self._random_indices = frozenset(
            random.sample(
                range(0, self._n_samples),
                max(self._n_vis, MODEL_CONVERSION_TEST_SAMPLE_CT),
            )
        )
        self._cur_samples = 0

        self._random_samples: List[SingleSample] = []
        self._worst_samples: List[Tuple[float, SingleSample]] = []
        self._rankable_metric, self._ordering = problem.get_rankable_metric()
        self._allow_non_positive_definite = (
            "MSE" not in self._rankable_metric
            and "EucDist" not in self._rankable_metric
        )

        self._running_raw_metas: List[RawMetas] = []
        self._running_data: List[List[torch.Tensor]] = []
        self._running_targets: List[List[Tuple[torch.Tensor, ...]]] = []
        self._running_outputs: List[List[torch.Tensor]] = []

        # pyre-fixme[8]: Attribute has type `DefaultDict[str, ndarray]`; used as `Def...
        self._data_metric: DefaultDict[str, np.ndarray] = defaultdict(list)

    def _cat_metas(self, running_raw_metas: List[RawMetas]) -> RawMetas:
        raw_metas: RawMetas = {}
        for meta_key in running_raw_metas[0]:
            vals = [m[meta_key] for m in running_raw_metas]
            if torch.is_tensor(vals[0]):
                typed_vals = cast(List[torch.Tensor], vals)
                raw_metas[meta_key] = torch.cat(typed_vals)
            elif isinstance(vals[0], list):
                typed_vals = cast(List[List[Union[float, str]]], vals)
                raw_metas[meta_key] = list(itertools.chain(*typed_vals))
            else:
                raise ValueError(
                    "Unsure how to concatenate meta field with type %s" % type(vals[0])
                )
        return raw_metas

    def _get_single_sample(
        self,
        intra_batch_idx: int,
        data: List[torch.Tensor],
        target: List[Tuple[torch.Tensor, ...]],
        output: List[torch.Tensor],
        meta: Any,
        sample_metric: Dict[str, np.ndarray],
    ) -> SingleSample:
        return SingleSample(
            data=[t[intra_batch_idx].cpu() for t in data],
            target=[
                tuple(t[intra_batch_idx].cpu() for t in head_targets)
                for head_targets in target
            ],
            meta={
                k: None if v is None else v[intra_batch_idx]
                for k, v in meta._asdict().items()
            },
            output=[t[intra_batch_idx].cpu() for t in output],
            metric={k: v[intra_batch_idx] for k, v in sample_metric.items()},
        )

    def append_sample(
        self,
        raw_metas: RawMetas,
        data: List[torch.Tensor],
        *,
        outputs: List[torch.Tensor],
        targets: List[Tuple[torch.Tensor, ...]],
    ) -> None:
        self._running_raw_metas.append(raw_metas)
        self._running_data.append([t.detach() for t in data])
        self._running_outputs.append([t.detach() for t in outputs])
        self._running_targets.append(
            [tuple(t.detach() for t in head_targets) for head_targets in targets]
        )

    def compute_metrics(self) -> None:
        if len(self._running_data) == 0:
            return
        meta = self._problem.refine_batch_meta(self._cat_metas(self._running_raw_metas))
        num_maps = len(self._running_outputs[0])

        target: List[Tuple[torch.Tensor, ...]] = []
        for i in range(num_maps):
            num_targets = len(self._running_targets[0][i])
            target.append(
                tuple(
                    torch.cat([t[i][j] for t in self._running_targets])
                    for j in range(num_targets)
                )
            )

        output = [
            torch.cat([o[i] for o in self._running_outputs]) for i in range(num_maps)
        ]

        # _running_data is a List[List[Tensor]]
        # Outer list contains an entry per minibatch
        # Inner list contains an entry per data source
        # Tensor is an M x ... tensor where M is the minibatch size
        data = [
            torch.cat([d[i] for d in self._running_data])
            for i in range(len(self._running_data[0]))
        ]

        sample_metric = self._problem.compute_batch_metrics(
            meta=meta, target=target, output=output, device=self._device
        )
        if sample_metric is not None:
            for k, v in sample_metric.items():
                self._data_metric[k] += list(v)

        if self._n_vis > 0 and sample_metric is not None:
            for intra_batch_idx in range(len(data[0])):
                if (self._cur_samples + intra_batch_idx) in self._random_indices:
                    self._random_samples.append(
                        self._get_single_sample(
                            intra_batch_idx, data, target, output, meta, sample_metric
                        )
                    )
                single_sample_metric = sample_metric[self._rankable_metric][
                    intra_batch_idx
                ]
                if self._allow_non_positive_definite or single_sample_metric >= 0:
                    if self._ordering == Ordering.DESC:
                        single_sample_metric = -single_sample_metric
                    if (
                        len(self._worst_samples) < self._n_vis
                        or single_sample_metric > self._worst_samples[0][0]
                    ):
                        heap_item = (
                            single_sample_metric,
                            self._get_single_sample(
                                intra_batch_idx,
                                data,
                                target,
                                output,
                                meta,
                                sample_metric,
                            ),
                        )
                        if len(self._worst_samples) < self._n_vis:
                            heapq.heappush(self._worst_samples, heap_item)
                        else:
                            heapq.heappushpop(self._worst_samples, heap_item)
        self._cur_samples += len(data[0])

        self._running_raw_metas = []
        self._running_data = []
        self._running_outputs = []
        self._running_targets = []
        torch.cuda.empty_cache()

    @property
    def n_samples(self) -> int:
        return self._n_samples

    @property
    def random_samples(self) -> List[SingleSample]:
        return self._random_samples

    @property
    def worst_samples(self) -> List[SingleSample]:
        return [s for _, s in self._worst_samples]

    @property
    def data_metric(self) -> Dict[str, np.ndarray]:
        return self._data_metric


class SolverWorker:
    def __init__(
        self,
        model: Union[torch.nn.Module, torch.nn.parallel.DistributedDataParallel],
        criterion: BaseParallelCriterion,
        optimizer,
        device: torch.device,
        run_opts: RunOpts,
        cache: SharedMemoryDatasetCache,
        *,
        local_rank: int,
        node_idx: int,
        node_count: int,
    ) -> None:
        self.model = model
        self.device = device
        self.run_opts = run_opts
        self.accessor = CachedDatasetAccessor(cache, process_idx=local_rank)

        self.criterion = criterion
        self.optimizer = optimizer
        self.optimizer.zero_grad()
        self.cur_epoch = 0

        # Turn on auto tune and verbose
        cudnn.fastest = True
        cudnn.benchmark = True
        cudnn.verbose = True

        self._node_idx = node_idx
        self._node_count = node_count

    # if mode is Train, then this takes one pass over each of the dataset,
    # and update models using dataset 'training', if mode is eval, then it go
    # though all dataset without updating model.
    def _pass_one_epoch(
        self,
        problem: Problem,
        loaders: Dict[Split, torch.utils.data.DataLoader],
        mode: Mode,
    ) -> Dict[Split, FractionalEpochSplitPerformanceSummary]:
        epoch_stats: Dict[Split, FractionalEpochSplitPerformanceSummary] = {}

        num_tasks = len(self.criterion.loss_names)
        avg_grad_contributions = [0.0] * num_tasks
        contribution_count = 0
        minibatch_timeout_ms: int = Config.get().read_key(  # pyre-ignore
            CONFIG_MINIBATCH_TIMEOUT_MS
        )

        for data_type, loader in loaders.items():
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                loader.sampler.set_epoch(self.cur_epoch)

            self.accessor.set_sequence_indices(list(iter(loader.sampler)))
            loader.dataset.set_accessor(self.accessor)

            logger.info("Starting split %s" % data_type.name)

            # Set NN module modes
            if (mode == Mode.TRAIN) and (data_type == Split.TRAIN):
                self.model.train()
                self.criterion.train()
            else:
                self.model.eval()
                self.criterion.eval()

            split_loss: DefaultDict[str, List[float]] = defaultdict(list)
            # record elasped time
            timer = EpochTimer()
            epoch_start = time.time()
            # ================== Start training for one epoch ======================
            dataset = [d for d in problem.datasets if d.data_type == data_type][0]
            sampler_state = SamplerState(
                problem,
                loader,
                dataset,
                self.device,
                max(
                    self.run_opts.numVisualizedSamples, MODEL_CONVERSION_TEST_SAMPLE_CT
                ),
            )

            batch_start = time.time()
            # pyre-ignore DataLoader doesn't explicitly extend Iterator[] yet.
            for minibatch_idx, (data, target, raw_meta) in enumerate(loader):
                with WatchdogTimer.create(minibatch_timeout_ms):
                    # Move data to device
                    data = [data_tensor.to(self.device) for data_tensor in data]
                    target = [
                        tuple(t.to(self.device) for t in head_targets)
                        for head_targets in target
                    ]

                    output, total_loss, sub_loss, grad_norms = self._pass_one_minibatch(
                        minibatch_idx, data_type, data, target
                    )

                    if grad_norms is not None:
                        contribution_count += 1
                        for i in range(num_tasks):
                            avg_grad_contributions[i] = (
                                avg_grad_contributions[i]
                                * ((contribution_count - 1) / contribution_count)
                                + grad_norms[i] / contribution_count
                            )

                    # now compute metrics defined by the problem
                    with torch.no_grad():
                        for k, v in sub_loss.items():
                            split_loss[k].append(v.item())

                        if (
                            minibatch_idx % self.run_opts.metricAmortizationSchedule
                            == 0
                        ):
                            sampler_state.compute_metrics()

                        sampler_state.append_sample(
                            raw_meta, data, outputs=output, targets=target
                        )

                        if (
                            self.run_opts.lossLoggingFreq > 0
                            and minibatch_idx % self.run_opts.lossLoggingFreq == 0
                        ):
                            self._summarize_times(dataset.data_type, timer)
                            losses = {x: y.item() for x, y in sub_loss.items()}
                            losses["total_loss"] = total_loss.item()
                            loss_str = json.dumps(losses)
                            logger.info("{}: {}".format(minibatch_idx, loss_str))
                    timer.batch.update(time.time() - batch_start)
                    batch_start = time.time()

            sampler_state.compute_metrics()
            # ================== End of one epoch for a dataset ===============
            timer.epoch.update(time.time() - epoch_start)

            epoch_stats[data_type] = self._epoch_summary(
                problem, sampler_state, dataset, split_loss, timer, mode
            )
        # ======================= End of iterating through all datasets ========

        if self.run_opts.debugGrad:
            logger.info(
                "Task grad contributions: "
                + ", ".join(
                    "%s: %f" % (loss_name, grad_contrib)
                    for loss_name, grad_contrib in zip(
                        self.criterion.loss_names, avg_grad_contributions
                    )
                )
            )

        return epoch_stats

    def _pass_one_minibatch(
        self,
        minibatch_idx: int,
        data_type: Split,
        data: Sequence[torch.Tensor],
        target: Sequence[Tuple[torch.Tensor, ...]],
    ) -> Tuple[
        List[torch.Tensor], torch.Tensor, Dict[str, torch.Tensor], Optional[List[float]]
    ]:
        # This incurs some extra computation so we don't do this for every
        # minibatch. We can get meaningful statistics even so.
        debug_grad = (
            self.run_opts.debugGrad
            and isinstance(self.model, MultiTaskModel)
            and minibatch_idx % 10 == 0
        )

        # model forward
        output = self.model(data)

        final_shared_params = None
        if debug_grad or isinstance(self.criterion, GradNormWeightedCriterion):
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                # pyre-fixme[16]: `Module` has no attribute `module`.
                final_shared_params = self.model.module.final_shared_params(output)
            else:
                # pyre-fixme[16]: `Module` has no attribute `final_shared_params`.
                final_shared_params = self.model.final_shared_params(output)
            if isinstance(self.criterion, GradNormWeightedCriterion):
                # pyre-fixme[16]: `BaseParallelCriterion` has no attribute
                #  `set_shared_params`.
                self.criterion.set_shared_params(final_shared_params)

        # loss forward
        total_loss, sub_loss = self.criterion(output, target)

        if torch.isnan(total_loss).any():
            raise FloatingPointError(
                "Losses become NaN for dataset {} at iteration {} "
                "minibatch {}!".format(data_type.value, self.cur_epoch, minibatch_idx)
            )

        # back-prop for training
        grad_norms: Optional[List[float]] = None
        if self.model.training:
            if debug_grad:
                grad_norms = [
                    torch.autograd.grad(loss, final_shared_params, retain_graph=True)[0]
                    .norm()
                    .item()
                    for loss in sub_loss.values()
                ]
            self.optimizer.zero_grad()
            total_loss.backward()

            if self.run_opts.optim.gradientClip:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.run_opts.optim.gradientClip
                )
            self.optimizer.step()

        return output, total_loss, sub_loss, grad_norms

    def _refine_sample_set(
        self, problem: Problem, samples: List[SingleSample]
    ) -> Tuple[
        List[torch.Tensor],
        List[Tuple[torch.Tensor, ...]],
        Any,
        List[torch.Tensor],
        BatchMetrics,
    ]:
        assert len(samples) > 0, "Can't refine an empty sample set"
        canonical_sample = samples[0]
        target = []
        for i in range(len(canonical_sample.target)):
            target.append(
                stack_recursive(
                    [
                        tuple(t.detach().cpu() for t in sample.target[i])
                        for sample in samples
                    ]
                )
            )
        raw_meta = {}
        for key in canonical_sample.meta:
            val = [sample.meta[key] for sample in samples]
            packed_val: Union[List[Any], torch.Tensor]
            if torch.is_tensor(val[0]):
                packed_val = torch.stack(val)
            else:
                packed_val = val
            raw_meta[key] = packed_val
        meta = problem.refine_batch_meta(raw_meta)
        output = []
        for i in range(len(canonical_sample.output)):
            output.append(torch.stack([sample.output[i] for sample in samples]).cpu())
        data = []
        for i in range(len(canonical_sample.data)):
            data.append(torch.stack([sample.data[i] for sample in samples]).cpu())
        metric = {}
        for key in canonical_sample.metric:
            metric[key] = np.array([sample.metric[key] for sample in samples])
        return data, target, meta, output, metric

    @staticmethod
    def _serialize_sample_summaries(
        summaries: Sequence[SampleSummary]
    ) -> List[SerializableSampleSummary]:
        return [
            SerializableSampleSummary(
                image=s.image,
                text=s.text,
                plot=json.dumps(s.plot) if s.plot else None,
                source=s.source,
            )
            for s in summaries
        ]

    def _summarize_times(self, split: Split, timer: EpochTimer) -> None:
        logger.info(
            "<{}>\tEpoch: {}\t"
            "Avg Batch Time: {batch_time.avg:.3f}\t"
            "Epoch Time: {epoch_time.sum: .3f}".format(
                split.value.upper(),
                self.cur_epoch,
                batch_time=timer.batch,
                epoch_time=timer.epoch,
            )
        )

    def _epoch_summary(
        self,
        problem: Problem,
        sampler: SamplerState,
        dataset: MultifieldDataset,
        split_loss: Dict[str, List[float]],
        timer: EpochTimer,
        mode: Mode,
    ) -> FractionalEpochSplitPerformanceSummary:
        self._summarize_times(dataset.data_type, timer)
        logger.info(
            "<{}>\tEpoch: {}\tLearning rate: {}\t".format(
                dataset.data_type.value.upper(),
                self.cur_epoch,
                self.optimizer.param_groups[0]["lr"],
            )
        )
        epoch_split_loss = {k: np.mean(v) for k, v in split_loss.items()}
        logger.info(
            "<{}>\tEpoch: {}\t{}".format(
                dataset.data_type.value.upper(),
                self.cur_epoch,
                utils.dict_to_str(epoch_split_loss),
            )
        )
        epoch_data_metric = problem.summarize_epoch_metrics(sampler.data_metric)
        logger.info(
            "<{}>\tEpoch: {}\t{}".format(
                dataset.data_type.value.upper(),
                self.cur_epoch,
                utils.dict_to_str(epoch_data_metric),
            )
        )

        # =============================== Visualizations =================== #
        summary = FractionalEpochSplitPerformanceSummary(
            nSamples=sampler.n_samples,
            losses=epoch_split_loss,
            metrics=epoch_data_metric,
            samples=[],
            worstSamples=[],
            testIO=sampler.random_samples[:MODEL_CONVERSION_TEST_SAMPLE_CT],
        )
        if self.run_opts.numVisualizedSamples == 0:
            return summary

        sample_summaries: List[SampleSummary] = []
        if sampler.random_samples:
            sample_summaries = problem.summarize_epoch_samples(
                # pyre-fixme[6]: Expected `List[Tensor]` for 1st param but got
                #  `Union[Any, Dict[str, ndarray], List[Tensor], List[Tuple[Tensor,
                #  ...]]]`.
                *self._refine_sample_set(problem, sampler.random_samples)
            )

        worst_summaries: List[SampleSummary] = []
        if sampler.worst_samples:
            worst_summaries = problem.summarize_epoch_samples(
                # pyre-fixme[6]: Expected `List[Tensor]` for 1st param but got
                #  `Union[Any, Dict[str, ndarray], List[Tensor], List[Tuple[Tensor,
                #  ...]]]`.
                *self._refine_sample_set(problem, sampler.worst_samples)
            )

        return summary._replace(
            samples=self._serialize_sample_summaries(sample_summaries),
            worstSamples=self._serialize_sample_summaries(worst_summaries),
        )

    def _get_worker_performance_summary(
        self, epoch_stats: Dict[Split, FractionalEpochSplitPerformanceSummary]
    ) -> FractionalPerformanceSummary:
        # save things at eval mode
        model = (
            # pyre-fixme[16]: `Module` has no attribute `module`.
            self.model.module
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel)
            else self.model
        ).eval()

        # We save these to buffers instead of just pickling them because they can
        # contain cuda buffers which the host cpu process is incapable of
        # reconstructing (since it doesnt necessarily have a cuda context).
        # `torch.save` makes them device agnostic.
        with io.BytesIO() as model_buffer:
            torch.save(model, model_buffer)
            model_buffer_bytes = model_buffer.getvalue()

        with io.BytesIO() as optimizer_state_buffer:
            torch.save(self.optimizer.state_dict(), optimizer_state_buffer)
            optimizer_state_buffer_bytes = optimizer_state_buffer.getvalue()

        return FractionalPerformanceSummary(
            epoch=self.cur_epoch,
            modelBuffer=model_buffer_bytes,
            optimizerStateBuffer=optimizer_state_buffer_bytes,
            performance=epoch_stats,
        )

    def train(
        self,
        problem: Problem,
        startEpoch: int,
        nEpochs: int,
        batchSize: int,
        scheduler: _LRScheduler,
    ) -> Iterator[FractionalPerformanceSummary]:
        self.cur_epoch = startEpoch

        d_types = [d.data_type for d in problem.datasets]
        assert Split.TRAIN in d_types, "training dataset should be included"

        ################### Start Training #####################################

        logger.info("Model layers")
        logger.info(str(self.model.modules))

        loaders = self._get_loaders(problem, batchSize=batchSize)

        # start epochs
        while self.cur_epoch < nEpochs:
            self.cur_epoch += 1
            logger.info("Starting epoch %d" % self.cur_epoch)
            epoch_stats = self._pass_one_epoch(problem, loaders, Mode.TRAIN)
            logger.info("Finished epoch %d" % self.cur_epoch)

            scheduler.step()

            yield self._get_worker_performance_summary(epoch_stats)

    def eval(
        self, problem: Problem, batchSize: int
    ) -> Iterator[FractionalPerformanceSummary]:
        logger.info("Model layers:")
        logger.info(str(self.model.modules))

        loaders = self._get_loaders(problem, batchSize=batchSize)
        epoch_stats = self._pass_one_epoch(problem, loaders, Mode.EVAL)

        yield self._get_worker_performance_summary(epoch_stats)

    def _get_loaders(
        self, problem: Problem, batchSize: int
    ) -> Dict[Split, torch.utils.data.DataLoader]:
        loaders = {}
        for dataset in problem.datasets:
            sampler = None
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                sampler = ScaffoldSampler(
                    dataset,
                    shuffle_type=self.run_opts.shuffleType,
                    node_idx=self._node_idx,
                    node_count=self._node_count,
                )
            ds_data_type = dataset.data_type
            if self.run_opts.maxEpochImages > 0:
                logger.info("Using %d images per epoch." % self.run_opts.maxEpochImages)
                dataset = SubsetMultifieldDataset(
                    dataset, range(self.run_opts.maxEpochImages)
                )
            loaders[ds_data_type] = torch.utils.data.DataLoader(
                dataset,
                batch_size=batchSize,
                shuffle=sampler is None,
                num_workers=self.run_opts.numThreads,
                pin_memory=self.device == Device.GPU,
                sampler=sampler,
            )
        return loaders
