import collections.abc
from typing import Any, Iterator, Optional, Tuple, Dict

import numpy as np
from skimage.measure import label
from skimage.segmentation import find_boundaries
import torch
from numba import njit


# https://numba.pydata.org/numba-examples/examples/density_estimation/histogram/results.html
# https://github.com/numba/numba-examples/blob/master/examples/density_estimation/histogram/gpu.py
# https://www.boost.org/doc/libs/1_78_0/libs/histogram/doc/html/index.html


class Metrics(collections.abc.Mapping):
    def __init__(self, probs: np.ndarray,
                 labels: Optional[np.ndarray] = None,
                 gt_labels: Optional[np.ndarray] = None,
                 num_classes: Optional[int] = None,
                 ignore_index: Optional[int] = None,
                 ) -> None:

        self.probs = probs
        self.ignore_index = ignore_index

        if labels is None:
            self.labels = probs.argmax(0)
        else:
            self.labels = labels

        if num_classes is None:
            self.num_classes = probs.shape[0]
        else:
            self.num_classes = num_classes

        if (gt_labels is not None) and (ignore_index is not None):
            self.labels[gt_labels == ignore_index] = ignore_index

        self.segments, self.num_segments = label(self.labels, background=0, return_num=True)
        self.segments = self.segments.astype(np.uint16)

        self.boundary_mask = find_boundaries(self.labels + 1, connectivity=2)

        self.metrics: Dict[str, np.ndarray] = {}

        self._compute_base_stats()
        self.add_probabilities(probs)
        if gt_labels is not None:
            self.add_adjusted_ious(gt_labels)

    def _compute_base_stats(self) -> None:
        n, n_bd, n_in, cl, x, y = _base_stats(self.segments, self.boundary_mask, self.labels, self.num_segments).T
        self.metrics["S"] = n
        self.metrics["S_bd"] = n_bd
        self.metrics["S_in"] = n_in
        self.metrics["S_rel"] = n / n_bd
        self.metrics["S_rel_in"] = n_in / n_bd
        self.metrics["class"] = cl
        self.metrics["x"] = x / n
        self.metrics["y"] = y / n

    def add_heatmap(self, heatmap: np.ndarray, prefix: str) -> None:
        n_tot = self.metrics["S"]
        n_bd = self.metrics["S_bd"]
        n_in = self.metrics["S_in"]
        n_rel = self.metrics["S_rel"]
        n_rel_in = self.metrics["S_rel_in"]

        sum_tot, sum_sq_tot, sum_bd, sum_sq_bd = _heatmap_stats(
            self.segments, self.boundary_mask, heatmap, self.num_segments
        ).T
        sum_in = sum_tot - sum_bd
        sum_sq_in = sum_sq_tot - sum_sq_bd

        self.metrics[prefix] = mean = sum_tot / n_tot
        self.metrics[f"{prefix}_var"] = var = sum_sq_tot / n_tot - mean**2
        self.metrics[f"{prefix}_in"] = mean_in = _safe_divide(sum_in, n_in)
        self.metrics[f"{prefix}_var_in"] = var_in = _safe_divide(sum_sq_in, n_in) - mean_in**2
        self.metrics[f"{prefix}_bd"] = mean_bd = sum_bd / n_bd
        self.metrics[f"{prefix}_var_bd"] = sum_sq_bd / n_bd - mean_bd**2
        self.metrics[f"{prefix}_rel"] = sum_tot / n_bd  # mean * n_rel
        self.metrics[f"{prefix}_var_rel"] = var * n_rel
        self.metrics[f"{prefix}_rel_in"] = sum_in / n_bd  # mean_in * n_rel_in
        self.metrics[f"{prefix}_var_rel_in"] = var_in * n_rel_in

    def add_probabilities(self, probs: np.ndarray, prefix: Optional[str] = None) -> None:
        prefix = "" if prefix is None else f"{prefix}_"

        segment_probs = _prob_stats(probs, self.segments, self.num_segments)
        for c, p in enumerate(segment_probs):
            self.metrics[f"{prefix}cprob_{c}"] = p / self.metrics["S"]

        probs_torch = torch.from_numpy(probs)
        e = _entropy(probs_torch)
        self.add_heatmap(e.numpy(), f"{prefix}E")
        v, m = _prob_differences(probs_torch)
        self.add_heatmap(v.numpy(), f"{prefix}V")
        self.add_heatmap(m.numpy(), f"{prefix}M")

    def add_adjusted_ious(self, gt_labels: np.ndarray) -> None:
        gt_segments = label(gt_labels, background=-1).astype(np.uint16)
        iou = _adjusted_ious(
            self.labels,
            self.segments,
            gt_labels,
            gt_segments,
            self.num_classes,
            self.ignore_index,
        )
        self.metrics["iou"] = iou
        self.metrics["iou0"] = iou == 0

    def __getitem__(self, name: str) -> Any:
        return self.metrics[name][1:]

    def __len__(self) -> int:
        return len(self.metrics)

    def __iter__(self) -> Iterator:
        return iter(self.metrics)


def _safe_divide(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return np.divide(p, q, out=np.zeros_like(p), where=q != 0)


@njit
def _split_by_ids(x: np.ndarray, y: np.ndarray, ids: np.ndarray) -> np.ndarray:
    return (y.reshape(1, -1) == ids.reshape(-1, 1)) * x.reshape(1, -1)


@njit
def _faster_unique(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(x)
    uniq = np.nonzero(counts)[0]
    counts = counts[uniq]
    return uniq, counts


@njit
def _base_stats(segments: np.ndarray, boundary_mask: np.ndarray, labels: np.ndarray, num_segments: np.ndarray):
    stats = np.zeros((num_segments + 1, 6), np.uint32)
    stats[0, 0] = stats[0, 1] = 1
    for i in range(segments.shape[0]):
        for j in range(segments.shape[1]):
            stats[segments[i, j], 0] += 1  # n
            stats[segments[i, j], 1] += boundary_mask[i, j]  # n_bd
            stats[segments[i, j], 2] += ~boundary_mask[i, j]  # n_in
            stats[segments[i, j], 3] = labels[i, j]  # cl
            stats[segments[i, j], 4] += j  # x
            stats[segments[i, j], 5] += i  # y
    return stats


@njit
def _heatmap_stats(segments: np.ndarray, boundary_mask: np.ndarray, heatmap: np.ndarray, num_segments: np.ndarray):
    segments = segments.ravel()
    boundary_mask = boundary_mask.ravel()
    heatmap = heatmap.ravel()
    stats = np.zeros((num_segments + 1, 4), heatmap.dtype)
    for i in range(len(segments)):
        h = heatmap[i]
        h_sq = h * h
        stats[segments[i], 0] += h
        stats[segments[i], 1] += h_sq
        stats[segments[i], 2] += h * boundary_mask[i]
        stats[segments[i], 3] += h_sq * boundary_mask[i]
    return stats


@njit
def _prob_stats(probs: np.ndarray, segments: np.ndarray, num_segments: int):
    segment_probs = np.zeros((probs.shape[0], num_segments + 1), dtype=np.float32)
    for i in range(probs.shape[0]):
        for j in range(probs.shape[1]):
            for k in range(probs.shape[2]):
                segment_probs[i, segments[j, k]] += probs[i, j, k]
    return segment_probs


def _entropy(probs: torch.Tensor) -> torch.Tensor:
    entropy_unnormalized = torch.sum(-probs * torch.log(probs), dim=0)
    entropy = torch.div(entropy_unnormalized, torch.log(torch.tensor(probs.shape[0], dtype=torch.float32)))
    return entropy


def _prob_differences(probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    largest = torch.topk(probs, 2, dim=0).values
    v = 1 - largest[1]  # variation ratio
    m = v + largest[0]  # probability margin
    return v, m


@njit
def _adjusted_ious(pr_labels: np.ndarray,
                   pr_segments: np.ndarray,
                   gt_labels: np.ndarray,
                   gt_segments: np.ndarray,
                   num_classes: int,
                   ignore_index: Optional[int] = None,
                   ) -> np.ndarray:
    ids = np.arange(num_classes)
    if ignore_index is not None:
        ids = ids[ids != ignore_index]

    pr_split = _split_by_ids(pr_segments, pr_labels, ids).ravel()
    gt_split = _split_by_ids(gt_segments, gt_labels, ids).ravel()

    n_pr = pr_segments.max() + 1
    n_gt = gt_segments.max() + 1
    mult = max(n_pr, n_gt)

    uniq, counts = _faster_unique(pr_split + gt_split * np.uint32(mult))

    div_, mod_ = np.divmod(uniq, mult)
    union = np.bincount(mod_, weights=counts, minlength=n_pr)
    mask = div_ != 0
    inter = np.bincount(mod_[mask], weights=counts[mask], minlength=n_pr)

    counts[mod_ != 0] = 0
    union += np.bincount(mod_[mask], weights=counts[np.searchsorted(uniq, div_[mask] * mult)], minlength=n_pr)
    union[union == 0] = np.nan
    return inter / union
