from typing import Any, List

import torch
import numpy as np
from torchmetrics import Metric
from torchmetrics.utilities.data import dim_zero_cat


class AdaptiveCalibrationError(Metric):
    r"""Compute Adaptive ECE metric.

    Args:
        Metric (_type_): _description_
    """

    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False

    confidences: List[torch.Tensor]
    accuracies: List[torch.Tensor]

    def __init__(self, n_bins: int = 15, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.nbins = n_bins
        self.add_state("confidences", [], dist_reduce_fx="cat")
        self.add_state("accuracies", [], dist_reduce_fx="cat")

    def histedges_equalN(self, x):
        npt = len(x)
        return np.interp(
            np.linspace(0, npt, self.nbins + 1), np.arange(npt), np.sort(x)
        )

    def update(self, probs: torch.Tensor, targets: torch.Tensor) -> None:
        confidences, preds = torch.max(probs, 1)
        accuracies = preds.eq(targets)
        self.confidences.append(confidences)
        self.accuracies.append(accuracies)

    def compute(self) -> torch.Tensor:
        confidences = dim_zero_cat(self.confidences)
        accuracies = dim_zero_cat(self.accuracies)

        n, bin_boundaries = np.histogram(
            confidences.cpu().detach(),
            self.histedges_equalN(confidences.cpu().detach()),
        )

        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

        ece = torch.zeros(1, device=confidences.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(
                bin_upper.item()
            )
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += (
                    torch.abs(avg_confidence_in_bin - accuracy_in_bin)
                    * prop_in_bin
                )

        return ece
