from typing import Tuple

import numpy as np
import scipy
import torch
import torch.nn.functional as F
from torch import Tensor


def beta_warping(
    x, alpha: float = 1.0, eps: float = 1e-12, inverse_cdf: bool = False
) -> float:
    if inverse_cdf:
        return scipy.stats.beta.ppf(x, a=alpha + eps, b=alpha + eps)
    else:
        return scipy.stats.beta.cdf(x, a=alpha + eps, b=alpha + eps)


def sim_gauss_kernel(
    dist, tau_max: float = 1.0, tau_std: float = 0.5, inverse_cdf=False
) -> float:
    dist = dist / np.mean(dist)
    dist_rate = tau_max * np.exp(-(dist - 1) / (2 * tau_std * tau_std))
    if inverse_cdf:
        return dist_rate
    else:
        return 1 / (dist_rate + 1e-12)


# def tensor_linspace(start: Tensor, stop: Tensor, num: int):
#     """
#     Creates a tensor of shape [num, *start.shape] whose values are evenly
#     spaced from start to end, inclusive.
#     Replicates but the multi-dimensional bahaviour of numpy.linspace in PyTorch.
#     """
#     # create a tensor of 'num' steps from 0 to 1
#     steps = torch.arange(num, dtype=torch.float32, device=start.device) / (
#         num - 1
#     )

#     # reshape the 'steps' tensor to [-1, *([1]*start.ndim)]
#     # to allow for broadcastings
#     # using 'steps.reshape([-1, *([1]*start.ndim)])' would be nice here
#     # but torchscript
#     # "cannot statically infer the expected size of a list in this contex",
#     # hence the code below
#     for i in range(start.ndim):
#         steps = steps.unsqueeze(-1)

#     # the output starts at 'start' and increments until 'stop' in each dimension
#     out = start[None] + steps * (stop - start)[None]

#     return out


# def torch_beta_cdf(
#     x: Tensor, c1: Tensor | float, c2: Tensor | float, npts=100, eps=1e-12
# ):
#     if isinstance(c1, float):
#         if c1 == c2:
#             c1 = Tensor([c1], device=x.device)
#             c2 = c1
#         else:
#             c1 = Tensor([c1], device=x.device)
#     if isinstance(c2, float):
#         c2 = Tensor([c2], device=x.device)
#     bt = torch.distributions.Beta(c1, c2)

#     if isinstance(x, float):
#         x = Tensor(x)

#     X = tensor_linspace(torch.zeros_like(x) + eps, x, npts)
#     return torch.trapezoid(bt.log_prob(X).exp(), X, dim=0)


# def torch_beta_warping(
#     x: Tensor, alpha_cdf: float | Tensor = 1.0, eps=1e-12, npts=100
# ):
#     return torch_beta_cdf(
#         x=x, c1=alpha_cdf + eps, c2=alpha_cdf + eps, npts=npts, eps=eps
#     )


# def torch_sim_gauss_kernel(dist: Tensor, tau_max=1.0, tau_std=0.5):
#     dist_rate = tau_max * torch.exp(
#         -(dist - 1) / (torch.mean(dist) * 2 * tau_std * tau_std)
#     )

#     return 1 / (dist_rate + 1e-12)


class AbstractMixup:
    def __init__(
        self, alpha: float = 1.0, mode: str = "batch", num_classes: int = 1000
    ) -> None:
        self.alpha = alpha
        self.num_classes = num_classes
        self.mode = mode

    def _get_params(self, batch_size: int, device: torch.device):
        if self.mode == "batch":
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = torch.tensor(
                np.random.beta(self.alpha, self.alpha, batch_size),
                device=device,
            )
        index = torch.randperm(batch_size, device=device)
        return lam, index

    def _linear_mixing(
        self,
        lam: Tensor | float,
        inp: Tensor,
        index: Tensor,
    ) -> Tensor:
        if isinstance(lam, Tensor):
            lam = lam.view(-1, *[1 for _ in range(inp.ndim - 1)]).float()

        return lam * inp + (1 - lam) * inp[index, :]

    def _mix_target(
        self,
        lam: Tensor | float,
        target: Tensor,
        index: Tensor,
    ) -> Tensor:
        y1 = F.one_hot(target, self.num_classes)
        y2 = F.one_hot(target[index], self.num_classes)
        if isinstance(lam, Tensor):
            lam = lam.view(-1, *[1 for _ in range(y1.ndim - 1)]).float()

        if isinstance(lam, Tensor) and lam.dtype == torch.bool:
            return lam * y1 + (~lam) * y2
        else:
            return lam * y1 + (1 - lam) * y2

    def __call__(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        return x, y


class Mixup(AbstractMixup):
    def __call__(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        lam, index = self._get_params(x.size()[0], x.device)
        mixed_x = self._linear_mixing(lam, x, index)
        mixed_y = self._mix_target(lam, y, index)
        return mixed_x, mixed_y


class MixupIO(AbstractMixup):
    def __call__(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        lam, index = self._get_params(x.size()[0], x.device)

        mixed_x = self._linear_mixing(lam, x, index)

        mixed_y = self._mix_target((lam > 0.5), y, index)

        return mixed_x, mixed_y


class MixupTO(AbstractMixup):
    def __call__(
        self, x: Tensor, y: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, float | Tensor]:
        lam, index = self._get_params(x.size()[0], x.device)

        mixed_y = self._mix_target(lam, y, index)

        return x, x[index], mixed_y, lam


class RegMixup(AbstractMixup):
    def __call__(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        lam, index = self._get_params(x.size()[0], x.device)
        part_x = self._linear_mixing(lam, x, index)
        part_y = self._mix_target(lam, y, index)
        mixed_x = torch.cat([x, part_x], dim=0)
        mixed_y = torch.cat([F.one_hot(y, self.num_classes), part_y], dim=0)
        return mixed_x, mixed_y


class MITMixup(AbstractMixup):
    def __init__(
        self,
        margin: float = 0.0,
        alpha: float = 1.0,
        mode: str = "batch",
        num_classes: int = 1000,
    ) -> None:
        super().__init__(alpha, mode, num_classes)
        self.margin = margin

    def _get_params(self, batch_size: int, device: torch.device):
        if self.mode == "batch":
            lam1 = np.random.beta(self.alpha, self.alpha)
            lam1 = max(lam1, 1 - lam1)
            lam2 = np.random.beta(self.alpha, self.alpha)
            lam2 = min(lam2, 1 - lam2)

            while abs(lam1 - lam2) < self.margin:
                lam1 = np.random.beta(self.alpha, self.alpha)
                lam1 = max(lam1, 1 - lam1)
                lam2 = np.random.beta(self.alpha, self.alpha)
                lam2 = min(lam2, 1 - lam2)

        else:
            lam1 = torch.tensor(
                np.random.beta(self.alpha, self.alpha, batch_size),
                device=device,
            )
            lam1 = torch.max(lam1, 1 - lam1)
            lam2 = torch.tensor(
                np.random.beta(self.alpha, self.alpha, batch_size),
                device=device,
            )
            lam2 = torch.min(lam2, 1 - lam2)

            while torch.abs(lam1 - lam2) < self.margin:
                lam1 = torch.tensor(
                    np.random.beta(self.alpha, self.alpha, batch_size),
                    device=device,
                )
                lam1 = torch.max(lam1, 1 - lam1)
                lam2 = torch.tensor(
                    np.random.beta(self.alpha, self.alpha, batch_size),
                    device=device,
                )
                lam2 = torch.min(lam2, 1 - lam2)

        index = torch.randperm(batch_size, device=device)
        return lam1, lam2, index

    def __call__(
        self, x: Tensor, y: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, float | Tensor, float | Tensor]:
        lam1, lam2, index = self._get_params(x.size()[0], x.device)
        mixed_x1 = self._linear_mixing(lam1, x, index)
        mixed_x2 = self._linear_mixing(lam2, x, index)
        return mixed_x1, mixed_x2, y, y[index], lam1, lam2


class QuantileMixup(AbstractMixup):
    def __init__(
        self,
        alpha=1.0,
        mode="batch",
        num_classes=1000,
        lower=False,
        quantile=0.5,
        warp=True,
    ) -> None:
        super().__init__(alpha, mode, num_classes)
        self.quantile = quantile
        self.warp = warp
        if self.warp:
            if lower:
                self.comp_func = np.less_equal
            else:
                self.comp_func = np.greater
        else:
            if lower:
                self.comp_func = torch.less_equal
            else:
                self.comp_func = torch.greater

    def _get_params(self, batch_size: int, device: torch.device):
        if self.mode == "batch":
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = np.random.beta(self.alpha, self.alpha, batch_size)

        index = torch.randperm(batch_size, device=device)

        return lam, index

    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        feats: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lam, index = self._get_params(x.size()[0], x.device)

        if self.warp:
            l2_dist = (
                (feats - feats[index])
                .pow(2)
                .sum([i for i in range(len(feats.size())) if i > 0])
                .cpu()
                .numpy()
            )

            quant = np.quantile(l2_dist, self.quantile)

            k_lam = torch.tensor(
                beta_warping(lam, 1 / (self.comp_func(l2_dist, quant) + 1e-12)),
                device=x.device,
            )

            mixed_x = self._linear_mixing(k_lam, x, index)

            mixed_y = self._mix_target(k_lam, y, index)
        else:
            pdist_mat = torch.cdist(feats, feats)

            quant = torch.quantile(pdist_mat, self.quantile)

            filter = torch.logical_or(
                self.comp_func(pdist_mat, quant),
                torch.eye(pdist_mat.size(0), device=pdist_mat.device),
            )

            new_index = torch.cat(
                [
                    filter[i]
                    .nonzero()
                    .squeeze(-1)[
                        torch.randint(0, len(filter[i].nonzero()), (1,))
                    ]
                    for i in range(len(filter))
                ]
            )

            mixed_x = self._linear_mixing(
                torch.tensor(lam, device=x.device), x, new_index
            )

            mixed_y = self._mix_target(
                torch.tensor(lam, device=x.device), y, new_index
            )

        return mixed_x, mixed_y


class WarpingMixup(AbstractMixup):
    def __init__(
        self,
        alpha: float = 1.0,
        mode: str = "batch",
        num_classes: int = 1000,
        apply_kernel: bool = True,
        warping: str = "beta_cdf",
        tau_max: float = 1.0,
        tau_std: float = 0.5,
        manifold: bool = False,
        regularization: bool = False,
    ) -> None:
        super().__init__(alpha, mode, num_classes)
        self.apply_kernel = apply_kernel
        self.tau_max = tau_max
        self.tau_std = tau_std
        self.manifold = manifold
        self.regularization = regularization
        self.warping = warping

    def _get_params(self, batch_size: int, device: torch.device):
        if self.mode == "batch":
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = np.random.beta(self.alpha, self.alpha, batch_size)

        index = torch.randperm(batch_size, device=device)
        return lam, index

    def __call__(
        self,
        x: Tensor,
        y: Tensor,
        feats: Tensor,
        warp_param=1.0,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor] | Tuple[Tensor, Tensor]:
        lam, index = self._get_params(x.size()[0], x.device)

        if self.apply_kernel:
            l2_dist = (
                (feats - feats[index])
                .pow(2)
                .sum([i for i in range(len(feats.size())) if i > 0])
                .cpu()
                .numpy()
            )
            if self.warping == "inverse_beta_cdf":
                warp_param = sim_gauss_kernel(
                    l2_dist, self.tau_max, self.tau_std, inverse_cdf=True
                )
            elif self.warping == "beta_cdf":
                warp_param = sim_gauss_kernel(
                    l2_dist, self.tau_max, self.tau_std
                )
            else:
                raise NotImplementedError()

        if self.warping == "inverse_beta_cdf":
            k_lam = torch.tensor(
                beta_warping(lam, warp_param, inverse_cdf=True), device=x.device
            )
        elif self.warping == "beta_cdf":
            k_lam = torch.tensor(
                beta_warping(lam, warp_param, inverse_cdf=False),
                device=x.device,
            )
        else:
            raise NotImplementedError()

        if self.manifold:
            mixed_y = self._mix_target(k_lam, y, index)
            return x, x[index], mixed_y, k_lam
        elif self.regularization:
            part_x = self._linear_mixing(k_lam, x, index)
            part_y = self._mix_target(k_lam, y, index)
            mixed_x = torch.cat([x, part_x], dim=0)
            mixed_y = torch.cat([F.one_hot(y, self.num_classes), part_y], dim=0)
            return mixed_x, mixed_y
        else:
            mixed_x = self._linear_mixing(k_lam, x, index)
            mixed_y = self._mix_target(k_lam, y, index)
            return mixed_x, mixed_y
