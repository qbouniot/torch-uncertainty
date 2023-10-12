import scipy
import torch
import torch.nn.functional as F
from torch import Tensor

import numpy as np

# TODO: torch beta warping (with tensor linspace + approx beta cdf using trapz)
# TODO: Mixup with roll to be more efficient (remove sampling of index)
# TODO: MIT and Rank Mixup


def beta_warping(x, alpha_cdf=1.0, eps=1e-12):
    return scipy.stats.beta.cdf(x, a=alpha_cdf + eps, b=alpha_cdf + eps)


def sim_gauss_kernel(dist, tau_max=1.0, tau_std=0.5):
    dist_rate = tau_max * np.exp(
        -(dist - 1) / (np.mean(dist) * 2 * tau_std * tau_std)
    )
    return 1 / (dist_rate + 1e-12)


def tensor_linspace(start: Tensor, stop: Tensor, num: int):
    """
    Creates a tensor of shape [num, *start.shape] whose values are evenly
    spaced from start to end, inclusive.
    Replicates but the multi-dimensional bahaviour of numpy.linspace in PyTorch.
    """
    # create a tensor of 'num' steps from 0 to 1
    steps = torch.arange(num, dtype=torch.float32, device=start.device) / (
        num - 1
    )

    # reshape the 'steps' tensor to [-1, *([1]*start.ndim)]
    # to allow for broadcastings
    # using 'steps.reshape([-1, *([1]*start.ndim)])' would be nice here
    # but torchscript
    # "cannot statically infer the expected size of a list in this contex",
    # hence the code below
    for i in range(start.ndim):
        steps = steps.unsqueeze(-1)

    # the output starts at 'start' and increments until 'stop' in each dimension
    out = start[None] + steps * (stop - start)[None]

    return out


def torch_beta_cdf(
    x: Tensor, c1: Tensor | float, c2: Tensor | float, npts=100, eps=1e-12
):
    if isinstance(c1, float):
        if c1 == c2:
            c1 = torch.tensor([c1], device=x.device)
            c2 = c1
        else:
            c1 = torch.tensor([c1], device=x.device)
    if isinstance(c2, float):
        c2 = torch.tensor([c2], device=x.device)
    bt = torch.distributions.Beta(c1, c2)

    if isinstance(x, float):
        x = torch.tensor(x)

    X = tensor_linspace(torch.zeros_like(x) + eps, x, npts)
    return torch.trapezoid(bt.log_prob(X).exp(), X, dim=0)


def torch_beta_warping(
    x: Tensor, alpha_cdf: float | Tensor = 1.0, eps=1e-12, npts=100
):
    return torch_beta_cdf(
        x=x, c1=alpha_cdf + eps, c2=alpha_cdf + eps, npts=npts, eps=eps
    )


def torch_sim_gauss_kernel(dist: Tensor, tau_max=1.0, tau_std=0.5):
    dist_rate = tau_max * torch.exp(
        -(dist - 1) / (torch.mean(dist) * 2 * tau_std * tau_std)
    )

    return 1 / (dist_rate + 1e-12)


class AbstractMixup:
    def __init__(self, alpha=1.0, mode="batch", num_classes=1000) -> None:
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
        lam: torch.Tensor | float,
        inp: torch.Tensor,
        index: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(lam, torch.Tensor):
            lam = lam.view(-1, *[1 for _ in range(inp.ndim - 1)]).float()

        return lam * inp + (1 - lam) * inp[index, :]

    def _mix_target(
        self,
        lam: torch.Tensor | float,
        target: torch.Tensor,
        index: torch.Tensor,
    ) -> torch.Tensor:
        y1 = F.one_hot(target, self.num_classes)
        y2 = F.one_hot(target[index], self.num_classes)
        if isinstance(lam, torch.Tensor):
            lam = lam.view(-1, *[1 for _ in range(y1.ndim - 1)]).float()

        if isinstance(lam, torch.Tensor) and lam.dtype == torch.bool:
            return lam * y1 + (~lam) * y2
        else:
            return lam * y1 + (1 - lam) * y2

    def __call__(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return x, y


class Mixup(AbstractMixup):
    def __call__(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lam, index = self._get_params(x.size()[0], x.device)

        mixed_x = self._linear_mixing(lam, x, index)

        mixed_y = self._mix_target(lam, y, index)

        return mixed_x, mixed_y


class MixupIO(AbstractMixup):
    def __call__(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lam, index = self._get_params(x.size()[0], x.device)

        mixed_x = self._linear_mixing(lam, x, index)

        mixed_y = self._mix_target((lam > 0.5), y, index)

        return mixed_x, mixed_y


class RegMixup(AbstractMixup):
    def __call__(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lam, index = self._get_params(x.size()[0], x.device)

        part_x = self._linear_mixing(lam, x, index)

        part_y = self._mix_target(lam, y, index)

        mixed_x = torch.cat([x, part_x], dim=0)
        mixed_y = torch.cat([F.one_hot(y, self.num_classes), part_y], dim=0)

        return mixed_x, mixed_y


class WarpingMixup(AbstractMixup):
    def __init__(
        self,
        alpha=1.0,
        mode="batch",
        num_classes=1000,
        apply_kernel=True,
        tau_max=1.0,
        tau_std=0.5,
    ) -> None:
        super().__init__(alpha, mode, num_classes)
        self.apply_kernel = apply_kernel
        self.tau_max = tau_max
        self.tau_std = tau_std

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
        warp_param=1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lam, index = self._get_params(x.size()[0], x.device)

        if self.apply_kernel:
            l2_dist = (
                (feats - feats[index])
                .pow(2)
                .sum([i for i in range(len(feats.size())) if i > 0])
                .cpu()
                .numpy()
            )
            warp_param = sim_gauss_kernel(l2_dist, self.tau_max, self.tau_std)

        k_lam = torch.tensor(beta_warping(lam, warp_param), device=x.device)

        mixed_x = self._linear_mixing(k_lam, x, index)

        mixed_y = self._mix_target(k_lam, y, index)

        return mixed_x, mixed_y