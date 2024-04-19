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
    return scipy.stats.beta.cdf(x, a=alpha + eps, b=alpha + eps)


def sim_gauss_kernel(
    dist, tau_max: float = 1.0, tau_std: float = 0.5, inverse_cdf=False
) -> float:
    # dist = dist / np.mean(dist)
    dist_rate = tau_max * np.exp(-(dist - 1) / (2 * tau_std * tau_std))
    if inverse_cdf:
        return dist_rate
    return 1 / (dist_rate + 1e-12)


# ruff: noqa: ERA001
# def tensor_linspace(start: Tensor, stop: Tensor, num: int):
#     """
#     Creates a tensor of shape [num, *start.shape] whose values are evenly
#     spaced from start to end, inclusive.
#     Replicates but the multi-dimensional behaviour of numpy.linspace in PyTorch.
#     """
#     # create a tensor of 'num' steps from 0 to 1
#     steps = torch.arange(num, dtype=torch.float32, device=start.device) / (
#         num - 1
#     )

#     # reshape the 'steps' tensor to [-1, *([1]*start.ndim)]
#     # to allow for broadcastings
#     # using 'steps.reshape([-1, *([1]*start.ndim)])' would be nice here
#     # but torchscript
#     # "cannot statically infer the expected size of a list in this context",
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
            lam = torch.as_tensor(
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

        return lam * y1 + (1 - lam) * y2

    def __call__(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        raise NotImplementedError


class Mixup(AbstractMixup):
    """Original Mixup method from Zhang et al.

    Reference:
        "mixup: Beyond Empirical Risk Minimization" (ICLR 2021)
        http://arxiv.org/abs/1710.09412.
    """

    def __call__(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        lam, index = self._get_params(x.size()[0], x.device)
        mixed_x = self._linear_mixing(lam, x, index)
        mixed_y = self._mix_target(lam, y, index)
        return mixed_x, mixed_y


class MixupIO(AbstractMixup):
    """Mixup on inputs only with targets unchanged, from Wang et al.

    Reference:
        "On the Pitfall of Mixup for Uncertainty Calibration" (CVPR 2023)
        https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_On_the_Pitfall_of_Mixup_for_Uncertainty_Calibration_CVPR_2023_paper.pdf.
    """

    def __call__(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        lam, index = self._get_params(x.size()[0], x.device)

        mixed_x = self._linear_mixing(lam, x, index)

        if self.mode == "batch":
            mixed_y = self._mix_target(float(lam > 0.5), y, index)
        else:
            mixed_y = self._mix_target((lam > 0.5).float(), y, index)

        return mixed_x, mixed_y


class MixupTO(AbstractMixup):
    def __call__(
        self, x: Tensor, y: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, float | Tensor]:
        lam, index = self._get_params(x.size()[0], x.device)

        mixed_y = self._mix_target(lam, y, index)

        return x, x[index], mixed_y, lam


class RegMixup(AbstractMixup):
    """RegMixup method from Pinto et al.

    Reference:
        'RegMixup: Mixup as a Regularizer Can Surprisingly Improve Accuracy and Out Distribution Robustness' (NeurIPS 2022)
        https://arxiv.org/abs/2206.14502.
    """

    def __call__(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
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
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, float | Tensor, float | Tensor]:
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

            filtering = torch.logical_or(
                self.comp_func(pdist_mat, quant),
                torch.eye(pdist_mat.size(0), device=pdist_mat.device),
            )

            new_index = torch.cat(
                [
                    filtering[i]
                    .nonzero()
                    .squeeze(-1)[
                        torch.randint(0, len(filtering[i].nonzero()), (1,))
                    ]
                    for i in range(len(filtering))
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
        lookup: bool = False,
        lookup_size: int = 4096,
    ) -> None:
        """Kernel Warping Mixup method from Bouniot et al.

        Reference:
            "Tailoring Mixup to Data using Kernel Warping functions" (2023)
            https://arxiv.org/abs/2311.01434.
        """
        super().__init__(alpha, mode, num_classes)
        self.apply_kernel = apply_kernel
        self.tau_max = tau_max
        self.tau_std = tau_std
        self.manifold = manifold
        self.regularization = regularization
        self.warping = warping
        self.lookup_size = lookup_size
        if self.warping == "lookup":
            self.rng_gen = None

    def _init_lookup_table(self, lookup_size, device):
        self.rng_gen = torch.distributions.Beta(
            torch.tensor(1.0, device=device), torch.tensor(1.0, device=device)
        )
        eps = 1e-12
        self.y_max = torch.tensor(50, device=device)
        self.nb_betas = self.y_max * 10
        xs = torch.linspace(0, 1, lookup_size)
        ys = torch.linspace(eps, self.y_max, self.nb_betas)

        lookup_table = []

        for x in xs:
            row = scipy.stats.beta.ppf(x, a=ys, b=ys)
            lookup_table.append(row)

        self.lookup_table = torch.tensor(np.array(lookup_table), device=device)

    def _get_params(self, batch_size: int, device: torch.device):
        if self.warping == "lookup":
            if self.rng_gen is None:
                self._init_lookup_table(self.lookup_size, device)
            lam = self.rng_gen.sample_n(batch_size)
        elif self.warping == "no_warp":
            lam = None
        else:
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
        warp_param: float = 1.0,
    ) -> tuple[Tensor, Tensor]:
        lam, index = self._get_params(x.size()[0], x.device)

        if self.apply_kernel:
            if self.warping == "lookup":
                l2_dist = (
                    (feats - feats[index])
                    .pow(2)
                    .sum([i for i in range(len(feats.size())) if i > 0])
                )
                l2_dist = l2_dist / torch.mean(l2_dist)
            else:
                l2_dist = (
                    (feats - feats[index])
                    .pow(2)
                    .sum([i for i in range(len(feats.size())) if i > 0])
                    .cpu()
                    .numpy()
                )
                l2_dist = l2_dist / l2_dist.mean()
                if (
                    self.warping == "inverse_beta_cdf"
                    or self.warping == "no_warp"
                ):
                    warp_param = sim_gauss_kernel(
                        l2_dist, self.tau_max, self.tau_std, inverse_cdf=True
                    )
                elif self.warping == "beta_cdf":
                    warp_param = sim_gauss_kernel(
                        l2_dist, self.tau_max, self.tau_std
                    )
                else:
                    raise NotImplementedError

        if self.warping == "inverse_beta_cdf":
            k_lam = torch.tensor(
                beta_warping(lam, warp_param, inverse_cdf=True), device=x.device
            )
        elif self.warping == "beta_cdf":
            k_lam = torch.tensor(
                beta_warping(lam, warp_param, inverse_cdf=False),
                device=x.device,
            )
        elif self.warping == "no_warp":
            k_lam = torch.tensor(
                np.random.beta(warp_param, warp_param, x.size()[0]),
                device=x.device,
            )
        elif self.warping == "lookup":
            lookup_y = torch.minimum(
                l2_dist // (self.y_max / self.nb_betas), self.nb_betas - 1
            ).int()
            lookup_x = torch.maximum(
                lam // (1 / self.lookup_size), torch.ones_like(self.y_max)
            ).int()
            k_lam = self.lookup_table[lookup_x, lookup_y]
        else:
            raise NotImplementedError

        if self.manifold:
            mixed_y = self._mix_target(k_lam, y, index)
            return x, x[index], mixed_y, k_lam
        if self.regularization:
            part_x = self._linear_mixing(k_lam, x, index)
            part_y = self._mix_target(k_lam, y, index)
            mixed_x = torch.cat([x, part_x], dim=0)
            mixed_y = torch.cat([F.one_hot(y, self.num_classes), part_y], dim=0)
            return mixed_x, mixed_y
        mixed_x = self._linear_mixing(k_lam, x, index)
        mixed_y = self._mix_target(k_lam, y, index)
        return mixed_x, mixed_y


class RankMixupMNDCG(AbstractMixup):
    def __init__(
        self,
        alpha: float = 1.0,
        mode: str = "batch",
        num_classes: int = 1000,
        num_mixup: int = 3,
    ) -> None:
        super().__init__(alpha, mode, num_classes)
        self.num_mixup = num_mixup

    def get_indcg(self, inputs, mixup, lam, targets):
        mixup = mixup.reshape(
            len(lam), -1, self.num_classes
        )  # mixup num x batch x num class
        targets = targets.reshape(
            len(lam), -1, self.num_classes
        )  # mixup num x batch x num class

        mixup = F.softmax(mixup, dim=2)
        inputs = F.softmax(inputs, dim=1)

        inputs_lam = torch.ones(inputs.size(0), 1, device=inputs.device)
        max_values = inputs.max(dim=1, keepdim=True)[0]
        max_mixup = mixup.max(dim=2)[0].t()  #  batch  x mixup num
        max_lam = targets.max(dim=2)[0].t()  #  batch  x mixup num
        # compute dcg
        sort_index = torch.argsort(max_lam, descending=True)
        max_mixup_sorted = torch.gather(max_mixup, 1, sort_index)
        order = torch.arange(1, 2 + len(lam), device=max_mixup.device)
        dcg_order = torch.log2(order + 1)
        max_mixup_sorted = torch.cat((max_values, max_mixup_sorted), dim=1)
        dcg = (max_mixup_sorted / dcg_order).sum(dim=1)

        max_lam_sorted = torch.gather(max_lam, 1, sort_index)
        max_lam_sorted = torch.cat((inputs_lam, max_lam_sorted), dim=1)
        idcg = (max_lam_sorted / dcg_order).sum(dim=1)

        # compute ndcg
        ndcg = dcg / idcg
        inv_ndcg = idcg / dcg
        ndcg_mask = idcg > dcg
        return ndcg_mask * ndcg + (~ndcg_mask) * inv_ndcg

    def __call__(
        self, x: Tensor, y: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        mixup_data = []
        lams = []
        mixup_y = []

        for _ in range(self.num_mixup):
            lam, index = self._get_params(x.size()[0], x.device)
            part_x = self._linear_mixing(lam, x, index)
            part_y = self._mix_target(lam, y, index)
            mixup_data.append(part_x)
            mixup_y.append(part_y)
            lams.append(lam)

        return (
            x,
            torch.cat(mixup_data, dim=0),
            y,
            torch.cat(mixup_y, dim=0),
            torch.tensor(lams, device=x.device),
        )
