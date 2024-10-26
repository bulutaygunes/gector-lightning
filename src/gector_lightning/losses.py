from typing import List, Optional

import torch
import torch.nn as nn

# noinspection PyPep8Naming
import torch.nn.functional as F


# noinspection PyShadowingBuiltins
class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self, binary: bool = False, ignore_index: int = -100) -> None:
        super().__init__()
        self.binary = binary
        self.ignore_index = ignore_index

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.binary:
            input = input.squeeze(dim=-1)
            target = target.type_as(input)
            loss = F.binary_cross_entropy_with_logits(input, target, reduction="none")
        else:
            input = input.permute(0, 2, 1)
            loss = F.cross_entropy(
                input, target, ignore_index=self.ignore_index, reduction="none"
            )
        return loss[target != self.ignore_index].mean()


# noinspection PyShadowingBuiltins
class MaskedFocalLoss(nn.Module):
    def __init__(
        self,
        binary: bool = False,
        alpha: Optional[List[float]] = None,
        gamma: float = 0.0,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        self.register_buffer("alpha", torch.Tensor(alpha) if alpha else None)
        self.gamma = gamma
        self.ignore_index = ignore_index

        if binary:
            self.focal_loss = self._binary_focal_loss
        else:
            self.focal_loss = torch.hub.load(
                "adeelh/pytorch-multi-class-focal-loss",
                model="FocalLoss",
                alpha=self.alpha,
                gamma=gamma,
                reduction="mean",
                ignore_index=ignore_index,
                force_reload=False,
            )

    def _binary_focal_loss(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        valid_mask = target != self.ignore_index
        target = target[valid_mask].unsqueeze(-1)  # N x 1
        input = input.squeeze(1)[valid_mask]

        logp_pos = F.logsigmoid(input)
        logp_neg = logp_pos - input
        logp = torch.stack([logp_neg, logp_pos], dim=1)  # N X 2

        alpha = self.alpha[target].squeeze(-1) if self.alpha is not None else 1
        logpt = logp.gather(1, target).squeeze(-1)
        pt = logpt.exp()

        loss = -alpha * (1 - pt) ** self.gamma * logpt
        return loss.mean()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input = input.permute(0, 2, 1)
        return self.focal_loss(input, target)
