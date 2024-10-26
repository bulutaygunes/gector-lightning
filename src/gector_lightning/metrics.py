from typing import Tuple

import torch
import torchmetrics
from torchmetrics.classification import BinaryFBetaScore, MulticlassFBetaScore

# noinspection PyProtectedMember
from torchmetrics.functional.classification.f_beta import _safe_divide

# noinspection PyProtectedMember
from torchmetrics.utilities.checks import _input_format_classification


def mask_invalid(
    preds: torch.Tensor, target: torch.Tensor, invalid_label_index: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    mask = target != invalid_label_index
    return preds[mask], target[mask]


# TODO: move two classes to a common mixin class
class MaskedBinaryFBetaScore(BinaryFBetaScore):
    def __init__(self, invalid_label_index: int = -100, **kwargs):
        self.invalid_label_index = invalid_label_index
        super().__init__(**kwargs)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = preds.squeeze(dim=-1)
        preds, target = mask_invalid(preds, target, self.invalid_label_index)
        super().update(preds, target)


class MaskedMulticlassFBetaScore(MulticlassFBetaScore):
    def __init__(self, invalid_label_index: int = -100, **kwargs):
        self.invalid_label_index = invalid_label_index
        super().__init__(**kwargs)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # preds = preds.permute(0, 2, 1)
        preds, target = mask_invalid(preds, target, self.invalid_label_index)
        super().update(preds, target)


class GectorCorrectionFBeta(torchmetrics.Metric):
    def __init__(
        self,
        ignore_index: int,
        beta: float = 0.5,
        invalid_label_index: int = -100,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.ignore_index = ignore_index
        self.beta = beta
        self.invalid_label_index = invalid_label_index

        self.add_state("tp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds, target = mask_invalid(preds, target, self.invalid_label_index)
        preds, target, _ = _input_format_classification(
            preds,
            target,
            threshold=0.5,
            top_k=None,
            num_classes=None,
            multiclass=None,
        )

        # Remove $KEEP token
        preds = torch.cat(
            [preds[:, : self.ignore_index], preds[:, self.ignore_index + 1 :]], dim=1
        )
        target = torch.cat(
            [target[:, : self.ignore_index], target[:, self.ignore_index + 1 :]], dim=1
        )

        true_pred = target == preds
        false_pred = target != preds
        pos_pred = preds == 1
        pos_target = target == 1

        # True positives
        tp = (true_pred * pos_pred).sum()
        self.tp += tp

        # False positives
        fp = (false_pred * pos_pred).sum()
        self.fp += fp

        # False negatives
        fn = (false_pred * pos_target).sum()
        self.fn += fn

    def compute(self):
        precision = _safe_divide(self.tp.float(), self.tp + self.fp)
        recall = _safe_divide(self.tp.float(), self.tp + self.fn)
        num = (1 + self.beta**2) * precision * recall
        denom = self.beta**2 * precision + recall
        return _safe_divide(num, denom)
