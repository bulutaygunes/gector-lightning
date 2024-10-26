import time

import torch
from lightning.pytorch.callbacks import BaseFinetuning, Callback
from lightning.pytorch.utilities import rank_zero_only

from gector_lightning.models import GectorModel


# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
class LogEpochTimeCallback(Callback):
    """Simple callback that logs how long each epoch takes, in seconds, to a pytorch
    lightning log.

    Taken from NVIDIA NeMo
    (https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/common/callbacks/callbacks.py).
    """

    @rank_zero_only
    def on_train_epoch_start(self, trainer, pl_module):
        # noinspection PyAttributeOutsideInit
        self.epoch_start = time.time()

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        curr_time = time.time()
        duration = curr_time - self.epoch_start
        trainer.logger.log_metrics({"epoch_time": duration}, step=trainer.global_step)


class GectorEncoderFreezeCallback(BaseFinetuning):
    def __init__(self, epochs: int = 0, lr: float | None = None) -> None:
        super().__init__()
        self._epochs = epochs
        self._lr = lr

    def freeze_before_training(self, pl_module: GectorModel) -> None:
        self.freeze(pl_module.encoder, train_bn=False)

        # LR for the steps where encoder is frozen
        init_lr = self._lr or pl_module.optim_cfg.lr

        # LR after the encoder is unfrozen
        self._lr = pl_module.optim_cfg.lr

        pl_module.optim_cfg.lr = init_lr

    def finetune_function(
        self,
        pl_module: GectorModel,
        current_epoch: int,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        if current_epoch == self._epochs:
            self.unfreeze_and_add_param_group(
                modules=pl_module.encoder,
                optimizer=optimizer,
                train_bn=True,
            )

            # Merge param groups
            params = []
            for param_group in optimizer.param_groups:
                params.extend(param_group["params"])
            optimizer.param_groups = [optimizer.param_groups[0]]
            optimizer.param_groups[0]["params"] = params
            optimizer.param_groups[0]["lr"] = self._lr
