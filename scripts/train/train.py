import shutil
import time
from pathlib import Path
from typing import Annotated

import typer
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig, OmegaConf

from gector_lightning.callbacks import GectorEncoderFreezeCallback, LogEpochTimeCallback
from gector_lightning.data import GectorDataModule
from gector_lightning.models import GectorModel


def main(
    config_path: Annotated[
        Path,
        typer.Option(
            help="Path to the config file. E.g. conf/train/state1.yaml",
            exists=True,
            file_okay=True,
            dir_okay=False,
        ),
    ],
    experiment_dir: Annotated[
        Path,
        typer.Option(
            help="Path to the experiment directory to save the logs and checkpoints.",
            exists=False,
            file_okay=False,
            dir_okay=True,
        ),
    ],
):
    cfg = OmegaConf.load(config_path)

    if "seed" in cfg:
        seed_everything(cfg["seed"])

    if cfg.get("init_from_pretrained_model"):
        model_path = cfg["init_from_pretrained_model"]
        model = GectorModel.load_from_checkpoint(model_path, **cfg.model)
    else:
        model = GectorModel(**cfg.model)

    datamodule = GectorDataModule(
        cfg=cfg.data_module, output_vocab=model.output_vocab, tokenizer=model.tokenizer
    )

    version = time.strftime("%Y-%m-%d_%H-%M-%S")
    logger = TensorBoardLogger(
        experiment_dir, name="", version=version, default_hp_metric=False
    )

    # Copy config to experiment directory
    experiment_dir_with_version = experiment_dir / str(version)
    experiment_dir_with_version.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, experiment_dir_with_version)

    trainer = Trainer(
        **{
            key: dict(val) if isinstance(val, DictConfig) else val
            for key, val in cfg.trainer.items()
        },
        logger=logger,
        callbacks=[
            LearningRateMonitor(),
            LogEpochTimeCallback(),
            ModelCheckpoint(
                dirpath=experiment_dir / str(version) / "checkpoints",
                filename="{epoch}-{step}-{v_detection_loss:.3f}-{"
                "v_correction_loss:.3f}-{v_detection_F0_5:.3f}-{v_correction_F0_5:.3f}",
                **cfg.model_checkpoint,
            ),
            GectorEncoderFreezeCallback(**cfg.encoder_freeze_callback),
        ],
    )
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    typer.run(main)
