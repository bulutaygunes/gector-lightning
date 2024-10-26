from pathlib import Path
from typing import Type

# noinspection PyPep8Naming
import lightning as L
import torch
from lightning import LightningModule
from omegaconf import DictConfig, OmegaConf
from torch import nn as nn
from transformers import AutoModel, AutoTokenizer

from gector_lightning.data import GectorOutputVocab
from gector_lightning.losses import MaskedCrossEntropyLoss, MaskedFocalLoss
from gector_lightning.metrics import (
    GectorCorrectionFBeta,
    MaskedBinaryFBetaScore,
    MaskedMulticlassFBetaScore,
)
from gector_lightning.utils.helpers import get_target_sent_by_edits


def load_model(
    model_cls: Type[LightningModule], config_path: Path | str, device: str = "cpu"
) -> LightningModule:
    cfg = OmegaConf.load(config_path)
    model_path = cfg["init_from_pretrained_model"]
    assert model_path is not None
    model = model_cls.load_from_checkpoint(model_path, **cfg.model)
    model.to(torch.device(device))
    return model.eval()


def tensor_to_list(
    tensors: torch.Tensor | list[torch.Tensor], ignore: int | float = -1
) -> list[list[int | float]]:
    """
    Parameters
    ----------
    tensors: list of 2-D tensors
    ignore
        The elements after the first appearance of `ignore` removed from each row.

    Returns
    -------
    list[list[int | float]]
        Listified version of `tensors`.

    """
    out = []
    for tensor in tensors if isinstance(tensors, list) else [tensors]:
        tensor_list = tensor.tolist()
        for i, (row_list, row_tensor) in enumerate(zip(tensor_list, tensor)):
            nonzero_idx = (row_tensor == ignore).nonzero()
            if len(nonzero_idx) == 0:
                cur_len = len(row_tensor)
            else:
                cur_len = nonzero_idx[0, 0]
            row_list = row_list[:cur_len]
            tensor_list[i] = row_list
        out.append(tensor_list)

    return out if isinstance(tensors, list) else out[0]


class GectorModel(L.LightningModule):
    def __init__(
        self,
        *,
        encoder_name: str,
        output_vocab: dict | DictConfig,
        optimizer: dict | DictConfig,
        loss: dict | DictConfig,
        scheduler: dict | DictConfig | None = None,
        dropout_correction: float = 0.0,
        dropout_detection: float = 0.0,
        num_classes_detection: int = 1,
        detection_loss_weight: float = 1.0,
        min_len: int = 4,
        max_correction_iterations: int = 5,
        min_correction_confidence: float = 0.0,
        min_error_probability: float = 0.0,
        additional_keep_confidence: float = 0.0,
        detection_pool_mode: str = "max",
        correction_pool_mode: str = "most_confident",
    ) -> None:
        super().__init__()
        assert num_classes_detection in {
            1,
            2,
        }, "Number of detection classes should be either 1 or 2."
        assert detection_pool_mode in {"min", "mean", "max"}
        assert correction_pool_mode in {"first", "most_confident"}

        self.num_classes_detection = num_classes_detection
        self.detection_loss_weight = detection_loss_weight
        self.min_len = min_len
        self.max_correction_iterations = max_correction_iterations
        self.min_correction_confidence = min_correction_confidence
        self.min_error_probability = min_error_probability
        self.additional_keep_confidence = additional_keep_confidence
        self.detection_pool_mode = detection_pool_mode
        self.correction_pool_mode = correction_pool_mode

        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.output_vocab = GectorOutputVocab(**output_vocab)

        # Error detection & correction layers
        num_hidden = self.encoder.encoder.layer[-1].output.dense.out_features
        self.correction_layer = nn.Linear(num_hidden, len(self.output_vocab), bias=True)
        self.detection_layer = nn.Linear(num_hidden, num_classes_detection, bias=True)
        self.dropout_correction = nn.Dropout(p=dropout_correction)
        self.dropout_detection = nn.Dropout(p=dropout_detection)

        # Losses
        self.correction_loss = self._prepare_loss(loss.correction)
        self.detection_loss = self._prepare_loss(
            loss.detection, binary=self.num_classes_detection == 1
        )

        # Metrics
        if self.num_classes_detection == 1:
            self.detection_f_0_5_train = MaskedBinaryFBetaScore(
                beta=0.5, invalid_label_index=-100
            )
            self.detection_f_0_5_val = MaskedBinaryFBetaScore(
                beta=0.5, invalid_label_index=-100
            )
        else:
            self.detection_f_0_5_train = MaskedMulticlassFBetaScore(
                num_classes=2, beta=0.5, ignore_index=0, invalid_label_index=-100
            )
            self.detection_f_0_5_val = MaskedMulticlassFBetaScore(
                num_classes=2, beta=0.5, ignore_index=0, invalid_label_index=-100
            )

        self.correction_f_0_5_train = GectorCorrectionFBeta(
            beta=0.5, ignore_index=0, invalid_label_index=-100
        )
        self.correction_f_0_5_val = GectorCorrectionFBeta(
            beta=0.5, ignore_index=0, invalid_label_index=-100
        )

        # Optimizer configuration
        self.optim_cfg = optimizer
        self.scheduler_cfg = scheduler

    @staticmethod
    def _prepare_loss(
        loss_cfg: dict | DictConfig, binary: bool = False
    ) -> MaskedFocalLoss | MaskedCrossEntropyLoss:
        loss_kwargs = loss_cfg.copy()
        loss_name = loss_kwargs.pop("name")
        if loss_name == "CrossEntropy":
            loss_cls = MaskedCrossEntropyLoss
        elif loss_name == "Focal":
            loss_cls = MaskedFocalLoss
        else:
            raise Exception(f"Loss {loss_name} not supported.")
        return loss_cls(binary=binary, **loss_kwargs)

    def forward(
        self, tokenized_inputs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            tokenized_inputs = tokenized_inputs.to(self.device)
            encoded = self.encoder(**tokenized_inputs)
            correction_logits = self.correction_layer(encoded.last_hidden_state)
            detection_logits = self.detection_layer(encoded.last_hidden_state)

            correction_probs = correction_logits.softmax(dim=-1)
            correction_probs[
                ..., self.output_vocab.keep_idx
            ] += self.additional_keep_confidence
            correction_probs, correction_ids = correction_probs.max(dim=-1)
            correction_probs.clamp_(max=1.0)

            if self.num_classes_detection == 1:
                error_probs = detection_logits.sigmoid().squeeze(dim=-1)
            else:
                error_probs = detection_logits.softmax(dim=-1)[..., 1]

            # Modify mask to remove end of sentence tokens
            eos_mask = tokenized_inputs.input_ids == self.tokenizer.sep_token_id
            mask = tokenized_inputs.attention_mask
            mask[eos_mask] = 0
            mask = mask.bool()

            # Mask out irrelevant parts
            correction_probs[~mask] = -1
            correction_ids[~mask] = -1
            error_probs[~mask] = -1

            return correction_probs, correction_ids, error_probs

    def training_step(self, batch, batch_idx):
        correction_labels = batch.correction_labels
        batch.pop("correction_labels")
        detection_labels = batch.detection_labels
        batch.pop("detection_labels")

        encoded = self.encoder(**batch)
        correction_logits = self.correction_layer(
            self.dropout_correction(encoded.last_hidden_state)
        )
        detection_logits = self.detection_layer(
            self.dropout_detection(encoded.last_hidden_state)
        )

        # Calculate losses
        correction_loss = self.correction_loss(correction_logits, correction_labels)
        detection_loss = self.detection_loss(detection_logits, detection_labels)
        loss = correction_loss + self.detection_loss_weight * detection_loss
        self.log("t_detection_loss", detection_loss)
        self.log("t_correction_loss", correction_loss)

        # Calculate metrics
        self.detection_f_0_5_train(detection_logits, detection_labels)
        self.correction_f_0_5_train(correction_logits, correction_labels)
        self.log(
            "t_detection_F0_5", self.detection_f_0_5_train, on_step=True, on_epoch=True
        )
        self.log(
            "t_correction_F0_5",
            self.correction_f_0_5_train,
            on_step=True,
            on_epoch=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        correction_labels = batch.correction_labels
        batch.pop("correction_labels")
        detection_labels = batch.detection_labels
        batch.pop("detection_labels")

        encoded = self.encoder(**batch)
        correction_logits = self.correction_layer(encoded.last_hidden_state)
        detection_logits = self.detection_layer(encoded.last_hidden_state)

        # Calculate losses
        correction_loss = self.correction_loss(correction_logits, correction_labels)
        detection_loss = self.detection_loss(detection_logits, detection_labels)
        loss = correction_loss + detection_loss
        self.log("v_detection_loss", detection_loss)
        self.log("v_correction_loss", correction_loss)
        self.log("v_loss", loss)

        # Calculate metrics
        self.detection_f_0_5_val(detection_logits, detection_labels)
        self.correction_f_0_5_val(correction_logits, correction_labels)
        self.log("v_detection_F0_5", self.detection_f_0_5_val)
        self.log("v_correction_F0_5", self.correction_f_0_5_val)

    def configure_optimizers(self):
        optim_kwargs = self.optim_cfg.copy()
        optim_name = optim_kwargs.pop("name")
        params = filter(lambda p: p.requires_grad, self.parameters())

        if optim_name == "Adam":
            optimizer = torch.optim.Adam(params, **optim_kwargs)
        else:
            raise Exception(f"Optimizer {optim_name} not supported.")

        if self.scheduler_cfg:
            scheduler_kwargs = self.scheduler_cfg.copy()
            scheduler_name = scheduler_kwargs.pop("name")
            metric_to_monitor = scheduler_kwargs.pop("monitor")

            if scheduler_name == "ReduceLROnPlateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, **scheduler_kwargs
                )
            else:
                raise Exception(f"Scheduler {scheduler_name} not supported.")

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": metric_to_monitor,
                },
            }
        else:
            return optimizer

    def get_token_action(
        self, index: int, confidence: float, sugg_token: str
    ) -> tuple[int, int, str] | None:
        # Cases when we don't need to do anything
        if confidence < self.min_correction_confidence or sugg_token == "$KEEP":
            return None

        if (
            sugg_token.startswith("$REPLACE_")
            or sugg_token.startswith("$TRANSFORM_")
            or sugg_token == "$DELETE"
        ):
            start_pos = index
            end_pos = index + 1
        elif sugg_token.startswith("$APPEND_") or sugg_token.startswith("$MERGE_"):
            start_pos = index + 1
            end_pos = index + 1
        else:
            raise Exception("Invalid token.")

        if sugg_token == "$DELETE":
            sugg_token_clear = ""
        elif sugg_token.startswith("$TRANSFORM_") or sugg_token.startswith("$MERGE_"):
            sugg_token_clear = sugg_token[:]
        else:
            sugg_token_clear = sugg_token[sugg_token.index("_") + 1 :]

        return start_pos, end_pos, sugg_token_clear

    @staticmethod
    def _update_final_batch(final_batch, pred_ids, pred_batch, prev_preds_dict):
        new_pred_ids = []
        for i, orig_id in enumerate(pred_ids):
            orig = final_batch[orig_id]
            pred = pred_batch[i]
            prev_preds = prev_preds_dict[orig_id]
            if orig != pred and pred not in prev_preds:
                final_batch[orig_id] = pred
                new_pred_ids.append(orig_id)
                prev_preds_dict[orig_id].append(pred)
            elif orig != pred and pred in prev_preds:
                # update final batch, but stop iterations
                final_batch[orig_id] = pred
            else:
                continue
        return final_batch, new_pred_ids

    def _postprocess_batch(
        self, all_tokens, all_probabilities, all_idxs, all_error_probs
    ):
        all_results = []
        noop_index = self.output_vocab.keep_idx

        for tokens, probabilities, idxs, error_probs in zip(
            all_tokens, all_probabilities, all_idxs, all_error_probs
        ):
            edits = []

            # Skip the whole sentence if there are no errors or probability of
            # correctness is high
            if max(idxs) == noop_index or max(error_probs) < self.min_error_probability:
                results = tokens
            else:
                for i, (idx, suggestion_confidence, error_prob) in enumerate(
                    zip(idxs, probabilities, error_probs), start=-1
                ):
                    # Skip if there is no error
                    if idx == noop_index:
                        continue

                    # TODO: experimental!!!
                    if error_prob < self.min_error_probability:
                        continue

                    suggestion = self.output_vocab.convert_ids_to_tokens(idx)
                    action = self.get_token_action(i, suggestion_confidence, suggestion)

                    if not action:
                        continue

                    edits.append(list(action) + [suggestion_confidence])
                results = get_target_sent_by_edits(tokens, edits)
            all_results.append(results)
        return all_results

    def _pool_error_probs(self, error_probs: list[float]) -> float:
        if self.detection_pool_mode == "max":
            return max(error_probs)
        elif self.detection_pool_mode == "mean":
            return sum(error_probs) / len(error_probs)
        else:
            return min(error_probs)

    def _pool_correction_probs(
        self, correction_probs: list[float], correction_ids: list[int]
    ) -> tuple[float, int]:
        if self.correction_pool_mode == "first":
            return correction_probs[0], correction_ids[0]
        else:
            max_prob = max(correction_probs)
            max_idx = correction_probs.index(max_prob)
            return max_prob, correction_ids[max_idx]

    def _pool_subword_results(
        self, tokenized_inputs, correction_probs, correction_ids, error_probs
    ):
        correction_probs_processed, correction_ids_processed, error_probs_processed = (
            [],
            [],
            [],
        )
        for batch_index, (
            cur_correction_probs,
            cur_correction_ids,
            cur_error_probs,
        ) in enumerate(zip(correction_probs, correction_ids, error_probs)):
            word_ids = tokenized_inputs.word_ids(batch_index=batch_index)
            previous_word_idx = None
            (
                cur_correction_probs_processed,
                cur_correction_ids_processed,
                cur_error_probs_processed,
            ) = ([], [], [])
            correction_probs_word, correction_ids_word, error_probs_word = [], [], []

            for i, word_idx in enumerate(word_ids):
                if i >= len(cur_correction_probs):
                    break

                if i == 0:  # Start token
                    cur_correction_probs_processed.append(cur_correction_probs[0])
                    cur_correction_ids_processed.append(cur_correction_ids[0])
                    cur_error_probs_processed.append(cur_error_probs[0])

                # Same word as the previous token
                elif word_idx == previous_word_idx:
                    correction_probs_word.append(cur_correction_probs[i])
                    correction_ids_word.append(cur_correction_ids[i])
                    error_probs_word.append(cur_error_probs[i])

                # New word
                else:
                    # Add results of the previous word
                    if correction_probs_word:
                        (
                            correction_prob_pooled,
                            correction_id_pooled,
                        ) = self._pool_correction_probs(
                            correction_probs_word, correction_ids_word
                        )
                        error_prob_pooled = self._pool_error_probs(error_probs_word)

                        cur_correction_probs_processed.append(correction_prob_pooled)
                        cur_correction_ids_processed.append(correction_id_pooled)
                        cur_error_probs_processed.append(error_prob_pooled)

                    # Current word
                    correction_probs_word = [cur_correction_probs[i]]
                    correction_ids_word = [cur_correction_ids[i]]
                    error_probs_word = [cur_error_probs[i]]

                previous_word_idx = word_idx

            # If some results left in the buffer
            if correction_probs_word:
                (
                    correction_prob_pooled,
                    correction_id_pooled,
                ) = self._pool_correction_probs(
                    correction_probs_word, correction_ids_word
                )
                error_prob_pooled = self._pool_error_probs(error_probs_word)

                cur_correction_probs_processed.append(correction_prob_pooled)
                cur_correction_ids_processed.append(correction_id_pooled)
                cur_error_probs_processed.append(error_prob_pooled)

            correction_probs_processed.append(cur_correction_probs_processed)
            correction_ids_processed.append(cur_correction_ids_processed)
            error_probs_processed.append(cur_error_probs_processed)

        return (
            correction_probs_processed,
            correction_ids_processed,
            error_probs_processed,
        )

    def handle_batch(
        self, batch: list[list[str]]
    ) -> tuple[list[list[str]], list[list[float]]]:
        batch_size = len(batch)
        final_batch = batch[:]

        prev_preds_dict = {i: [final_batch[i]] for i in range(batch_size)}
        pred_ids = [i for i in range(batch_size) if len(batch[i]) >= self.min_len]
        error_probs_first_iter = [[0] * len(batch[i]) for i in range(batch_size)]

        for n_iter in range(self.max_correction_iterations):
            orig_batch = [final_batch[i] for i in pred_ids]
            if not orig_batch or all(len(entry) == 0 for entry in orig_batch):
                break

            # FIXME: fix truncation
            tokenized_inputs = self.tokenizer(
                orig_batch, padding=True, is_split_into_words=True, return_tensors="pt"
            )
            correction_probs, correction_ids, error_probs = self(tokenized_inputs)
            correction_probs, correction_ids, error_probs = tensor_to_list(
                [correction_probs, correction_ids, error_probs],
                ignore=-1,
            )
            correction_probs, correction_ids, error_probs = self._pool_subword_results(
                tokenized_inputs, correction_probs, correction_ids, error_probs
            )

            pred_batch = self._postprocess_batch(
                orig_batch,
                correction_probs,
                correction_ids,
                error_probs,
            )

            # Return the token error probabilities for the first iteration only
            if n_iter == 0:
                for cur_idx, cur_error_probs in zip(pred_ids, error_probs):
                    error_probs_first_iter[cur_idx] = cur_error_probs

            final_batch, pred_ids = self._update_final_batch(
                final_batch, pred_ids, pred_batch, prev_preds_dict
            )

            # No more iterations needed
            if not pred_ids:
                break

        return final_batch, error_probs_first_iter
