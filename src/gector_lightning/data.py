import logging
import re
from functools import partial, singledispatchmethod
from typing import Iterator

import torch
from lightning.pytorch.core.datamodule import LightningDataModule
from omegaconf import DictConfig, ListConfig
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from gector_lightning.utils.helpers import SEQ_DELIMITERS, START_TOKEN

logger = logging.getLogger(__name__)

LabelsT = list[list[str]]
InstanceT = tuple[list[str], LabelsT]


# Fix broken sentences mostly in Lang8
BROKEN_SENTENCES_REGEXP = re.compile(r"\.[a-zA-RT-Z]")


def _read_line_positions(*, path: str) -> list[tuple[int, int]]:
    positions = []

    with open(path, "r") as data_file:
        logger.info("Reading line positions from lines in file at: %s", path)

        start_pos = 0
        while True:
            data_file.readline()
            end_pos = data_file.tell()

            # No lines left
            if start_pos == end_pos:
                break

            positions.append((start_pos, end_pos - start_pos))
            start_pos = end_pos
    return positions


def _process_line(
    line: str,
    min_len: int = 1,
    max_len: int | None = None,
    skip_complex: int = 0,
    broken_dot_strategy: str = "keep",
) -> InstanceT | None:
    line = line.strip("\n")
    # skip blank and broken lines
    if not line or (
        broken_dot_strategy == "skip"
        and BROKEN_SENTENCES_REGEXP.search(line) is not None
    ):
        return None

    tokens_and_tags = [
        pair.rsplit(SEQ_DELIMITERS["labels"], 1)
        for pair in line.split(SEQ_DELIMITERS["tokens"])
    ]
    try:
        tokens = [token for token, tag in tokens_and_tags]
        tags = [tag for token, tag in tokens_and_tags]
    except ValueError:
        return None

    if tokens and tokens[0] != START_TOKEN:
        tokens = [START_TOKEN] + tokens

    # Minimum length check
    if len(tokens) < min_len + 1:  # +1 for $START
        return None

    if max_len is not None:
        tokens = tokens[:max_len]
        tags = tags[:max_len]
    instance = _text_to_instance(tokens, tags, skip_complex)
    return instance


def _read_data(
    *,
    path: str,
    min_len: int = 1,
    max_len: int | None = None,
    skip_complex: int = 0,
    broken_dot_strategy: str = "keep",
) -> Iterator[InstanceT]:
    with open(path, "r") as data_file:
        logger.info("Reading instances from lines in file at: %s", path)
        for line_idx, line in enumerate(data_file):
            instance = _process_line(
                line,
                min_len=min_len,
                max_len=max_len,
                skip_complex=skip_complex,
                broken_dot_strategy=broken_dot_strategy,
            )

            if instance:
                yield instance


def _text_to_instance(
    tokens: list[str], tags: list[str], skip_complex: int = 0
) -> InstanceT | None:
    op_del = SEQ_DELIMITERS["operations"]
    labels = [x.split(op_del) for x in tags]

    if skip_complex and any(len(x) > skip_complex for x in labels):
        return None

    return tokens, labels


def _process_labels(
    labels: LabelsT, output_vocab: "GectorOutputVocab"
) -> tuple[list[int], list[int]]:
    # TODO: currently "keep_one" only, add other tag strategies here
    correction_labels = [label_list[0] for label_list in labels]
    detection_labels = [0 if label == "$KEEP" else 1 for label in correction_labels]
    correction_labels = output_vocab.convert_tokens_to_ids(correction_labels)
    return correction_labels, detection_labels


class GectorOutputVocab:
    def __init__(self, path: str, num_tokens: int, unknown_token_action: str) -> None:
        assert unknown_token_action in {"keep", "skip"}
        self.unknown_token_action = unknown_token_action

        # Load vocab
        with open(path, "r") as f:
            self._tokens = [line.strip() for line in f]
        assert len(self) == num_tokens, (
            f"Number of tokens in the vocab ({len(self)}) doesn't match "
            f"num_tokens ({num_tokens})."
        )
        self._token_to_id_map = {token: i for i, token in enumerate(self._tokens)}
        self.keep_idx = self._token_to_id_map["$KEEP"]
        self.unknown_token_idx = (
            self.keep_idx if self.unknown_token_action == "keep" else -100
        )

    def __len__(self):
        return len(self._tokens)

    @singledispatchmethod
    def convert_tokens_to_ids(self, tokens: str | list[str]) -> int | list[int]:
        raise NotImplementedError("Unsupported input type")

    @convert_tokens_to_ids.register
    def _(self, tokens: str) -> int:
        return self._token_to_id_map.get(tokens, self.unknown_token_idx)

    @convert_tokens_to_ids.register(list)
    def _(self, tokens: list[str]) -> list[int]:
        return [
            self._token_to_id_map.get(token, self.unknown_token_idx) for token in tokens
        ]

    @singledispatchmethod
    def convert_ids_to_tokens(self, ids: int | list[int]) -> str | list[str]:
        raise NotImplementedError("Unsupported input type")

    @convert_ids_to_tokens.register
    def _(self, ids: int) -> str:
        return self._tokens[ids]

    @convert_ids_to_tokens.register(list)
    def _(self, ids: list[int]) -> list[str]:
        return [self._tokens[id_] for id_ in ids]


class GectorDataset(Dataset):
    def __init__(
        self,
        tokens: list[list[str]],
        correction_labels: list[list[int]],
        detection_labels: list[list[int]],
    ) -> None:
        self.tokens = tokens
        self.correction_labels = correction_labels
        self.detection_labels = detection_labels

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, idx: int) -> tuple[list[str], list[int], list[int]]:
        # TODO: this currently discards the special $START token and uses the original
        #  start token of the tokenizer (e.g. [CLS] in BERT without the pooling layer).
        #  We might experiment with adding the start token to the vocab.
        return (
            self.tokens[idx][1:],
            self.correction_labels[idx],
            self.detection_labels[idx],
        )


class LazyGectorDataset(Dataset):
    def __init__(
        self,
        path: str,
        positions: list[tuple[int, int]],
        output_vocab: GectorOutputVocab,
        min_len: int = 1,
        max_len: int | None = None,
        skip_correct: bool = False,
        skip_complex: int = 0,
        broken_dot_strategy: str = "keep",
    ) -> None:
        self.file_descriptor = open(path, "r")
        self.positions = positions
        self.output_vocab = output_vocab
        self.min_len = min_len
        self.max_len = max_len
        self.skip_correct = skip_correct
        self.skip_complex = skip_complex
        self.broken_dot_strategy = broken_dot_strategy

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, idx: int) -> tuple[list[str], list[int], list[int]]:
        start_pos, size = self.positions[idx]
        self.file_descriptor.seek(start_pos)
        line = self.file_descriptor.read(size)
        instance = _process_line(
            line,
            min_len=self.min_len,
            max_len=self.max_len,
            skip_complex=self.skip_complex,
            broken_dot_strategy=self.broken_dot_strategy,
        )

        if instance:
            tokens, labels = instance
            correction_labels, detection_labels = _process_labels(
                labels, self.output_vocab
            )

            if not self.skip_correct or any(detection_labels):
                return (
                    tokens[1:],
                    correction_labels,
                    detection_labels,
                )  # TODO: see __getitem__ in GectorDataset

        # Empty data for default return condition
        return [], [-100], [-100]


# noinspection PyAttributeOutsideInit
class GectorDataModule(LightningDataModule):
    def __init__(
        self,
        *,
        cfg: DictConfig,
        output_vocab: GectorOutputVocab,
        tokenizer: PreTrainedTokenizerFast,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.output_vocab = output_vocab
        self.tokenizer = tokenizer

    def _create_dataset(self, cfg: DictConfig) -> Dataset:
        lazy = cfg.get("lazy", False)
        min_len = cfg.get("min_len", 1)
        max_len = cfg.get("max_len")
        skip_correct = cfg.get("skip_correct", False)

        def create_single_dataset(**kwargs):
            def create_lazy_dataset(
                *, path: str, skip_complex: int = 0, broken_dot_strategy: str = "keep"
            ):
                positions = _read_line_positions(path=path)
                return LazyGectorDataset(
                    path,
                    positions,
                    self.output_vocab,
                    min_len=min_len,
                    max_len=max_len,
                    skip_correct=skip_correct,
                    skip_complex=skip_complex,
                    broken_dot_strategy=broken_dot_strategy,
                )

            if lazy:
                return create_lazy_dataset(**kwargs)
            else:
                # Read data with labels
                tokens, correction_labels, detection_labels = [], [], []
                for cur_tokens, cur_labels in _read_data(
                    min_len=min_len, max_len=max_len, **kwargs
                ):
                    cur_correction_labels, cur_detection_labels = _process_labels(
                        cur_labels, self.output_vocab
                    )
                    if skip_correct and not any(cur_detection_labels):
                        continue

                    tokens.append(cur_tokens)
                    correction_labels.append(cur_correction_labels)
                    detection_labels.append(cur_detection_labels)
                return GectorDataset(tokens, correction_labels, detection_labels)

        def create_multiple_datasets(*param_list) -> ConcatDataset:
            datasets = [
                create_single_dataset(**cur_param_list) for cur_param_list in param_list
            ]
            return ConcatDataset(datasets)

        if isinstance(cfg.dataset_params, ListConfig):  # Multiple datasets
            return create_multiple_datasets(*cfg.dataset_params)
        else:
            return create_single_dataset(**cfg.dataset_params)

    def _collate_fn_template(self, batch, label_all_tokens: bool):
        tokens_batch = [item[0] for item in batch]
        correction_labels_batch = [item[1] for item in batch]
        detection_labels_batch = [item[2] for item in batch]

        tokenized_inputs = self.tokenizer(
            tokens_batch,
            padding=True,
            truncation=True,
            is_split_into_words=True,
            return_tensors="pt",
        )
        correction_labels_batch_processed, detection_labels_batch_processed = [], []

        for batch_index, (correction_labels, detection_labels) in enumerate(
            zip(correction_labels_batch, detection_labels_batch)
        ):
            word_ids = tokenized_inputs.word_ids(batch_index=batch_index)
            previous_word_idx = None
            correction_labels_processed, detection_labels_processed = [], []

            for i, word_idx in enumerate(word_ids):
                if word_idx is None:
                    if i == 0:  # Start token
                        correction_labels_processed.append(correction_labels[0])
                        detection_labels_processed.append(detection_labels[0])
                    else:  # Other special tokens (-100 to ignore)
                        correction_labels_processed.append(-100)
                        detection_labels_processed.append(-100)

                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    correction_labels_processed.append(correction_labels[word_idx + 1])
                    detection_labels_processed.append(detection_labels[word_idx + 1])

                # For the other tokens in a word, we set the label to either the
                # current label or -100, depending on the label_all_tokens flag
                else:
                    correction_labels_processed.append(
                        correction_labels[word_idx + 1] if label_all_tokens else -100
                    )
                    detection_labels_processed.append(
                        detection_labels[word_idx + 1] if label_all_tokens else -100
                    )
                previous_word_idx = word_idx

            correction_labels_batch_processed.append(
                torch.LongTensor(correction_labels_processed)
            )
            detection_labels_batch_processed.append(
                torch.LongTensor(detection_labels_processed)
            )

        tokenized_inputs["correction_labels"] = pad_sequence(
            correction_labels_batch_processed, batch_first=True, padding_value=-100
        )
        tokenized_inputs["detection_labels"] = pad_sequence(
            detection_labels_batch_processed, batch_first=True, padding_value=-100
        )
        return tokenized_inputs

    def setup(self, stage: str) -> None:
        self.train_ds = self._create_dataset(self.cfg.train)
        self.val_ds = self._create_dataset(self.cfg.val)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            collate_fn=partial(
                self._collate_fn_template,
                label_all_tokens=self.cfg.train.label_all_tokens,
            ),
            **self.cfg.train.dataloader_params,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            collate_fn=partial(
                self._collate_fn_template,
                label_all_tokens=self.cfg.val.label_all_tokens,
            ),
            **self.cfg.val.dataloader_params,
        )
