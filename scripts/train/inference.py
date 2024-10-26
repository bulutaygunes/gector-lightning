import itertools
import os
import os.path as op
from collections import defaultdict
from pathlib import Path
from typing import Annotated, cast

import numpy as np
import typer
from errant.commands.compare_m2 import computeFScore
from tqdm import tqdm

from gector_lightning.models import GectorModel, load_model
from gector_lightning.utils.errant import print_results, score, to_m2
from gector_lightning.utils.spacy import tokenize as spacy_tokenize

app = typer.Typer()


DEFAULT_BATCH_SIZE = 32
DEFAULT_DEVICE = "cpu"

ConfigPathT = Annotated[
    Path,
    typer.Option(
        "--config_path",
        "-c",
        help="Path to the config file. E.g. conf/inference.yaml",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
]


def batch_lines(lines: list[str], batch_size: int) -> list[list[list[str]]]:
    lines = [line.split() for line in lines]  # Split into tokens
    return [lines[i : i + batch_size] for i in range(0, len(lines), batch_size)]


def predict_batched(
    model: GectorModel, lines_batched: list[list[list[str]]], progress_bar: bool = True
) -> list[str]:
    out = []
    for batch in tqdm(lines_batched) if progress_bar else lines_batched:
        preds, _ = model.handle_batch(batch)
        out.extend([" ".join(pred) for pred in preds])
    return out


def read_lines(input_path: Path) -> list[str]:
    with open(input_path, "r") as f:
        if input_path.suffix == ".m2":
            m2_entries = f.read().strip().split("\n\n")
            lines = [
                sent.split("\n")[0].strip().split(maxsplit=1)[1] for sent in m2_entries
            ]
        else:
            lines = [line.strip() for line in f]
    return lines


@app.command()
def predict(
    input_path: Annotated[
        Path,
        typer.Option(
            "--input_path",
            "-i",
            help=".txt or .m2",
            exists=True,
            file_okay=True,
            dir_okay=False,
        ),
    ],
    output_path: Annotated[Path, typer.Option("--output_path", "-o")],
    config_path: ConfigPathT,
    tokenize: Annotated[
        bool,
        typer.Option(
            help="Tokenize inputs with spaCy before feeding them to the model."
        ),
    ],
    device: str = DEFAULT_DEVICE,
    batch_size: int = DEFAULT_BATCH_SIZE,
):
    model = cast(GectorModel, load_model(GectorModel, config_path, device))
    lines = read_lines(input_path)

    if tokenize:
        lines = spacy_tokenize(lines, batch_size)
        lines = [" ".join(line) for line in lines]

    lines_batched = batch_lines(lines, batch_size)
    out = predict_batched(model, lines_batched)

    # Create output directories (if doesn't exist)
    directory = op.split(output_path)[0]
    if directory:
        os.makedirs(directory, exist_ok=True)

    out = [line + "\n" for line in out]
    with open(output_path, "w") as f:
        f.writelines(out)


@app.command()
def evaluate(
    input_path: Annotated[
        Path,
        typer.Option(
            "--input_path",
            "-i",
            help="Reference .m2 file (e.g. conll14-errant-auto.m2)",
            exists=True,
            file_okay=True,
            dir_okay=False,
        ),
    ],
    config_path: ConfigPathT,
    device: str = DEFAULT_DEVICE,
    batch_size: int = DEFAULT_BATCH_SIZE,
):
    model = cast(GectorModel, load_model(GectorModel, config_path, device))
    lines = read_lines(input_path)
    lines_batched = batch_lines(lines, batch_size)

    with open(input_path, "r") as f:
        ref_m2 = f.read().strip().split("\n\n")

    lines_predicted = predict_batched(model, lines_batched)
    hyp_m2 = to_m2(list(zip(lines, lines_predicted)))

    # Score calculation
    best_dict, best_cats = score(hyp_m2, ref_m2)
    print_results(best_dict, best_cats)


@app.command()
def grid_search(
    input_path: Annotated[
        Path,
        typer.Option(
            "--input_path",
            "-i",
            help="Reference .m2 file (e.g. conll14-errant-auto.m2)",
            exists=True,
            file_okay=True,
            dir_okay=False,
        ),
    ],
    config_path: ConfigPathT,
    device: str = DEFAULT_DEVICE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    min_correction_confidence_start: float = 0.0,
    min_correction_confidence_end: float = 0.8,
    min_error_probability_start: float = 0.0,
    min_error_probability_end: float = 0.8,
    additional_keep_confidence_start: float = 0.0,
    additional_keep_confidence_end: float = 0.8,
    num_samples_per_param: int = 10,
):
    model = cast(GectorModel, load_model(GectorModel, config_path, device))
    lines = read_lines(input_path)
    lines_batched = batch_lines(lines, batch_size)

    with open(input_path, "r") as f:
        ref_m2 = f.read().strip().split("\n\n")

    # Prepare the search space
    min_correction_confidence_values = np.linspace(
        min_correction_confidence_start,
        min_correction_confidence_end,
        num_samples_per_param,
    )
    min_error_probability_values = np.linspace(
        min_error_probability_start, min_error_probability_end, num_samples_per_param
    )
    additional_keep_confidence_values = np.linspace(
        additional_keep_confidence_start,
        additional_keep_confidence_end,
        num_samples_per_param,
    )
    param_list = list(
        itertools.product(
            min_correction_confidence_values,
            min_error_probability_values,
            additional_keep_confidence_values,
        )
    )

    # Grid search loop over hparams
    scores = defaultdict(list)
    for (
        min_correction_confidence,
        min_error_probability,
        additional_keep_confidence,
    ) in tqdm(param_list):
        model.min_correction_confidence = min_correction_confidence
        model.min_error_probability = min_error_probability
        model.additional_keep_confidence = additional_keep_confidence

        lines_predicted = predict_batched(model, lines_batched, progress_bar=False)
        hyp_m2 = to_m2(list(zip(lines, lines_predicted)))

        # Score calculation
        best_dict, best_cats = score(hyp_m2, ref_m2)
        precision, recall, f_0_5 = computeFScore(
            best_dict["tp"], best_dict["fp"], best_dict["fn"], beta=0.5
        )
        scores["precision"].append(precision)
        scores["recall"].append(recall)
        scores["f_0_5"].append(f_0_5)

    # Print best precision, recall, F0.5 and corresponding params
    for metric in ("precision", "recall", "f_0_5"):
        best_score = max(scores[metric])
        best_idx = scores[metric].index(best_score)
        (
            best_min_correction_confidence,
            best_min_error_probability,
            best_additional_keep_confidence,
        ) = param_list[best_idx]

        print(f"Best {metric.upper()}: {best_score}")
        print(f"min_correction_confidence: {best_min_correction_confidence}")
        print(f"min_error_probability: {best_min_error_probability}")
        print(f"additional_keep_confidence: {best_additional_keep_confidence}")
        print()


if __name__ == "__main__":
    app()
