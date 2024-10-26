from pathlib import Path
from typing import Annotated

import typer

from gector_lightning.utils.helpers import read_parallel_lines
from gector_lightning.utils.preprocess_data import convert_data_from_raw_files


def main(
    incorrect: Annotated[
        Path,
        typer.Option(
            "--incorrect",
            "-i",
            help="Path to the incorrect file",
            exists=True,
            file_okay=True,
            dir_okay=False,
        ),
    ],
    correct: Annotated[
        Path,
        typer.Option(
            "--correct",
            "-c",
            help="Path to the correct file",
            exists=True,
            file_okay=True,
            dir_okay=False,
        ),
    ],
    output_file: Annotated[
        Path,
        typer.Option(
            "--output_file",
            "-o",
            help="Path to the output file",
        ),
    ],
    chunk_size: Annotated[
        int,
        typer.Option(help="Dump each chunk size."),
    ] = 1_000_000,
):
    """Convert parallel files into GECToR format."""

    source_data, target_data = read_parallel_lines(incorrect, correct)
    convert_data_from_raw_files(source_data, target_data, output_file, chunk_size)


if __name__ == "__main__":
    typer.run(main)
