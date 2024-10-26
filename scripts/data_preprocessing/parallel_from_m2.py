from pathlib import Path
from typing import Annotated

import typer
from tqdm import tqdm


# noinspection PyShadowingBuiltins
def main(
    m2: Annotated[
        Path,
        typer.Option(
            help="Path to the input m2 file.",
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
            help="Path to save the corrected output text file.",
        ),
    ],
    incorrect: Annotated[
        Path,
        typer.Option(
            "--incorrect",
            "-i",
            help="Path to save the incorrect output text file.",
        ),
    ],
    id: Annotated[
        int, typer.Option(help="The id of the target annotator in the m2 file.")
    ] = 0,
):
    with open(m2, "r") as m2:
        m2_entries = m2.read().strip().split("\n\n")

    # Output files
    out_corr = open(correct, "w")
    out_incorr = open(incorrect, "w")

    # Do not apply edits with these error types
    skip = {"noop", "UNK", "Um"}

    for sent in tqdm(m2_entries):
        sent = sent.split("\n")
        incorr_sent = sent[0].split()[1:]  # Ignore "S "
        cor_sent = incorr_sent.copy()
        edits = sent[1:]
        offset = 0
        for edit in edits:
            edit = edit.split("|||")
            # Ignore certain edits
            if edit[1] in skip:
                continue

            coder = int(edit[-1])

            # Ignore other coders
            if coder != id:
                continue

            span = edit[0].split()[1:]  # Ignore "A "
            start = int(span[0])
            end = int(span[1])
            cor = edit[2].split()
            cor_sent[start + offset : end + offset] = cor
            offset = offset - (end - start) + len(cor)

        out_corr.write(" ".join(cor_sent) + "\n")
        out_incorr.write(" ".join(incorr_sent) + "\n")

    out_corr.close()
    out_incorr.close()


if __name__ == "__main__":
    typer.run(main)
