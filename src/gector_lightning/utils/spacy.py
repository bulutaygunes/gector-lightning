from functools import singledispatch
from typing import Literal

import spacy
from spacy.tokens import Token

nlp = spacy.load(
    "en_core_web_sm", exclude=["tagger", "parser", "ner", "lemmatizer", "textcat"]
)


@singledispatch
def tokenize(
    text: str | list[str],
    batch_size: int = 32,
    return_type: Literal["text", "token"] = "text",
):
    raise NotImplementedError("Unsupported type")


@tokenize.register
def _(
    text: str,
    batch_size: int = 32,
    return_type: Literal["text", "token"] = "text",
) -> list[str] | list[Token]:
    return [token.text if return_type == "text" else token for token in nlp(text)]


@tokenize.register(list)
def _(
    text: list[str],
    batch_size: int = 32,
    return_type: Literal["text", "token"] = "text",
) -> list[list[str]] | list[list[Token]]:
    return [
        [token.text if return_type == "text" else token for token in doc]
        for doc in nlp.pipe(text, batch_size=batch_size)
    ]
