from typing import Union, Iterable
from pathlib import Path
import io

import attr
from mrl_nmt.utils import read_lines
import sentencepiece as spm


def process_with_sentencepiece(
    lines: Iterable[str],
    vocab_size: int,
    use_pretrained_model: bool = False,
    model_file: Union[str, Path] = "",
    model_base_path: Union[str, Path] = "",
    input_sentence_size: int = 0,
    shuffle_input_sentence: bool = False,
) -> Iterable[str]:

    # create an iterator
    lines_iterator = iter(lines)

    model_file = (Path(model_base_path) / Path(model_file)).expanduser()

    if not model_file.parent.exists():
        model_file.parent.mkdir(parents=True)

    model_file = str(model_file)

    if use_pretrained_model:
        sp = spm.SentencePieceProcessor(model_file=model_file)

    else:

        # create a BytesIO object to hold the model
        model = io.BytesIO()

        # train the model
        spm.SentencePieceTrainer.train(
            sentence_iterator=lines_iterator,
            model_writer=model,
            vocab_size=vocab_size,
            input_sentence_size=input_sentence_size,
            shuffle_input_sentence=shuffle_input_sentence,
        )

        # write model to disk
        if model_file:
            with open(model_file, "wb") as f:
                f.write(model.getvalue())

        # apply the model to text
        sp = spm.SentencePieceProcessor(model_proto=model.getvalue())

    segmented_lines = [" ".join(tokens) for tokens in sp.encode(lines, out_type=str)]

    return segmented_lines
