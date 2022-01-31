from pathlib import Path
import time
import sys
import io

from rich import print
from rich.panel import Panel
from rich.prompt import Confirm

import sentencepiece as spm
import mrl_nmt.utils as u
import click

SPM_MODEL_TYPES = ["unigram", "bpe"]


@click.command()
@click.option("--model-type", type=click.Choice(SPM_MODEL_TYPES), required=True)
@click.option("--vocab-size", required=True)
@click.option("--input-sentence-size", default=5000000)
@click.option("--shuffle-input-sentence/--no-shuffle-input-sentence", default=False)
@click.option("--extremely-large-corpus", is_flag=True, default=False)
@click.option(
    "--input-file", default="-", type=click.Path(readable=True, allow_dash=True)
)
@click.option(
    "--output-file", required=True, type=click.Path(writable=True, allow_dash=False)
)
def main(
    model_type,
    vocab_size,
    input_sentence_size,
    shuffle_input_sentence,
    extremely_large_corpus,
    input_file,
    output_file,
):
    model = io.BytesIO()

    model_train_msg = """
About to train the following model:
Model type: {}
Vocab size: {}
Number of sentences to load: {}
Shuffle? {}
Extremely large corpus? {}

Input file: {}
Output file: {}

Press CTRL-C to abort in 5 seconds...
""".format(
        model_type,
        vocab_size,
        input_sentence_size,
        shuffle_input_sentence,
        extremely_large_corpus,
        input_file,
        output_file,
    )
    print(Panel(model_train_msg), file=sys.stderr)

    time.sleep(5)

    with open(
        input_file, "r", encoding="utf-8"
    ) if input_file != "-" else sys.stdin as fin:
        spm.SentencePieceTrainer.train(
            # We read from standard input...
            sentence_iterator=iter(fin),
            # And write to a variable in RAM...
            model_writer=model,
            # Make sure we preserve raw UTF bytes
            normalization_rule_name="identity",
            # Leave extra whitespace as is
            remove_extra_whitespaces=False,
            # Making sure our character vocab is in order
            character_coverage=1.0,  # Make sure to cover all characters
            byte_fallback=True,  # Segment UNKs into Unicode bytes
            # User defined options
            model_type=model_type,
            vocab_size=vocab_size,
            input_sentence_size=input_sentence_size,
            shuffle_input_sentence=shuffle_input_sentence,
            train_extremely_large_corpus=extremely_large_corpus
        )

        # Write model to disk
        with open(output_file, "wb") as fout:
            fout.write(model.getvalue())


if __name__ == "__main__":
    main()
