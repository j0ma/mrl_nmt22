import pyconll
from pathlib import Path

import click
import orjson


@click.command()
@click.option(
    "--input-file",
    "-i",
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, readable=True, allow_dash=True
    ),
)
def main(input_file):

    for sent in pyconll.iter_from_file(input_file):
        sent_json = {}

        for k in "text", "text_en":
            try:
                sent_json[k] = sent.meta_value(k)
            except KeyError:
                pass

        sent_json["tokens"] = []
        sent_json["lemmas"] = []
        sent_json["msd"] = []
        sent_json["lemma_msd"] = []

        for token in sent:
            _token = token.form
            _lemma = token.lemma

            if token.feats:
                _msd = list(token.feats).pop()
            else:
                _msd = ""

            sent_json["tokens"].append(_token)
            sent_json["lemmas"].append(_lemma)
            sent_json["msd"].append(_msd)

            if _msd:
                sent_json["lemma_msd"].append(f"{_lemma}-{_msd}")
            else:
                sent_json["lemma_msd"].append("")

        print(orjson.dumps(sent_json).decode("utf-8"))


if __name__ == "__main__":
    main()
