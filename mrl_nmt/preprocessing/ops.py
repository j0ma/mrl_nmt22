from typing import Union, Iterable, Optional, Dict, Any, Sequence
from pathlib import Path
import io

import attr
from mrl_nmt.utils import read_lines
import mrl_nmt.preprocessing.corpora as crp
import sentencepiece as spm

SPACE_SYMBOL = "﹏"

#### Language pair/dataset specific ops


def load_flores101_pair(
    folder: Path, src_language: str, tgt_language: str, split: str = "train"
):

    lang_code_map = {
        "en": "eng",
        "cs": "ces",
        "de": "deu",
        "fi": "fin",
        "ru": "rus",
    }
    # Handle paths
    src_path = (folder / f"{lang_code_map[src_language]}.{split}").expanduser()
    tgt_path = (folder / f"{lang_code_map[tgt_language]}.{split}").expanduser()

    # Load files
    f_src = crp.LoadedTextFile(
        path=src_path, language=src_language, side="src", load_to_memory=False
    )
    f_tgt = crp.LoadedTextFile(
        path=tgt_path, language=tgt_language, side="tgt", load_to_memory=False
    )

    # Create corpus split
    return crp.CorpusSplit.from_src_tgt(src=f_src, tgt=f_tgt, split=split)


def load_commoncrawl(
    folder: Path, src_language: str, tgt_language: str, split: str = "train"
):

    # Handle paths
    pair = f"{src_language}-{tgt_language}"
    src_path = (folder / f"commoncrawl.{pair}.{src_language}").expanduser()
    tgt_path = (folder / f"commoncrawl.{pair}.{tgt_language}").expanduser()

    # Load files
    f_src = crp.LoadedTextFile(
        path=src_path, language=src_language, side="src", load_to_memory=False
    )
    f_tgt = crp.LoadedTextFile(
        path=tgt_path, language=tgt_language, side="tgt", load_to_memory=False
    )

    # Create corpus split
    return crp.CorpusSplit.from_src_tgt(src=f_src, tgt=f_tgt, split=split)


def load_paracrawl(folder: Path, foreign_language: str, split: str = "train"):
    """Load ParaCrawl data into X-EN format where X = foreign language"""
    paracrawl_path = (folder / f"en-{foreign_language}.txt").expanduser()
    tsv = crp.LoadedTSVFile(
        path=paracrawl_path,
        side="both",
        src_column=foreign_language,
        tgt_column="en",
        src_language=foreign_language,
        tgt_language="en",
        fieldnames=["en", foreign_language],
    )

    return crp.CorpusSplit.from_tsv(tsv=tsv, split=split)


def load_europarl(
    folder: Path, src_language: str, tgt_language: str, split: str = "train"
):

    # Handle paths
    europarl_path = (
        folder / f"europarl-v10.{src_language}-{tgt_language}.tsv"
    ).expanduser()

    tsv = crp.LoadedTSVFile(
        path=europarl_path,
        side="both",
        src_language=src_language,
        tgt_language=tgt_language,
        src_column=src_language,
        tgt_column=tgt_language,
        fieldnames=[src_language, tgt_language],
    )

    return crp.CorpusSplit.from_tsv(tsv=tsv, split=split)


def load_news_commentary(
    folder: Path, src_language: str, tgt_language: str, split: str = "train"
):
    pair = f"{src_language}-{tgt_language}"
    news_commentary_path = (folder / f"news-commentary-v16.{pair}.tsv").expanduser()
    tsv = crp.LoadedTSVFile(
        path=news_commentary_path,
        side="both",
        src_language=src_language,
        tgt_language=tgt_language,
        src_column=src_language,
        tgt_column=tgt_language,
        fieldnames=[src_language, tgt_language],
    )

    return crp.CorpusSplit.from_tsv(tsv=tsv, split=split)


def load_rapid_2019_xlf(
    folder: Path, src_language: str, tgt_language: str, split: str = "train"
) -> crp.CorpusSplit:
    pair = f"{src_language}-{tgt_language}"
    rapid_2019_path = (folder / f"RAPID_2019.{pair}.xlf").expanduser()
    tsv = crp.LoadedXLIFFFile(
        path=rapid_2019_path,
        side="both",
        src_language=src_language,
        tgt_language=tgt_language,
    )

    return crp.CorpusSplit.from_tsv(tsv=tsv, split=split)


def process_cs_en(input_base_folder, split="train") -> crp.CorpusSplit:
    # get train
    if split == "train":
        commoncrawl_train = load_commoncrawl(
            folder=input_base_folder, src_language="cs", tgt_language="en", split=split
        )
        paracrawl_train = load_paracrawl(
            folder=input_base_folder, foreign_language="cs", split=split
        )
        europarl_train = load_europarl(
            folder=input_base_folder, src_language="cs", tgt_language="en", split=split
        )
        news_commentary_train = load_news_commentary(
            folder=input_base_folder, src_language="cs", tgt_language="en", split=split
        )

        rapid_train = load_rapid_2019_xlf(
            folder=input_base_folder, src_language="cs", tgt_language="en", split=split
        )

        train = crp.CorpusSplit.stack_corpus_splits(
            corpus_splits=[
                commoncrawl_train,
                paracrawl_train,
                europarl_train,
                news_commentary_train,
                rapid_train,
            ],
            split="train",
        )

        # TODO: bpe / char processing

        return train
    elif split == "dev":
        flores_dev = load_flores101_pair(
            folder=input_base_folder, src_language="cs", tgt_language="en", split="dev"
        )

        # TODO: bpe / char processing

        return flores_dev

    else:
        raise ValueError("Only train and dev sets supported!")


def process_de_en():
    pass


def process_fi_en():
    pass


def process_iu_en():
    pass


def process_ru_en():
    pass


def process_tr_en(*args, **kwargs) -> None:

    # get train
    train_en = crp.LoadedTextFile()
    train_tr = crp.LoadedTextFile()

    # get dev


def process_uz_en():
    pass


####


def duplicate_lines(
    corpus: crp.CorpusSplit, n: int = 1, round_robin: bool = True
) -> crp.CorpusSplit:
    """Produces a new corpus with each line repeated n times.
    With round_robin=False, each line is repeated n_times.
    With round_robin=True, lines are concatenated n times (requires loading to RAM).
    """
    if round_robin:

        def duplicated(lines: Iterable[Any], n: int) -> Iterable[Any]:
            try:
                length = len(lines)
                line_iter = lines
            except:
                # if not a sequence, convert to list
                line_iter = list(lines)
            for _ in range(n):
                for line in line_iter:
                    yield line

    else:

        def duplicated(lines: Iterable[Any], n: int) -> Iterable[Any]:
            for line in lines:
                for _ in range(n):
                    yield line

    return crp.CorpusSplit(
        lines=duplicated(corpus.lines, n),
        split=corpus.split,
        src_lang=corpus.src_lang,
        tgt_lang=corpus.tgt_lang,
        verbose=corpus.verbose,
    )


def convert_to_chars(corpus: crp.CorpusSplit, side: str) -> crp.CorpusSplit:
    """Converts one side of corpus to characters."""
    # other_side = {"src": "tgt", "tgt": "src"}[side]

    def to_char_level(s: str, space_symbol: str = SPACE_SYMBOL) -> str:
        return " ".join(space_symbol if c == " " else c for c in s)

    def lines_as_chars(
        line_dict: Dict[str, Optional[Dict[str, str]]], side: str
    ) -> Dict[str, Optional[Dict[str, str]]]:
        """Modifies lines in-place and returns"""
        line_dict[side]["text"] = to_char_level(line_dict[side]["text"])
        return line_dict

    return crp.CorpusSplit(
        lines=(lines_as_chars(ld, side=side) for ld in corpus.lines),
        split=corpus.split,
        src_lang=corpus.src_lang,
        tgt_lang=corpus.tgt_lang,
        verbose=corpus.verbose,
    )


def process_with_sentencepiece(
    corpus: crp.CorpusSplit,
    side: str,
    vocab_size: int,
    use_pretrained_model: bool = False,
    model_file: Union[str, Path] = "",
    model_base_path: Union[str, Path] = "",
    input_sentence_size: int = 0,
    shuffle_input_sentence: bool = False,
) -> Iterable[str]:

    # load lines to RAM (sad)
    lines = [ld[side]["text"] for ld in corpus.lines]

    model_file = (Path(model_base_path) / Path(model_file)).expanduser()

    if not model_file.parent.exists():
        model_file.parent.mkdir(parents=True)

    if use_pretrained_model:
        model_file = str(model_file)
        sp = spm.SentencePieceProcessor(model_file=model_file)

    else:

        # create a BytesIO object to hold the model
        model = io.BytesIO()

        # train the model
        spm.SentencePieceTrainer.train(
            sentence_iterator=iter(lines),
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
