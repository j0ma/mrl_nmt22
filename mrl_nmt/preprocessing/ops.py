import io
from pathlib import Path
from typing import Union, Iterable, Optional, Dict, Any

import sentencepiece as spm
from tqdm import tqdm

import mrl_nmt.preprocessing.corpora as crp

SPACE_SYMBOL = "﹏"
OUTPUT_LEVELS = {"word", "sentencepiece", "morph", "char"}

#### Language pair/dataset specific ops


def load_flores101_pair(
    folder: Path, src_language: str, tgt_language: str, split: str = "train", prefix=""
):

    lang_code_map = {
        "en": "eng",
        "cs": "ces",
        "de": "deu",
        "fi": "fin",
        "ru": "rus",
        "tr": "tur",
        "uz": "uzb",
    }
    # Handle paths
    src_path = (folder / f"{prefix}{lang_code_map[src_language]}.{split}").expanduser()
    tgt_path = (folder / f"{prefix}{lang_code_map[tgt_language]}.{split}").expanduser()

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
        src_column="en",
        src_language="en",
        tgt_column=foreign_language,
        tgt_language=foreign_language,
        fieldnames=["en", foreign_language],
        load_to_memory=False,
        use_custom_reader=True,
    )

    return crp.CorpusSplit.from_tsv(tsv=tsv, split=split)


def load_europarl_v10(
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
        load_to_memory=False,
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
        load_to_memory=False,
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
        load_to_memory=False,
    )

    return crp.CorpusSplit.from_tsv(tsv=tsv, split=split)


def validate_sentencepiece_config(config: Dict[str, Any]) -> bool:
    required_params = [
        "vocab_size",
        "use_pretrained_model",
        "model_file",
    ]
    print(f"Got config: {config}")
    for param_name in required_params:
        assert (
            param_name in config
        ), f"Could not find '{param_name}' in config dictionary!"


def process_subwords(
    out: crp.CorpusSplit,
    src_output_lvl: str,
    tgt_output_lvl: str,
    sentencepiece_config: Optional[Dict[str, str]],
) -> crp.CorpusSplit:

    assert (
        src_output_lvl in OUTPUT_LEVELS
    ), f"src output level must be one of {OUTPUT_LEVELS}. Got: {src_output_lvl}"
    assert (
        tgt_output_lvl in OUTPUT_LEVELS
    ), f"tgt output level must be one of {OUTPUT_LEVELS}. Got: {tgt_output_lvl}"

    for side, output_level in zip(("src", "tgt"), (src_output_lvl, tgt_output_lvl)):
        print(
            f'[process_subwords] Processing {side} side into output level "{output_level}"'
        )
        if output_level == "sentencepiece":
            if not sentencepiece_config or side not in sentencepiece_config:
                raise ValueError(
                    "SentencePiece requires a config dictionary with src and tgt as the top level keys. "
                    f"Got: {sentencepiece_config}"
                )
            sp_conf = sentencepiece_config[side]
            validate_sentencepiece_config(sp_conf)
            out = process_with_sentencepiece(corpus=out, side=side, **sp_conf)
        elif output_level == "char":
            out = convert_to_chars(out, side=side)
        elif output_level == "morph":
            raise NotImplementedError("Morphological preprocessing not yet supported.")

    return out


def process_cs(
    input_base_folder,
    split="train",
    cs_output_level="word",
    en_output_level="word",
    sentencepiece_config=None,
) -> crp.CorpusSplit:

    if split == "train":
        print("Loading CommonCrawl...")
        commoncrawl_train = load_commoncrawl(
            folder=input_base_folder, src_language="en", tgt_language="cs", split=split
        )
        print("Loading ParaCrawl...")
        paracrawl_train = load_paracrawl(
            folder=input_base_folder, foreign_language="cs", split=split
        )
        print("Loading Europarl v10...")
        europarl_train = load_europarl_v10(
            folder=input_base_folder, src_language="en", tgt_language="cs", split=split
        )
        print("Loading NewsCommentary...")
        news_commentary_train = load_news_commentary(
            folder=input_base_folder, src_language="en", tgt_language="cs", split=split
        )

        print("Loading TILDE RAPID corpus...")
        rapid_train = load_rapid_2019_xlf(
            folder=input_base_folder, src_language="en", tgt_language="cs", split=split
        )

        print("Stacking everything together...")
        out = crp.CorpusSplit.stack_corpus_splits(
            corpus_splits=[
                commoncrawl_train,
                paracrawl_train,
                europarl_train,
                news_commentary_train,
                rapid_train,
            ],
            split="train",
        )

    elif split == "dev":
        print("Loading dev data from FLORES")
        out = load_flores101_pair(
            folder=input_base_folder, src_language="en", tgt_language="cs", split="dev"
        )

    else:
        raise ValueError("Only train and dev sets supported!")

    out = process_subwords(
        out=out,
        src_output_lvl=en_output_level,
        tgt_output_lvl=cs_output_level,
        sentencepiece_config=sentencepiece_config,
    )

    return out


def process_de():
    pass


def process_fi():
    pass


def process_iu():
    pass


def process_ru():
    pass


def process_tr(
    input_base_folder,
    split="train",
    tr_output_level="word",
    en_output_level="word",
    sentencepiece_config=None,
    prefix="",
) -> crp.CorpusSplit:

    # get train
    print(f"[process_tr] Loading {split} corpus...")
    if split == "train":
        train_path = Path(f"{input_base_folder}/")
        train_en = crp.LoadedTextFile(
            train_path / "en-tr.en", side="src", language="en", load_to_memory=False
        )
        train_tr = crp.LoadedTextFile(
            train_path / "en-tr.tr", side="tgt", language="tr", load_to_memory=False
        )

        print(f"[process_tr] Creating CorpusSplit...")
        out = crp.CorpusSplit.from_src_tgt(
            src=train_en, tgt=train_tr, split=split, verbose=True
        )

        # get dev
    elif split == "dev":
        out = load_flores101_pair(
            folder=input_base_folder,
            src_language="en",
            tgt_language="tr",
            split="dev",
            prefix=prefix,
        )

    else:
        raise ValueError("Only train and dev sets supported!")

    print(f"[process_tr] Processing corpus into subword representations")
    out = process_subwords(
        out,
        src_output_lvl=en_output_level,
        tgt_output_lvl=tr_output_level,
        sentencepiece_config=sentencepiece_config,
    )
    return out


def process_uz(
    input_base_folder,
    split="train",
    uz_output_level="word",
    en_output_level="word",
    sentencepiece_config=None,
    prefix="",
) -> crp.CorpusSplit:

    if split == "train":
        train_path = Path(f"{input_base_folder}/")
        train_en = crp.LoadedTextFile(
            train_path / "en-uz.en", side="src", language="en", load_to_memory=False
        )
        train_uz = crp.LoadedTextFile(
            train_path / "en-uz.uz", side="tgt", language="uz", load_to_memory=False
        )
        out = crp.CorpusSplit.from_src_tgt(
            src=train_en, tgt=train_uz, split=split, verbose=True
        )

        # get dev
    elif split == "dev":
        out = load_flores101_pair(
            folder=input_base_folder,
            src_language="en",
            tgt_language="uz",
            split="dev",
            prefix=prefix,
        )

    else:
        raise ValueError("Only train and dev sets supported!")

    out = process_subwords(
        out,
        src_output_lvl=en_output_level,
        tgt_output_lvl=uz_output_level,
        sentencepiece_config=sentencepiece_config,
    )
    return out


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
                # if not a sequence, convert to list :(
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
    model_base_path: Union[str, Path] = "/tmp",
    input_sentence_size: int = 0,
    shuffle_input_sentence: bool = False,
) -> crp.CorpusSplit:
    """Process one side of a CorpusSplit with SentencePiece

    Notes:
    - With use_pretrained_model=False a new model will be trained
    and optionally saved if model_file argument was passed in
    - With use_pretrained_model=True, a pre-trained model will be loaded
    from model_path
    """
    model_file_is_correct = bool(str(model_file))

    # load lines to RAM (sad)
    other_side = "src" if side == "tgt" else "tgt"
    lines = []
    lines_str = []
    other_side_lines = []
    line_count = 0
    for ld in corpus.lines:
        side_ld = ld[side]
        lines.append(side_ld)
        lines_str.append(side_ld["text"])
        other_side_lines.append(ld[other_side])
        line_count += 1

    assert len(lines) == len(
        other_side_lines
    ), f"Line length mismatch: {len(lines)} ({side}) != {len(other_side_lines)} ({other_side})"

    model_file = (Path(model_base_path) / Path(model_file)).expanduser()

    if not model_file.parent.exists():
        model_file.parent.mkdir(parents=True)

    if use_pretrained_model:
        assert model_file.exists(), "Must specify a valid model file."
        model_file = str(model_file)
        print(f"Applying SP model from {model_file}...")
        sp = spm.SentencePieceProcessor(model_file=model_file)

    else:

        # create a BytesIO object to hold the model
        model = io.BytesIO()

        # train the model
        spm.SentencePieceTrainer.train(
            sentence_iterator=iter(lines_str),
            model_writer=model,
            vocab_size=vocab_size,
            input_sentence_size=input_sentence_size,
            shuffle_input_sentence=shuffle_input_sentence,
        )

        if model_file_is_correct:
            # optionally write model to disk
            print(
                f"[process_with_sentencepiece] Writing model binary to disk under {model_file}"
            )
            with open(model_file, "wb") as f:
                f.write(model.getvalue())

        # apply the model to text
        sp = spm.SentencePieceProcessor(model_proto=model.getvalue())

    def final_lines(sp, lines, other_side_lines, line_count):
        print(f"[process_with_sentencepiece] Outputting final lines...")
        for side_line_dict, other_line_dict in tqdm(
            zip(lines, other_side_lines), total=line_count
        ):

            new_line = " ".join(sp.encode(side_line_dict["text"], out_type=str))
            side_line_dict["text"] = new_line

            yield {side: side_line_dict, other_side: other_line_dict}

    output = list(final_lines(sp, lines, other_side_lines, line_count))

    assert (
        len(output) == len(lines) == line_count
    ), "Line length mismatch: {len(lines)} (before) != {len(output)} (after)"

    return crp.CorpusSplit(
        lines=output,
        src_lang=corpus.src_lang,
        tgt_lang=corpus.tgt_lang,
        split=corpus.split,
        verbose=corpus.verbose,
    )
