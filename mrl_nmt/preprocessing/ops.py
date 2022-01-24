import io
import os
from pathlib import Path
import multiprocessing as mp
from typing import Union, Iterable, Optional, Dict, Any
import itertools as it
import functools as ft
import math
import time

from sacremoses import MosesDetokenizer
import sentencepiece as spm
from tqdm import tqdm
import shutil

import mrl_nmt.preprocessing as pp
import mrl_nmt.preprocessing.corpora as crp
import mrl_nmt.utils as u
from mrl_nmt.utils import SPACE_SYMBOL


OUTPUT_LEVELS = {"word", "sentencepiece", "morph", "char", "bpe", "none"}
MOSES_DEFAULTS = {"min_len": 1, "max_len": 80, "ratio": 9}

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
        expected_n_fields=2,
        validate_lines=True,
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
        expected_n_fields=2,
        validate_lines=True,
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
        expected_n_fields=2,
        validate_lines=True,
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
        "use_pretrained_model",
        "model_file",
    ]
    print(f"Got config: {config}")

    for param_name in required_params:
        assert (
            param_name in config
        ), f"Could not find '{param_name}' in config dictionary!"

    if not config["use_pretrained_model"]:
        assert (
            "vocab_size" in config
        ), "vocab_size required when not using pretrained model."


def process_subwords(
    out: crp.CorpusSplit,
    src_output_lvl: str,
    tgt_output_lvl: str,
    sentencepiece_config: Optional[Dict[str, str]],
    n_workers: int = 1
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

        if output_level in ["sentencepiece", "bpe"]:
            if not sentencepiece_config or side not in sentencepiece_config:
                raise ValueError(
                    "SentencePiece requires a config dictionary with src and tgt as the top level keys. "
                    f"Got: {sentencepiece_config}"
                )
            sp_conf = sentencepiece_config[side]
            validate_sentencepiece_config(sp_conf)
            out = process_with_sentencepiece(corpus=out, side=side, n_workers=n_workers, **sp_conf)
        elif output_level == "char":
            out = convert_to_chars(out, side=side)
        elif output_level == "morph":
            raise NotImplementedError("Morphological preprocessing not yet supported.")

    return out


def _detok(l):
    return MosesDetokenizer().detokenize(l.split(" "))


def process_download(
    input_base_folder,
    src_lang,
    tgt_lang,
    split="train",
    src_output_level="word",
    tgt_output_level="word",
    sentencepiece_config=None,
    kind="mtdata",
    write_detokenized=False,
    detokenized_output_path="",
    detokenized_copy_only=False,
    detokenized_link_only=False,
    use_clean_corpus_n_perl=False,
    moses_config=None,
    n_workers=(os.cpu_count() - 4),
    chunksize=10000,
) -> crp.CorpusSplit:
    """Processes TIL / MTData download into a CorpusSplit object"""

    if not use_clean_corpus_n_perl and not moses_config:
        moses_config = MOSES_DEFAULTS

    input_base_path = Path(input_base_folder)
    downloads = input_base_path / "download"

    if kind == "til":
        pair = f"{src_lang}-{tgt_lang}"
        src_path, tgt_path = (
            downloads / split / pair / f"{pair}.{src_lang}",
            downloads / split / pair / f"{pair}.{tgt_lang}",
        )
    elif kind == "phomt":
        print("Loading PhoMT corpus...")
        src_path, tgt_path = (
            input_base_path / f"{split}.{src_lang}",
            input_base_path / f"{split}.{tgt_lang}",
        )
    elif kind == "mtdata":
        src_lang_long, tgt_lang_long = (
            u.get_long_lang_name(src_lang),
            u.get_long_lang_name(tgt_lang),
        )
        # mtdata doesn't merge test files when handling recipes, work around this and take test1
        _split = split if split != "test" else "test1"
        src_path, tgt_path = (
            downloads / f"{_split}.{src_lang_long}",
            downloads / f"{_split}.{tgt_lang_long}",
        )
    else:
        raise ValueError(f"Expected kind=mtdata or til but got {kind}")

    if use_clean_corpus_n_perl:
        cleaner = pp.MosesCleanCorpusNProcessor(**moses_config)

        src_path_clean = f"{src_path}.clean"
        tgt_path_clean = f"{tgt_path}.clean"
        print("[process_download] Cleaning corpus using clean-corpus-n.perl...")

        cleaner.process_files(
            src_input_file=src_path,
            tgt_input_file=tgt_path,
            src_output_file=src_path_clean,
            tgt_output_file=tgt_path_clean,
        )

        p_src, p_tgt = src_path_clean, tgt_path_clean
    else:
        p_src, p_tgt = src_path, tgt_path

    print(f"Loading src text file from {src_path}")
    f_src = crp.LoadedTextFile(
        path=p_src, side="src", load_to_memory=False, language=src_lang
    )
    print(f"Loading tgt text file from {tgt_path}")
    f_tgt = crp.LoadedTextFile(
        path=p_tgt, side="tgt", load_to_memory=False, language=tgt_lang
    )

    print("Creating corpus split...")
    out = crp.CorpusSplit.from_src_tgt(f_src, f_tgt, split=split, verbose=True)

    if write_detokenized:
        assert detokenized_output_path, "Unset argument: detokenized_output_path"

        detokenized_output_path = Path(detokenized_output_path).expanduser().resolve()
        if not detokenized_output_path.exists():
            print(f"Creating folder: {detokenized_output_path}")
            detokenized_output_path.mkdir(exist_ok=True, parents=True)

        src_output_path = (
            detokenized_output_path / f"{src_lang}-{tgt_lang}.{split}.detok.{src_lang}"
        )
        tgt_output_path = (
            detokenized_output_path / f"{src_lang}-{tgt_lang}.{split}.detok.{tgt_lang}"
        )

        if detokenized_copy_only:
            # This can be useful when we only need to copy a single file per side
            for p, output_path in [(p_src, src_output_path), (p_tgt, tgt_output_path)]:
                shutil.copy(src=p, dst=output_path)

        elif detokenized_link_only:
            # Alternatively we can just create symlinks to the raw data if it's not pretokenized
            for p, output_path in [(p_src, src_output_path), (p_tgt, tgt_output_path)]:
                u.create_symlink(link_path=output_path, dest_path=p)

        else:
            # As a last resort, we can write the data out using Moses detokenizer
            for f, output_path in [(f_tgt, src_output_path), (f_tgt, tgt_output_path)]:

                with mp.Pool(n_workers) as pool:
                    print(
                        f"Detokenizing using {n_workers} cores and chunksize {chunksize}"
                    )
                    t_start = time.time()
                    detok_lines = pool.map(_detok, f.stream_lines, chunksize=chunksize)
                    t_end = time.time()
                    print(f"Time elapsed: {round(t_end - t_start, 3)}s")
                    print(f"Writing detokenized lines to {output_path}")
                    u.write_lines(
                        path=output_path,
                        lines=detok_lines,
                    )

    print("Processing subwords...")
    out = process_subwords(
        out=out,
        src_output_lvl=src_output_level,
        tgt_output_lvl=tgt_output_level,
        sentencepiece_config=sentencepiece_config,
        n_workers=n_workers
    )

    return out


def process_vi(
    input_base_folder,
    split="train",
    vi_output_level="word",
    en_output_level="word",
    sentencepiece_config=None,
    prefix="",
    write_detokenized=True,
    detokenized_output_path="",
    detokenized_copy_only=False,
    detokenized_link_only=False,
    phomt=False,
    *args,
    **kwargs,
) -> crp.CorpusSplit:

    return process_download(
        input_base_folder,
        split=split,
        src_lang="en",
        tgt_lang="vi",
        src_output_level=en_output_level,
        tgt_output_level=vi_output_level,
        sentencepiece_config=sentencepiece_config,
        kind="mtdata" if not phomt else "phomt",
        write_detokenized=True,
        detokenized_output_path=detokenized_output_path,
        detokenized_copy_only=detokenized_copy_only,
        detokenized_link_only=detokenized_link_only,
    )


def process_et(
    input_base_folder,
    split="train",
    et_output_level="word",
    en_output_level="word",
    sentencepiece_config=None,
    prefix="",
    write_detokenized=True,
    detokenized_output_path="",
    detokenized_copy_only=False,
    detokenized_link_only=False,
    *args,
    **kwargs,
) -> crp.CorpusSplit:

    return process_download(
        input_base_folder,
        split=split,
        src_lang="en",
        tgt_lang="et",
        src_output_level=en_output_level,
        tgt_output_level=et_output_level,
        sentencepiece_config=sentencepiece_config,
        kind="mtdata",
        write_detokenized=True,
        detokenized_output_path=detokenized_output_path,
        detokenized_copy_only=detokenized_copy_only,
        detokenized_link_only=detokenized_link_only,
    )


def process_cs(
    input_base_folder,
    split="train",
    cs_output_level="word",
    en_output_level="word",
    sentencepiece_config=None,
    prefix="",
    write_detokenized=True,
    detokenized_output_path="",
    detokenized_copy_only=False,
    detokenized_link_only=False,
    *args,
    **kwargs,
) -> crp.CorpusSplit:

    return process_download(
        input_base_folder,
        split=split,
        src_lang="en",
        tgt_lang="cs",
        src_output_level=en_output_level,
        tgt_output_level=cs_output_level,
        sentencepiece_config=sentencepiece_config,
        kind="mtdata",
        write_detokenized=True,
        detokenized_output_path=detokenized_output_path,
        detokenized_copy_only=detokenized_copy_only,
        detokenized_link_only=detokenized_link_only,
    )


def process_tr(
    input_base_folder,
    split="train",
    tr_output_level="word",
    en_output_level="word",
    sentencepiece_config=None,
    prefix="",
    detokenized_output_path="",
    write_detokenized=True,
    *args,
    **kwargs,
) -> crp.CorpusSplit:

    return process_download(
        input_base_folder,
        split=split,
        src_lang="en",
        tgt_lang="tr",
        src_output_level=en_output_level,
        tgt_output_level=tr_output_level,
        sentencepiece_config=sentencepiece_config,
        kind="til",
        write_detokenized=True,
        detokenized_output_path=detokenized_output_path,
        *args,
        **kwargs,
    )


def process_uz(
    input_base_folder,
    split="train",
    uz_output_level="word",
    en_output_level="word",
    sentencepiece_config=None,
    prefix="",
    write_detokenized=True,
    detokenized_output_path="",
    *args,
    **kwargs,
) -> crp.CorpusSplit:

    return process_download(
        input_base_folder,
        split=split,
        src_lang="en",
        tgt_lang="uz",
        src_output_level=en_output_level,
        tgt_output_level=uz_output_level,
        sentencepiece_config=sentencepiece_config,
        kind="til",
        write_detokenized=write_detokenized,
        detokenized_output_path=detokenized_output_path,
        *args,
        **kwargs,
    )


def process_de(
    input_base_folder,
    split="train",
    de_output_level="word",
    en_output_level="word",
    sentencepiece_config=None,
    prefix="",
    write_detokenized=True,
    detokenized_output_path="",
    detokenized_copy_only=False,
    detokenized_link_only=False,
    *args,
    **kwargs,
):
    return process_download(
        input_base_folder,
        split=split,
        src_lang="en",
        tgt_lang="de",
        src_output_level=en_output_level,
        tgt_output_level=de_output_level,
        sentencepiece_config=sentencepiece_config,
        kind="mtdata",
        write_detokenized=write_detokenized,
        detokenized_output_path=detokenized_output_path,
        detokenized_copy_only=detokenized_copy_only,
        detokenized_link_only=detokenized_link_only,
        *args,
        **kwargs,
    )


def process_fi(
    input_base_folder,
    split="train",
    fi_output_level="word",
    en_output_level="word",
    sentencepiece_config=None,
    prefix="",
    write_detokenized=True,
    detokenized_output_path="",
    detokenized_copy_only=False,
    detokenized_link_only=False,
    *args,
    **kwargs,
):

    return process_download(
        input_base_folder,
        split=split,
        src_lang="en",
        tgt_lang="fi",
        src_output_level=en_output_level,
        tgt_output_level=fi_output_level,
        sentencepiece_config=sentencepiece_config,
        kind="mtdata",
        write_detokenized=write_detokenized,
        detokenized_output_path=detokenized_output_path,
        detokenized_copy_only=detokenized_copy_only,
        detokenized_link_only=detokenized_link_only,
        *args,
        **kwargs,
    )


def process_iu(
    input_base_folder,
    split="train",
    iu_output_level="word",
    en_output_level="word",
    sentencepiece_config=None,
    prefix="",
    write_detokenized=True,
    detokenized_output_path="",
    detokenized_copy_only=False,
    detokenized_link_only=False,
    *args,
    **kwargs,
):
    return process_download(
        input_base_folder,
        split=split,
        src_lang="en",
        tgt_lang="iu",
        src_output_level=en_output_level,
        tgt_output_level=iu_output_level,
        sentencepiece_config=sentencepiece_config,
        kind="mtdata",
        write_detokenized=write_detokenized,
        detokenized_output_path=detokenized_output_path,
        detokenized_copy_only=detokenized_copy_only,
        detokenized_link_only=detokenized_link_only,
        *args,
        **kwargs,
    )


def process_ru(
    input_base_folder,
    split="train",
    ru_output_level="word",
    en_output_level="word",
    sentencepiece_config=None,
    prefix="",
    write_detokenized=True,
    detokenized_output_path="",
    detokenized_copy_only=False,
    detokenized_link_only=False,
    *args,
    **kwargs,
):
    return process_download(
        input_base_folder,
        split=split,
        src_lang="en",
        tgt_lang="ru",
        src_output_level=en_output_level,
        tgt_output_level=ru_output_level,
        sentencepiece_config=sentencepiece_config,
        kind="mtdata",
        write_detokenized=write_detokenized,
        detokenized_output_path=detokenized_output_path,
        detokenized_copy_only=detokenized_copy_only,
        detokenized_link_only=detokenized_link_only,
        *args,
        **kwargs,
    )


####


def add_lang_token(corpus: crp.CorpusSplit, side: str) -> crp.CorpusSplit:
    """Produces a new CorpusSplit with a magic <lang> token prepended to each sentence"""

    assert side in ["src", "tgt", "both"]

    def _add_token(
        ld: Dict[str, Optional[Dict[str, str]]], side: str
    ) -> Dict[str, Optional[Dict[str, str]]]:
        out = {**ld}

        side_dict = out[side]

        if side_dict is None:
            return out
        else:
            side_lang = side_dict["language"]
            side_text = side_dict["text"]
            out[side]["text"] = f"<{side_lang}> {side_text}"

        return out

    new_lines = (_add_token(ld, side) for ld in corpus.lines)

    return crp.CorpusSplit(
        lines=new_lines,
        split=corpus.split,
        src_lang=corpus.src_lang,
        tgt_lang=corpus.tgt_lang,
        verbose=corpus.verbose,
    )


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

        new_ld = line_dict.copy()

        new_ld[side]["text"] = to_char_level(new_ld[side]["text"])

        return new_ld

    return crp.CorpusSplit(
        lines=(lines_as_chars(ld, side=side) for ld in corpus.lines),
        split=corpus.split,
        src_lang=corpus.src_lang,
        tgt_lang=corpus.tgt_lang,
        verbose=corpus.verbose,
    )


def _apply_sp_to_side(sp, line_dict, side, other_side):
    """Apply sentencepiece processing to `side`, leaving `other_side` unchanged"""
    side_line_dict = line_dict[side]
    new_line = " ".join(sp.encode(side_line_dict["text"], out_type=str))
    sld = side_line_dict.copy()
    sld["text"] = new_line
    return {side: sld, other_side: line_dict[other_side]}


def process_with_sentencepiece(
    corpus: crp.CorpusSplit,
    side: str,
    vocab_size: int = math.inf,
    use_pretrained_model: bool = False,
    model_file: Union[str, Path] = "",
    model_base_path: Union[str, Path] = "/tmp",
    input_sentence_size: int = 0,
    shuffle_input_sentence: bool = False,
    model_type: str = "unigram",
    chunksize: int = 10000,
    n_workers: int = 2
) -> crp.CorpusSplit:
    """Process one side of a CorpusSplit with SentencePiece

    Notes:
    - With use_pretrained_model=False a new model will be trained
    and optionally saved if model_file argument was passed in
    - With use_pretrained_model=True, a pre-trained model will be loaded
    from model_path
    """
    model_file_is_correct = bool(str(model_file))

    other_side = "src" if side == "tgt" else "tgt"

    model_file = (Path(model_base_path) / Path(model_file)).expanduser()

    if not model_file.parent.exists():
        model_file.parent.mkdir(parents=True)

    if use_pretrained_model:
        assert model_file.exists(), "Must specify a valid model file."
        model_file = str(model_file)
        print(f"Applying SP model from {model_file}...")
        sp = spm.SentencePieceProcessor(model_file=model_file)

        # Don't load to RAM if don't have to
        lines = corpus.lines

    else:

        # This loading to RAM is minimal since SP training needs it
        lines_str = []
        lines = []
        line_count = 0

        for ld in corpus.lines:
            side_ld = ld[side]
            lines.append(ld)
            lines_str.append(side_ld["text"])
            line_count += 1

        # create a BytesIO object to hold the model
        model = io.BytesIO()

        spm.SentencePieceTrainer.train(
            sentence_iterator=iter(lines_str),
            model_writer=model,
            vocab_size=vocab_size,
            input_sentence_size=input_sentence_size,
            shuffle_input_sentence=shuffle_input_sentence,
            remove_extra_whitespaces=False,
            normalization_rule_name="identity",
            model_type=model_type,
            character_coverage=1.0
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

    def final_lines(sp, lines, n_workers, chunksize=1000):
        print(f"[process_with_sentencepiece] Outputting final lines...")

        for line_dict in tqdm(lines):
            side_line_dict = line_dict[side]
            new_line = " ".join(sp.encode(side_line_dict["text"], out_type=str))
            sld = side_line_dict.copy()
            sld["text"] = new_line
            yield {side: sld, other_side: line_dict[other_side]}


    output = final_lines(sp, lines, n_workers=n_workers, chunksize=chunksize)

    return crp.CorpusSplit(
        lines=output,
        src_lang=corpus.src_lang,
        tgt_lang=corpus.tgt_lang,
        split=corpus.split,
        verbose=corpus.verbose,
    )
