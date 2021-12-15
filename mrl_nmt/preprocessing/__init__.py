import sys
from typing import (
    Any,
    Iterable,
    Dict,
    Union,
    DefaultDict,
    Optional,
)
from pathlib import Path
from collections import defaultdict
import tempfile as tf
import multiprocessing
import subprocess

import attr
import mrl_nmt.preprocessing.ops as ops
from mrl_nmt import utils as u
from mrl_nmt.preprocessing.corpora import CorpusSplit

import sentencepiece as spm

CPU_COUNT = multiprocessing.cpu_count()


@attr.s(auto_attribs=True)
class FairseqPreprocessor:
    src_lang: str
    tgt_lang: str
    data_folder: str
    data_bin_folder: str
    n_workers: int = attr.ib(default=CPU_COUNT)
    verbose: bool = attr.ib(default=False)

    fairseq_cmd: str = "scripts/preprocess_with_fairseq.sh"

    def process(self) -> None:
        """Runs fairseq-preprocess using scripts/preprocess_with_fairseq.sh"""
        popen_args = [
            self.fairseq_cmd,
            self.src_lang,
            self.tgt_lang,
            self.data_folder,
            self.data_bin_folder,
        ]

        if self.n_workers > 0:
            popen_args.append(str(self.n_workers))

        completed_pid = subprocess.run(
            popen_args,
            capture_output=False,
            encoding="utf-8",
            text=True,
        )


@attr.s
class Preprocessor:
    verbose: bool = attr.ib(default=False)

    # TODO: tweak the type annotations
    def __call__(self, ops: Iterable[Dict[str, Any]], initial_input: Any = None) -> Any:

        output = initial_input

        for op_dict in ops:
            output = self.apply_op(
                output, op_name=op_dict["name"], options=op_dict.get("options", {})
            )
        return output

    def apply_op(self, input_obj: Any, op_name: str, options: dict) -> Any:
        op_func = getattr(ops, op_name)

        if self.verbose:
            print(f"[Preprocessor.apply_op] op_name={op_name}")
            print(f"[Preprocessor.apply_op] options={options}")
        output = op_func(input_obj, **options)

        return output


@attr.s
class ExperimentPreprocessingPipeline:
    """Class representing a preprocessing pipeline for a single experiment."""

    config: dict = attr.ib()
    verbose: bool = attr.ib(default=False)

    def __attrs_post_init__(self):
        self.lang_pairs = self.config["lang_pairs"]
        try:
            self.use_fairseq = self.config["use_fairseq"]
        except KeyError:
            self.maybe_print(
                "[ExperimentPreprocessingPipeline] Fairseq is on by default! "
                "To turn off, set 'use_fairseq=false' in the TOML config."
            )
            self.use_fairseq = True

    def maybe_print(self, msg: str) -> None:
        """Print out msg if verbose mode is on."""

        if self.verbose:
            print(msg)

    @classmethod
    def from_toml(
        cls, toml_path: Union[str, Path], verbose: bool = False
    ) -> "ExperimentPreprocessingPipeline":
        toml_reader = u.TOMLConfigReader()
        config_dict = toml_reader(toml_path)

        return cls(config=config_dict, verbose=verbose)

    def process(self):
        for lang_pair in self.lang_pairs:
            self.maybe_print(
                f"[ExperimentPreprocessingPipeline] Language pair: {lang_pair}"
            )
            self.process_lang_pair(lang_pair)

    def process_lang_pair(self, lang_pair: str) -> None:
        """Processes raw language pair data into format usable by NMT toolkit.

        This processing can include
        - Parsing various formats (XML, CSV) into one-sentence-per-line format
        - Converting between various formats (char, token, subword) per line
        """
        config = self.config[lang_pair]
        splits = config["splits"]
        src, tgt = config["src"], config["tgt"]
        lines: DefaultDict[Dict[str, CorpusSplit]] = defaultdict(dict)

        # Load in all the data and preprocess it

        preprocessor = Preprocessor(verbose=self.verbose)
        for split in splits:
            split_config = config[split]
            input_base_path = Path(split_config["input_base_path"]).expanduser()
            lines[split] = preprocessor(
                initial_input=input_base_path,
                ops=split_config["preprocessing_steps"],
            )

        # Finally write all the data out
        output_folder = Path(config["output_base_path"]).expanduser()
        output_folder.mkdir(parents=True, exist_ok=True)
        for split, corpus_split in lines.items():
            corpus_split.write_to_disk(folder=output_folder)

        # Finally binarize using fairseq if needed

        if self.use_fairseq:
            data_bin_folder = Path(config["data_bin_folder"]).expanduser()
            data_bin_folder.mkdir(parents=True, exist_ok=True)
            fairseq = FairseqPreprocessor(
                src_lang=src,
                tgt_lang=tgt,
                data_folder=str(output_folder),
                data_bin_folder=str(data_bin_folder),
            )
            fairseq.process()


@attr.s(auto_attribs=True)
class Postprocessor:

    remove_sentencepiece: bool = False
    remove_char: bool = False
    verbose: bool = True

    def __attrs_post_init__(self):
        assert not (
            self.remove_sentencepiece and self.remove_char
        ), "Can only convert one of {SP, char} to word-level"
        if self.remove_sentencepiece:
            self.postprocess = self.sp_to_words
        elif self.remove_char:
            self.postprocess = self.chars_to_words
        else:
            self.postprocess = lambda s: s

    def __call__(self, s: str) -> str:
        return self.postprocess(s)

    def sp_to_words(self, s: str) -> str:
        if self.verbose:
            print(f"Original: {s}")
        tokens = [t for t in s.split(" ")]
        out = "".join(tokens).replace(u.SP_BOW_SYMBOL, " ").lstrip()
        if self.verbose:
            print(f"Post-processed: {out}")
        return out

    def chars_to_words(self, s: str) -> str:
        if self.verbose:
            print(f"Original: {s}")
        out = s.replace(" ", "").replace(u.SPACE_SYMBOL, " ")
        if self.verbose:
            print(f"Post-processed: {out}")
        return out


@attr.s(auto_attribs=True)
class MosesCleanCorpusNProcessor:

    ratio: int

    min_len: int
    max_len: int

    moses_scripts_path: Union[Path, str] = Path("moses/scripts").expanduser()

    def __attrs_post_init__(self) -> None:

        if not self.moses_scripts_path.exists():
            raise ValueError(
                "Must provide a existing path to Moses scripts! "
                f"Got: {str(self.moses_scripts_path.expanduser())}"
            )

        self.cmd = str(self.moses_scripts_path / "training/clean-corpus-n.perl")

    def __call__(
        self,
        src_suffix: str,
        tgt_suffix: str,
        input_prefix: Union[str, Path] = "",
        output_prefix: Union[str, Path] = "",
        src_input_file: Union[str, Path] = "",
        tgt_input_file: Union[str, Path] = "",
    ):

        with tf.TemporaryDirectory() as temp_dir:

            mode = self._infer_mode(
                input_prefix, src_suffix, tgt_suffix, src_input_file, tgt_input_file
            )

            if mode == "file":
                input_prefix = self._file_to_input_prefix(
                    src_suffix,
                    tgt_suffix,
                    src_input_file,
                    tgt_input_file,
                    temp_folder=Path(temp_dir),
                )
            else:
                input_prefix = Path(input_prefix).expanduser()

            output_prefix = Path(output_prefix).expanduser()

            # Infer output folder and mkdir if necessary
            output_folder = Path(output_prefix).parent
            if not output_folder.exists():
                output_folder.mkdir(parents=True, exist_ok=True)

            popen_args = [
                "perl",
                self.cmd,
                f"-ratio={self.ratio}",
                str(input_prefix),
                str(src_suffix),
                str(tgt_suffix),
                str(output_prefix),
                str(self.min_len),
                str(self.max_len),
            ]

            completed_pid = subprocess.run(
                popen_args,
                capture_output=True,
                encoding="utf-8",
                text=True,
            )

            print(completed_pid.stderr, file=sys.stderr)

    @staticmethod
    def _infer_mode(
        input_prefix: Union[str, Path],
        src_suffix: Union[str, Path],
        tgt_suffix: Union[str, Path],
        src_input_file: Union[str, Path],
        tgt_input_file: Union[str, Path],
    ):
        _f_src = Path(src_input_file)
        _f_tgt = Path(tgt_input_file)
        if input_prefix:
            _src = Path(f"{input_prefix}.{src_suffix}")
            _tgt = Path(f"{input_prefix}.{tgt_suffix}")
            assert (
                _src.exists or _f_src.exists
            ), "Must provide an input prefix or src input file"
            assert (
                _tgt.exists or _f_tgt.exists
            ), "Must provide an input prefix or tgt input file"

            mode = "prefix"
        else:
            mode = "file"

        if mode == "prefix":
            for side, suf in zip(["src", "tgt"], [src_suffix, tgt_suffix]):
                f = Path(f"{input_prefix}.{suf}")
                assert f.exists(), f"Nonexistent {side} file: {f}"

        return mode

    @staticmethod
    def _file_to_input_prefix(
        src_suffix: str,
        tgt_suffix: str,
        src_input_file: Union[str, Path],
        tgt_input_file: Union[str, Path],
        temp_folder: Union[str, Path],
    ):

        input_prefix = Path(temp_folder) / "corpus"

        # symlink the files we need
        src_input = f"{input_prefix}.{src_suffix}"
        tgt_input = f"{input_prefix}.{tgt_suffix}"
        u.create_symlink(link_path=src_input, dest_path=src_input_file)
        u.create_symlink(link_path=tgt_input, dest_path=tgt_input_file)

        return input_prefix
