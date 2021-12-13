from typing import (
    Any,
    Iterable,
    Dict,
    Union,
    DefaultDict,
)
from pathlib import Path
from collections import defaultdict
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
        tokens = [t or u.SP_BOW_SYMBOL for t in s.split(" ")]
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
