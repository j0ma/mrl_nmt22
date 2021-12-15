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
from mrl_nmt.preprocessing.corpora import CorpusSplit, LoadedTextFile

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

    def process_corpus_split(self, cs: CorpusSplit) -> CorpusSplit:
        """Filter lines in a CorpusSplit object.

        NOTE: intended to be used before any subword tokenization
        i.e. cs.detok_lines should not exist.
        """
        _msg = f"Corpus line filtering only supported with unmodified corpora whose .detok_lines = None!"
        assert not cs.detok_lines, _msg

        split = cs.split
        src_lang = cs.src_lang
        tgt_lang = cs.tgt_lang
        verbose = cs.verbose

        with tf.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            tmp_prefix_in = "corpus"
            tmp_prefix_out = "corpus_clean"

            # write lines to temporary directory
            cs.write_to_disk(
                folder=temp_dir_path, prefix=tmp_prefix_in, write_detok_lines=False
            )

            src_input_file = temp_dir_path / f"{tmp_prefix_in}.{src_lang}"
            tgt_input_file = temp_dir_path / f"{tmp_prefix_in}.{tgt_lang}"

            src_output_file = temp_dir_path / f"{tmp_prefix_out}.{src_lang}"
            tgt_output_file = temp_dir_path / f"{tmp_prefix_out}.{tgt_lang}"

            self.process_files(
                src_input_file=src_input_file,
                tgt_input_file=tgt_input_file,
                src_output_file=src_output_file,
                tgt_output_file=tgt_output_file,
            )

            src_clean = LoadedTextFile(
                path=src_output_file,
                side="src",
                language=src_lang,
                load_to_memory=True,
            )
            tgt_clean = LoadedTextFile(
                path=tgt_output_file,
                side="tgt",
                language=tgt_lang,
                load_to_memory=True,
            )
            new_cs = CorpusSplit.from_src_tgt(
                src=src_clean, tgt=tgt_clean, split=split, verbose=verbose
            )
            return new_cs

    def process_files(
        self,
        src_input_file: Union[str, Path],
        tgt_input_file: Union[str, Path],
        src_output_file: Union[str, Path],
        tgt_output_file: Union[str, Path],
    ):
        for side, f in zip(["src", "tgt"], [src_input_file, tgt_input_file]):
            assert Path(f).exists(), f"Nonexistent {side} input file: {f}"

        src_suff, tgt_suff = "src", "tgt"
        with tf.TemporaryDirectory() as staging_path:

            staging = Path(staging_path)

            # Symlink input files to temp dir
            tmp_input_prefix = staging / "corpus"
            tmp_src_input = Path(f"{tmp_input_prefix}.{src_suff}")
            tmp_tgt_input = Path(f"{tmp_input_prefix}.{tgt_suff}")

            u.create_symlink(link_path=tmp_src_input, dest_path=src_input_file)
            u.create_symlink(link_path=tmp_tgt_input, dest_path=tgt_input_file)

            tmp_output_prefix = staging / "corpus_clean"
            tmp_src_output = Path(f"{tmp_output_prefix}.{src_suff}")
            tmp_tgt_output = Path(f"{tmp_output_prefix}.{tgt_suff}")

            popen_args = [
                "perl",
                self.cmd,
                f"-ratio={self.ratio}",
                str(tmp_input_prefix),
                src_suff,
                tgt_suff,
                str(tmp_output_prefix),
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

            # Move files from temp folder to output
            u.move(tmp_src_output, src_output_file)
            u.move(tmp_tgt_output, tgt_output_file)
