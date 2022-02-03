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
import multiprocessing as mp
import subprocess
import time

import toml
import attr
from rich import print
import mrl_nmt.preprocessing.ops as ops
from mrl_nmt import utils as u
from mrl_nmt.preprocessing.corpora import CorpusSplit, LoadedTextFile

from sacremoses import MosesDetokenizer
import sentencepiece as spm

CPU_COUNT = mp.cpu_count()

# keep list of types of supported subword processing
PROCESSING_TYPES = ops.OUTPUT_LEVELS


@attr.s(auto_attribs=True)
class FairseqPreprocessor:
    src_lang: str
    tgt_lang: str
    data_folder: str
    data_bin_folder: str
    src_dict: str = ""
    tgt_dict: str = ""
    train_prefix: str = ""
    dev_prefix: str = ""
    test_prefix: str = ""
    splits: Iterable[str] = ("train", "valid")
    joined_dictionary: bool = False

    gpu_devices: str = ""
    use_gpu: bool = False

    n_workers: int = attr.ib(default=CPU_COUNT)
    verbose: bool = attr.ib(default=False)

    fairseq_cmd: str = "fairseq-preprocess"

    def __attrs_post_init__(self):

        self.train_prefix = f"{self.data_folder}/{self.src_lang}-{self.tgt_lang}.train"
        self.dev_prefix = f"{self.data_folder}/{self.src_lang}-{self.tgt_lang}.dev"
        self.test_prefix = f"{self.data_folder}/{self.src_lang}-{self.tgt_lang}.test"
        self.prefixes = {
            "train": self.train_prefix,
            "dev": self.dev_prefix,
            "test": self.test_prefix,
            "valid": self.dev_prefix,
        }
        if self.use_gpu:
            assert (
                self.gpu_devices
            ), f"Unset GPUs! Got self.gpu_devices={self.gpu_devices}"

    def process(self) -> None:

        popen_args = [
            f"CUDA_VISIBLE_DEVICES={self.gpu_devices} && {self.fairseq_cmd}"
            if self.use_gpu
            else self.fairseq_cmd,
            f"--source-lang {self.src_lang}",
            f"--target-lang {self.tgt_lang}",
            f"--destdir {self.data_bin_folder}",
        ]

        for side, dict_path in zip(("src", "tgt"), (self.src_dict, self.tgt_dict)):
            if dict_path:
                popen_args.append(f"--{side}dict {dict_path}")

        popen_args.extend(
            [
                f"--{spl if spl != 'dev' else 'valid'}pref {self.prefixes[spl]}"
                for spl in self.splits
            ]
        )

        if not self.use_gpu:
            popen_args.extend(["--cpu"])

        popen_args.append(f"--workers {self.n_workers}")

        if self.joined_dictionary:
            popen_args.append("--joined-dictionary")

        popen_args = " ".join([a for a in popen_args if len(a)])

        subprocess.run(
            args=popen_args,
            capture_output=False,
            encoding="utf-8",
            text=True,
            shell=True,
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
    use_gpu: bool = attr.ib(default=False)
    gpu_devices: str = attr.ib(default="")
    n_workers: int = attr.ib(default=1)
    joined_dictionary: bool = attr.ib(default=False)

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
        cls,
        toml_path: Union[str, Path],
        verbose: bool = False,
        use_gpu: bool = False,
        gpu_devices: str = "",
        n_workers: int = 1,
        joined_dictionary: bool = False,
    ) -> "ExperimentPreprocessingPipeline":
        config_dict = toml.load(toml_path)

        return cls(
            config=config_dict,
            verbose=verbose,
            use_gpu=use_gpu,
            gpu_devices=gpu_devices,
            n_workers=n_workers,
            joined_dictionary=joined_dictionary,
        )

    @classmethod
    def from_yaml(
        cls,
        yaml_path: Union[str, Path],
        verbose: bool = False,
        use_gpu: bool = False,
        gpu_devices: str = "",
        n_workers: int = 1,
        joined_dictionary: bool = False,
    ) -> "ExperimentPreprocessingPipeline":
        config_dict = u.read_yaml(yaml_path)

        return cls(
            config=config_dict,
            verbose=verbose,
            use_gpu=use_gpu,
            gpu_devices=gpu_devices,
            n_workers=n_workers,
            joined_dictionary=joined_dictionary,
        )

    def process(self):

        # process first language pair
        _lang_pairs = [l for l in self.lang_pairs]
        first_lang = _lang_pairs.pop(0)
        print(f"Processing first lang: {first_lang}")
        self.process_lang_pair(first_lang)

        time.sleep(5)

        # if there are others left, process them
        # (this is necessary in case we re-use a fairseq dict from the first lang pair)
        if _lang_pairs:
            print(f"Processing rest of language pairs: {_lang_pairs}")
            if self.n_workers == 1:
                for lang_pair in _lang_pairs:
                    self.maybe_print(
                        f"[ExperimentPreprocessingPipeline] Language pair: {lang_pair}"
                    )
                    self.process_lang_pair(lang_pair)
            else:
                with mp.Pool(self.n_workers) as pool:
                    print(
                        f"[ExperimentPreprocessingPipeline] Processing language pairs {self.lang_pairs} using {self.n_workers} workers"
                    )
                    pool.map(self.process_lang_pair, self.lang_pairs)

    def process_lang_pair(self, lang_pair: str) -> None:
        """Processes raw language pair data into format usable by an NMT toolkit.

        This processing can include
        - Parsing various formats (XML, CSV) into one-sentence-per-line format
        - Converting between various formats (char, token, subword) per line
        """
        config = self.config[lang_pair]
        corpus_names = config["corpora"]
        src, tgt = config["src"], config["tgt"]
        lines: DefaultDict[Dict[str, CorpusSplit]] = defaultdict(dict)

        # Load in all the data and text_processing it

        preprocessor = Preprocessor(verbose=self.verbose)

        for corpus_name in corpus_names:
            corpus_config = config[corpus_name]
            splits = corpus_config["splits"]

            for split in splits:
                split_config = corpus_config[split]
                input_base_path = Path(split_config["input_base_path"]).expanduser()
                lines[corpus_name][split] = preprocessor(
                    initial_input=input_base_path,
                    ops=split_config["preprocessing_steps"],
                )

        # Finally write all the data out

        for corpus_name, split_dict in lines.items():
            corpus_config = config[corpus_name]
            output_folder = Path(corpus_config["output_base_path"]).expanduser()

            for split, corpus_split in split_dict.items():

                output_folder.mkdir(parents=True, exist_ok=True)
                corpus_split.write_to_disk(folder=output_folder)

            # Finally binarize using fairseq if needed

            if self.use_fairseq:
                src_dict = str(
                    Path(corpus_config["fairseq_src_dict"]).expanduser()
                    if "fairseq_src_dict" in corpus_config
                    else ""
                )
                tgt_dict = str(
                    Path(corpus_config["fairseq_tgt_dict"]).expanduser()
                    if "fairseq_tgt_dict" in corpus_config
                    else ""
                )
                data_bin_folder = Path(corpus_config["data_bin_folder"]).expanduser()
                data_bin_folder.mkdir(parents=True, exist_ok=True)
                fairseq = FairseqPreprocessor(
                    src_lang=src,
                    tgt_lang=tgt,
                    data_folder=str(output_folder),
                    data_bin_folder=str(data_bin_folder),
                    src_dict=src_dict,
                    tgt_dict=tgt_dict,
                    splits=corpus_config["splits"],
                    use_gpu=self.use_gpu,
                    gpu_devices=self.gpu_devices,
                    joined_dictionary=self.joined_dictionary,
                )
                fairseq.process()


@attr.s(auto_attribs=True)
class Postprocessor:

    remove_sentencepiece: bool = False
    remove_char: bool = False
    detokenize: bool = False

    verbose: bool = True

    def __attrs_post_init__(self):
        assert not (
            self.remove_sentencepiece and self.remove_char
        ), "Can only convert one of {SP, char} to word-level"

        if self.remove_sentencepiece:
            self._postprocess = self.sp_to_words
        elif self.remove_char:
            self._postprocess = self.chars_to_words
        else:
            self._postprocess = lambda s: s

        # TODO: languages for which Moses isn't suitable?
        self.moses = MosesDetokenizer()

    def postprocess(self, s: str) -> str:
        out = self._postprocess(s)

        if self.detokenize:
            out = self.moses.detokenize(out.split(" "))

        return out

    def __call__(self, s: str) -> str:
        return self.postprocess(s)

    def sp_to_words(self, s: str) -> str:
        if self.verbose:
            print(f"Original: {s}")

        if not s:
            out = ""
        else:
            tokens = [t for t in s.split(" ")]
            out = "".join(tokens).replace(u.SP_BOW_SYMBOL, " ").lstrip()

        if self.verbose:
            print(f"Post-processed: {out}")

        return out

    def chars_to_words(self, s: str) -> str:
        if self.verbose:
            print(f"Original: {s}")

        if not s:
            out = ""
        else:
            out = s.replace(" ", "").replace(u.SPACE_SYMBOL, " ")

        if self.verbose:
            print(f"Post-processed: {out}")

        return out

    @classmethod
    def from_strings(
        cls, processing_to_remove: str, detokenize: bool, verbose: bool = False
    ):
        return cls(
            remove_sentencepiece=processing_to_remove == "sentencepiece",
            remove_char=processing_to_remove in ["characters", "char", "chars"],
            detokenize=detokenize,
            verbose=verbose,
        )


@attr.s(auto_attribs=True)
class MosesCleanCorpusNProcessor:

    ratio: int

    min_len: int
    max_len: int

    moses_scripts_path: Union[Path, str] = Path("moses/scripts").expanduser()

    def __attrs_post_init__(self) -> None:

        self.moses_scripts_path = Path(self.moses_scripts_path)

        if not self.moses_scripts_path.exists():
            raise ValueError(
                "Must provide a existing path to Moses scripts! "
                f"Got: {str(self.moses_scripts_path.expanduser())}"
            )

        self.cmd = str(self.moses_scripts_path / "training/clean-corpus-n.perl")

    def process_corpus_split(
        self, cs: CorpusSplit, load_to_memory: bool = False
    ) -> CorpusSplit:
        """Filter lines in a CorpusSplit object.

        Note: this method handles the general case where the Moses script is applied
        to a CorpusSplit at an arbitrary point in a processing pipeline. Instead of
        loading a CorpusSplit from disk and calling this right after, consider simply
        processing the files first and then creating a CorpusSplit object.
        """

        split = cs.split
        src_lang = cs.src_lang
        tgt_lang = cs.tgt_lang
        verbose = cs.verbose

        with tf.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            tmp_prefix_in = "corpus"
            tmp_prefix_out = "corpus_clean"

            # write lines to temporary directory
            cs.write_to_disk(folder=temp_dir_path, prefix=tmp_prefix_in)

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
                load_to_memory=load_to_memory,
            )
            tgt_clean = LoadedTextFile(
                path=tgt_output_file,
                side="tgt",
                language=tgt_lang,
                load_to_memory=load_to_memory,
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
