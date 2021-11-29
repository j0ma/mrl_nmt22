from typing import Any, Iterable, Dict, Union, DefaultDict
from pathlib import Path
from collections import defaultdict
import multiprocessing
import subprocess

import attr
import toml
import mrl_nmt.utils as u
import mrl_nmt.preprocessing.ops as ops

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


@attr.s(auto_attribs=True)
class CorpusSplit:
    src_lang: str
    tgt_lang: str
    split: str
    lines: Dict[str, Iterable[str]]
    verbose: bool

    def write_to_disk(self, folder: Path, prefix: str = "") -> None:
        """Writes self.lines to the folder specified by folder,
        inside which two files will be created, one for src and tgt.

        The file prefix can optionally be specified and if not, the
        default value of <src>-<tgt>.<split> will be used
        """
        prefix = prefix or f"{self.src_lang}-{self.tgt_lang}.{self.split}"
        for side, lines in self.lines.items():
            lang = {"src": self.src_lang, "tgt": self.tgt_lang}[side]
            output_file = folder / f"{prefix}.{lang}"
            u.write_lines(path=output_file, lines=self.lines[side])


@attr.s
class Preprocessor:
    verbose: bool = attr.ib(default=False)

    def apply_op(self, input_obj: Any, op_name: str, options: dict) -> Any:
        op_func = getattr(ops, op_name)
        if self.verbose:
            print(f"[Preprocessor.apply_op] op_name={op_name}")
            print(f"[Preprocessor.apply_op] options={options}")
        output = op_func(input_obj, **options)
        return output

    # TODO: tweak the type annotations
    def __call__(self, input_obj: Any, ops: Iterable[Dict[str, Any]]) -> Any:

        output = input_obj
        for op_dict in ops:
            op = op_dict["name"]
            try:
                options = op_dict["options"]
            except KeyError:
                options = {}
            output = self.apply_op(output, op_name=op, options=options)

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
        lines: DefaultDict[Dict[str, Any]] = defaultdict(dict)

        # Load in all the data and preprocess it
        for split in splits:
            split_config = config[split]
            input_base_path = Path(split_config["input_base_path"]).expanduser()
            for f in split_config["files"]:

                # Take note of source/target, the preprocessing steps and input
                kind, steps = f["kind"], f["preprocessing_steps"]
                input_file_path = input_base_path / f["path"]

                # Use Preprocessor class to execute the steps
                preprocessor = Preprocessor(verbose=self.verbose)
                lines[split][kind] = preprocessor(input_obj=input_file_path, ops=steps)
                # TODO: handle files like CSVs where kind = "combined"

        # Then create CorpusSplit objects out of the lines
        corpus_splits = {
            split: CorpusSplit(
                src_lang=src,
                tgt_lang=tgt,
                lines=lines[split],
                split=split,
                verbose=self.verbose,
            )
            for split in splits
        }

        # Write all the data out
        for split, corpus_split in corpus_splits.items():
            output_folder = Path(config["output_base_path"]).expanduser()
            output_folder.mkdir(parents=True, exist_ok=True)
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
