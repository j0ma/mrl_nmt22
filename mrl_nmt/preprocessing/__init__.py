from typing import (
    Any,
    Iterable,
    Dict,
    Union,
    DefaultDict,
    Set,
    Callable,
    Sequence,
    Optional,
    Mapping,
)
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
class LoadedFile:
    path: Union[str, Path]
    side: str  # src, tgt, both

    # language or "multi" if multilingual
    language: str

    # use this to toggle streaming/loading to ram
    load_to_memory: bool = True

    # use this to toggle loading of intermediate lines to RAM
    load_intermediate: bool = False

    sides: Set[str] = set(["src", "tgt", "both"])

    parse_language_func: Optional[Callable[[Any], str]] = None

    def parse_language(self, s: str) -> str:
        if self.parse_language_func is None:
            return self.language

        return self.parse_language_func(s)

    def __attrs_post_init__(self) -> None:
        assert self.side in self.sides, f"'kind' must be one of {self.sides}"
        self.both_sides = self.side == "both"
        self.path = Path(self.path).expanduser()

    @property
    def stream_lines(self) -> Iterable[str]:
        raise NotImplementedError

    @property
    def lines(self) -> Iterable[str]:
        raise NotImplementedError

    @property
    def lines_as_dicts(self) -> Iterable[Dict[str, Optional[Dict[str, Optional[str]]]]]:
        raise NotImplementedError


@attr.s(auto_attribs=True)
class LoadedTextFile(LoadedFile):

    parse_src_tgt_func: Optional[Callable[[Any], Dict[str, str]]] = None

    def line_to_dict(self, line: str) -> Dict[str, Optional[Dict[str, Optional[str]]]]:
        """Parses a raw line to dictionary form depending on side"""

        if self.both_sides:
            raise NotImplementedError("Text files can only handle one side per file!")
        else:
            other_side = {"src": "tgt", "tgt": "src"}[self.side]

            return {
                self.side: {"text": line, "language": self.parse_language(line)},
                other_side: None,
            }

    @property
    def stream_lines(self) -> Iterable[str]:
        return u.stream_lines(self.path)

    @property
    def lines(self) -> Iterable[str]:
        if not (self.load_to_memory or self.load_intermediate):
            raise ValueError(
                "Cannot load lines to RAM without either load_to_memory=True "
                "or load_intermediate=True. Use stream_lines instead."
            )

        return u.read_lines(self.path)

    @property
    def lines_as_dicts(self) -> Iterable[Dict[str, Optional[Dict[str, Optional[str]]]]]:

        line_iterator = self.lines if self.load_intermediate else self.stream_lines

        lines = (self.line_to_dict(line) for line in line_iterator)

        return list(lines) if self.load_to_memory else lines


@attr.s(auto_attribs=True)
class LoadedTSVFile(LoadedFile):

    language: str = "multi"
    src_language: Optional[str] = None
    tgt_language: Optional[str] = None
    src_column: Optional[Union[str, int]] = None
    tgt_column: Optional[Union[str, int]] = None
    fieldnames: Optional[Sequence[str]] = []
    delimiter: str = "\t"

    def __attrs_post_init__(self):
        self.use_dict_reader = bool(self.fieldnames)
        self.validate_columns()

    def validate_columns(self) -> None:
        """If we are using csv.DictReader, need string keys.

        Otherwise, need integer keys for csv.reader.
        """
        ftype = str if self.use_dict_reader else int
        assert isinstance(
            self.src_column, ftype
        ), f"src_column must be {ftype} when fieldnames={self.fieldnames}"
        assert isinstance(
            self.tgt_column, ftype
        ), f"tgt_column must be {ftype} when fieldnames={self.fieldnames}"

    def line_to_dict(
        self, line: Mapping[Union[str, int], str]
    ) -> Dict[str, Optional[Dict[str, Optional[str]]]]:
        """Parses a raw line to dictionary form depending on side"""

        out: Dict[str, Optional[Dict[str, Optional[str]]]] = {}

        if self.src_column is not None:
            src_line = line[self.src_column]
            out["src"] = {
                "text": src_line,
                "language": self.src_language or self.parse_language(src_line),
            }
        else:
            out["src"] = None

        if self.tgt_column is not None:
            tgt_line = line[self.tgt_column]
            out["tgt"] = {
                "text": tgt_line,
                "language": self.tgt_language or self.parse_language(tgt_line),
            }
        else:
            out["tgt"] = None

        return out

    @property
    def lines_as_dicts(self) -> Iterable[Dict[str, Optional[Dict[str, Optional[str]]]]]:

        if self.use_dict_reader:
            line_iterator = u.read_tsv_dict(
                self.path,
                field_names=self.fieldnames,
                delimiter=self.delimiter,
                load_to_memory=self.load_intermediate,
            )
        else:
            line_iterator = u.read_tsv_list(
                self.path,
                delimiter=self.delimiter,
                load_to_memory=self.load_intermediate,
            )

        if self.load_to_memory:
            return [self.line_to_dict(line) for line in line_iterator]
        else:
            for line in line_iterator:
                yield self.line_to_dict(line)


@attr.s(auto_attribs=True)
class CorpusSplit:
    split: str
    lines: Iterable[Dict[str, Optional[Dict[str, Optional[str]]]]]
    src_lang: Optional[str] = None
    tgt_lang: Optional[str] = None
    verbose: bool = True

    def __attrs_post_init__(self) -> None:
        assert self.src_lang or self.tgt_lang, "src_lang or tgt_lang must be specified."

    def write_to_disk(self, folder: Path, prefix: str = "") -> None:
        """Writes self.lines to the folder specified by folder,
        inside which two files will be created, one for src and tgt.

        The file prefix can optionally be specified and if not, the
        default value of <src>-<tgt>.<split> will be used
        """
        prefix = prefix or f"{self.src_lang}-{self.tgt_lang}.{self.split}"

        for side, lines in self.lines.items():
            if lines:
                lang = {"src": self.src_lang, "tgt": self.tgt_lang}[side]
                output_file = folder / f"{prefix}.{lang}"
                lines = (d[side]["text"] for d in lines)
                u.write_lines(path=output_file, lines=self.lines[side])

    @classmethod
    def from_src_tgt(
        cls, src: LoadedFile, tgt: LoadedFile, split: str, verbose: bool = True
    ) -> "CorpusSplit":

        joined_lines = (
            {"src": src_line["src"], "tgt": tgt_line["tgt"]}

            for src_line, tgt_line in zip(src.lines_as_dicts, tgt.lines_as_dicts)
        )

        return cls(
            src_lang=src.language,
            tgt_lang=tgt.language,
            split=split,
            verbose=verbose,
            lines=joined_lines,
        )

    @classmethod
    def from_tsv(
        cls, tsv: LoadedTSVFile, split: str, verbose: bool = True
    ) -> "CorpusSplit":
        return cls(
            src_lang=tsv.src_language,
            tgt_lang=tsv.tgt_language,
            split=split,
            verbose=verbose,
            lines=tsv.lines_as_dicts,
        )


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

        # Finally write all the data out
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
