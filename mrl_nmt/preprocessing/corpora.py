from pathlib import Path
from typing import (
    Union,
    Set,
    Optional,
    Callable,
    Any,
    Iterable,
    Dict,
    Tuple,
    Sequence,
    Mapping,
)
import itertools as it
import sys

import attr
from lxml import etree as etree
from tqdm import tqdm

from mrl_nmt import utils as u
from mrl_nmt.utils import get_line_from_dict
from mrl_nmt.preprocessing.tmx import TMXHandler

from sacremoses import MosesDetokenizer


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

    sides: Set[str] = {"src", "tgt", "both"}

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
    fieldnames: Optional[Sequence[str]] = None
    delimiter: str = "\t"
    use_custom_reader: bool = False
    expected_n_fields: int = 0
    validate_lines: bool = True
    faulty_lines_path: str = ""

    def __attrs_post_init__(self):
        self.fieldnames = self.fieldnames or []
        self.both_sides = self.side == "both"
        self.other_side_map = {"src": "tgt", "tgt": "src"}
        self.use_dict_reader = bool(self.fieldnames)
        self.validate_columns()
        if not self.expected_n_fields:
            # default: two fields (src/tgt) or no. of field names given
            self.expected_n_fields = max(2, len(self.fieldnames))

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

    def validate_line(
        self, line: Mapping[Union[str, int], str], line_ix: Optional[int] = None
    ) -> bool:
        """Validates a line by performing assertion-based checks.

        If all assertions evaluate to True, the line is deemed valid.
        Otherwise, False is returned.
        """
        try:
            n = len(line)
            m = self.expected_n_fields
            assert n == m, f"Expected {m} fields, got {n}"
        except AssertionError as err:
            prefix = f"[LoadedTSVFile] Error @ line {line_ix}: " if line_ix else ""

            if prefix:
                print(f"{prefix} {err}", file=sys.stderr)
            return False

        return True

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
                use_custom_reader=self.use_custom_reader,
            )
        else:
            line_iterator = u.read_tsv_list(
                self.path,
                delimiter=self.delimiter,
                load_to_memory=self.load_intermediate,
            )

        n_valid_lines, n_lines = 0, 0
        print(f"[LoadedTSVFile] Loading lines from {self.path}", file=sys.stderr)
        for ix, line in enumerate(line_iterator):
            n_lines += 1
            cond = self.validate_line(line, line_ix=ix) if self.validate_line else True
            if cond:
                n_valid_lines += 1
                yield self.line_to_dict(line)

        print(
            f"[LoadedTSVFile] Loaded {n_valid_lines} valid lines / {n_lines} total lines",
            file=sys.stderr,
        )


@attr.s(auto_attribs=True)
class LoadedXLIFFFile(LoadedTSVFile):

    language: str = "multi"
    side: str = "both"

    src_language: Optional[str] = None
    tgt_language: Optional[str] = None

    load_to_memory: bool = False

    def __attrs_post_init__(self):
        self.both_sides = self.side == "both"
        parsed_xml = self.parse_xml()
        self.src_lines = parsed_xml["src"]["lines"]
        self.tgt_lines = parsed_xml["tgt"]["lines"]

    def parse_xml(self) -> Dict[str, Dict[str, Union[str, Iterable[str]]]]:

        # load xml into memory
        str_path = str(self.path)
        tree = etree.parse(str_path).getroot()

        src_lang = tree.attrib["srcLang"]
        tgt_lang = tree.attrib["trgLang"]

        # find all source and targetlines
        src_lines = [
            element.text for element in tree.xpath("//*[local-name() = 'source']")
        ]
        tgt_lines = [
            element.text for element in tree.xpath("//*[local-name() = 'target']")
        ]

        return {
            "src": {"language": src_lang, "lines": src_lines},
            "tgt": {"language": tgt_lang, "lines": tgt_lines},
        }

    def line_to_dict(
        self, line: Tuple[str, str]
    ) -> Dict[str, Optional[Dict[str, Optional[str]]]]:
        """Parses a raw line to dictionary form depending on side"""

        out: Dict[str, Optional[Dict[str, Optional[str]]]] = {}
        src_line, tgt_line = line

        out["src"] = {
            "text": src_line,
            "language": self.src_language or self.parse_language(src_line),
        }

        out["tgt"] = {
            "text": tgt_line,
            "language": self.tgt_language or self.parse_language(tgt_line),
        }

        return out

    @property
    def lines_as_dicts(self) -> Iterable[Dict[str, Optional[Dict[str, Optional[str]]]]]:

        line_iterator = zip(self.src_lines, self.tgt_lines)

        lines = (self.line_to_dict(line) for line in line_iterator)

        return list(lines) if self.load_to_memory else lines


@attr.s(auto_attribs=True)
class LoadedTMXFile(LoadedTSVFile):

    language: str = "multi"
    src_language: Optional[str] = None
    tgt_language: Optional[str] = None
    delimiter: str = "\t"

    # TMX files always have both sides
    side: str = "both"

    def __attrs_post_init__(self):
        self.src_column = self.src_language
        self.tgt_column = self.tgt_language
        self.fieldnames = [self.src_column, self.tgt_column]

    def _line_to_dict_lazy(self, line_iterator):
        for line in line_iterator:
            yield self.line_to_dict(line)

    @property
    def lines_as_dicts(self) -> Iterable[Dict[str, Optional[Dict[str, Optional[str]]]]]:

        line_iterator = TMXHandler(
            input_file=self.path,
            src_language=self.src_language,
            tgt_language=self.tgt_language,
            legacy_mode=False,
        ).process()

        if self.load_to_memory:
            return [self.line_to_dict(line) for line in line_iterator]
        else:
            return self._line_to_dict_lazy(line_iterator)


@attr.s(auto_attribs=True)
class CorpusSplit:
    split: str
    lines: Iterable[Dict[str, Optional[Dict[str, Optional[str]]]]]
    src_lang: Optional[str] = None
    tgt_lang: Optional[str] = None
    verbose: bool = True
    detok_lines: Optional[
        Iterable[Dict[str, Optional[Dict[str, Optional[str]]]]]
    ] = None
    use_moses_detokenizer: bool = True

    def __attrs_post_init__(self) -> None:
        assert self.src_lang or self.tgt_lang, "src_lang or tgt_lang must be specified."
        if self.detok_lines is None:
            self.detok_lines = []

        # TODO: language specific rules?
        self.moses = MosesDetokenizer()

    @property
    def detokenized_lines(self) -> Iterable[str]:
        if not self.detok_lines:
            return []

        def _detok_line(ld: Dict[str, Dict[str, str]]):
            new_d = {**ld}
            src_line, tgt_line = u.get_line_from_dict(ld)
            new_d["src"]["text"] = self.moses.detokenize(src_line.split(" "))
            new_d["tgt"]["text"] = self.moses.detokenize(tgt_line.split(" "))
            return new_d

        for line in self.detok_lines:
            if self.use_moses_detokenizer:
                detok = _detok_line(line)
                yield detok
            else:
                yield line

    def write_to_disk(
        self,
        folder: Path,
        prefix: str = "",
        detok_prefix: str = "",
        skip_upon_fail: bool = True,
        write_detok_lines: bool = False,
    ) -> None:
        """Writes self.lines to the folder specified by folder,
        inside which two files will be created, one for src and tgt.

        The file prefix can optionally be specified and if not, the
        default value of <src>-<tgt>.<split> will be used

        By default, skip_upon_fail=True which causes lines where either side is empty to be skipped.
        """
        prefix = prefix or f"{self.src_lang}-{self.tgt_lang}.{self.split}"
        src_out_path = folder / f"{prefix}.{self.src_lang}"
        tgt_out_path = folder / f"{prefix}.{self.tgt_lang}"

        detok_prefix = detok_prefix or f"{prefix}.detok"
        src_detok_out_path = folder / f"{detok_prefix}.{self.src_lang}"
        tgt_detok_out_path = folder / f"{detok_prefix}.{self.tgt_lang}"

        src_lines_written = 0
        tgt_lines_written = 0
        src_detok_lines_written = 0
        tgt_detok_lines_written = 0
        skipped_lines = 0
        with open(src_out_path, "w", encoding="utf-8") as src_out, open(
            tgt_out_path, "w", encoding="utf-8"
        ) as tgt_out, open(
            src_detok_out_path, "w", encoding="utf-8"
        ) as src_detok_out, open(
            tgt_detok_out_path, "w", encoding="utf-8"
        ) as tgt_detok_out:

            line_iter = enumerate(
                it.zip_longest(self.detokenized_lines, self.lines), start=1
            )
            for ix, (detok_line, line) in tqdm(line_iter):
                src_line, tgt_line = get_line_from_dict(line)
                try:
                    assert (
                        len(src_line) > 0 and len(tgt_line) > 0
                    ), f"Null source and/or target line! Got: src={src_line}, tgt={tgt_line}"

                    src_out.write(f"{src_line}\n")
                    src_lines_written += 1
                    tgt_out.write(f"{tgt_line}\n")
                    tgt_lines_written += 1

                    if detok_line:
                        src_detok_line, tgt_detok_line = get_line_from_dict(detok_line)
                        assert (
                            len(src_detok_line) > 0 and len(tgt_detok_line) > 0
                        ), f"Null detokenized source and/or target line! Got: src={src_detok_line}, tgt={tgt_detok_line}"
                        src_detok_out.write(f"{src_detok_line}\n")
                        src_detok_lines_written += 1
                        tgt_detok_out.write(f"{tgt_detok_line}\n")
                        tgt_detok_lines_written += 1

                except:
                    if skip_upon_fail:
                        skipped_lines += 1
                        print(
                            f"WARNING: Skipping since both sides not complete. Line: {line}",
                            file=sys.stderr,
                        )
                        continue
                    else:
                        raise ValueError("Failing since skip_upon_fail=False.")

        assert (
            src_lines_written == tgt_lines_written
        ), f"Different number of src/tgt lines written: {src_lines_written} (src) vs. {tgt_lines_written} (tgt)"

        if self.detok_lines is not None:
            assert (
                src_detok_lines_written == tgt_detok_lines_written
            ), f"Different number of detok src/tgt lines written: {src_detok_lines_written} (src) vs. {tgt_detok_lines_written} (tgt)"

    @classmethod
    def from_src_tgt(
        cls,
        src: Union[LoadedFile, "CorpusSplit"],
        tgt: Union[LoadedFile, "CorpusSplit"],
        split: str,
        verbose: bool = True,
    ) -> "CorpusSplit":
        assert (
            len({type(src), type(tgt)}) == 1
        ), f"Both src and tgt must have the same type! Got: {type(src)} (src) {type(tgt)} (tgt)"
        if isinstance(src, LoadedFile):
            return cls._from_src_tgt_file(src, tgt, split=split, verbose=verbose)
        elif isinstance(src, CorpusSplit):
            return cls._from_src_tgt_corpus_split(
                src, tgt, split=split, verbose=verbose
            )
        else:
            raise ValueError(
                f"Only CorpusSplit and LoadedFile objects supported. Got: {type(src)}"
            )

    @classmethod
    def _from_src_tgt_corpus_split(
        cls, src: "CorpusSplit", tgt: "CorpusSplit", split: str, verbose: bool = True
    ) -> "CorpusSplit":

        joined_lines = (
            {"src": s["src"], "tgt": t["tgt"]} for s, t in zip(src.lines, tgt.lines)
        )

        return cls(
            src_lang=src.src_lang,
            tgt_lang=tgt.tgt_lang,
            split=split,
            verbose=verbose,
            lines=joined_lines,
        )

    @classmethod
    def _from_src_tgt_file(
        cls, src: LoadedFile, tgt: LoadedFile, split: str, verbose: bool = True
    ) -> "CorpusSplit":

        assert src.side == "src", f"The side of {src} must be 'src'."
        assert tgt.side == "tgt", f"The side of {tgt} must be 'tgt'."

        joined_lines = (
            {"src": s["src"], "tgt": t["tgt"]}
            for s, t in zip(src.lines_as_dicts, tgt.lines_as_dicts)
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
        assert (
            tsv.both_sides
        ), "Only TSVs with both source and target text are supported."

        return cls(
            src_lang=tsv.src_language,
            tgt_lang=tsv.tgt_language,
            split=split,
            verbose=verbose,
            lines=tsv.lines_as_dicts,
        )

    @classmethod
    def from_xliff(
        cls, xliff: LoadedXLIFFFile, split: str, verbose: bool = True
    ) -> "CorpusSplit":
        assert (
            xliff.both_sides
        ), "Only XLF files with both source and target text are supported."

        return cls(
            src_lang=xliff.src_language,
            tgt_lang=xliff.tgt_language,
            split=split,
            verbose=verbose,
            lines=xliff.lines_as_dicts,
        )

    @classmethod
    def from_text_file(
        cls,
        text_file: LoadedTextFile,
        split: str,
    ) -> "CorpusSplit":
        other_side = {"src": "tgt", "tgt": "src"}[text_file.side]
        lang_kwarg = {f"{text_file.side}_lang": text_file.language}
        return cls(lines=text_file.lines_as_dicts, split=split, **lang_kwarg)

    @classmethod
    def stack_text_files(
        cls, text_files: Sequence[LoadedTextFile], split: str, verbose: bool = True
    ) -> "CorpusSplit":
        """Create a single CorpusSplit by concatenating lines of multiple files together.

        Note: files must all be the same side (i.e. src or tgt)
        """

        all_languages = set(tf.language for tf in text_files)
        all_sides = set(tf.side for tf in text_files)

        assert len(all_sides) == 1, "All text files must have the same side"

        side = text_files[0].side

        if len(all_languages) > 1:
            language = "multi"
        else:
            language = text_files[0].language

        concatenated_lines = (line for tf in text_files for line in tf.lines_as_dicts)

        lang_kwargs = {("src_lang" if side == "src" else "tgt_lang"): language}

        return cls(
            **lang_kwargs, split=split, verbose=verbose, lines=concatenated_lines
        )

    @classmethod
    def stack_corpus_splits(
        cls, corpus_splits: Sequence["CorpusSplit"], split: str, verbose: bool = True
    ) -> "CorpusSplit":
        """Create a single CorpusSplit by concatenating lines of multiple CorpusSplits together."""

        all_src = list(set(cs.src_lang for cs in corpus_splits))
        assert len(all_src) == 1, "All corpus splits must have the same source language"

        all_tgt = list(set(cs.tgt_lang for cs in corpus_splits))
        assert len(all_tgt) == 1, "All corpus splits must have the same target language"

        src_lang, tgt_lang = all_src[0], all_tgt[0]

        concatenated_lines = (line for cs in corpus_splits for line in cs.lines)

        return cls(
            lines=concatenated_lines,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            split=split,
            verbose=verbose,
        )
