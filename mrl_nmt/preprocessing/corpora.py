from pathlib import Path
from typing import Union, Set, Optional, Callable, Any, Iterable, Dict, Tuple, Sequence, Mapping

import attr
from lxml import etree as etree

from mrl_nmt import utils as u


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
class LoadedXLIFFFile(LoadedFile):

    language: str = "multi"
    side: str = "both"

    src_language: Optional[str] = None
    tgt_language: Optional[str] = None

    load_to_memory: bool = True

    def __attrs_post_init__(self):
        parsed_xml = self.parse_xml()
        self.src_lines = parsed_xml["src"]["lines"]
        self.tgt_lines = parsed_xml["tgt"]["lines"]

    def parse_xml(self) -> Dict[str, Dict[str, Union[str, Iterable[str]]]]:

        # load xml into memory
        tree = etree.parse(self.path).getroot()

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

        for side, lang in zip(("src", "tgt"), (self.src_lang, self.tgt_lang)):
            if lang:
                output_path = folder / f"{prefix}.{lang}"
                lines = (line[side] for line in self.lines)
                u.write_lines(path=output_path, lines=lines)

    @classmethod
    def from_src_tgt(
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