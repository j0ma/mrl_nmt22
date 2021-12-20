import itertools
import subprocess
import sys
from io import StringIO
from typing import Union, Iterable, Dict, Any, Sequence, Generator, TextIO
from pathlib import Path
import unicodedata as ud
import csv

import attr
import toml

SPACE_SYMBOL = "﹏"
SP_BOW_SYMBOL = "▁"


@attr.s(auto_attribs=True)
class CustomDictReader:
    """Reader for iterating over files that contain
    lines separated by delimiter (default=tab).
    """

    f: StringIO
    field_names: Sequence[str]
    delimiter: str = "\t"

    def process_line(self, line: str, it=None) -> Dict[str, str]:
        out = {}
        for fn, item in itertools.zip_longest(
            self.field_names, line.split(self.delimiter)
        ):
            if fn:
                out[fn] = item

        return out

    def __iter__(self):
        return self.lines

    @property
    def lines(self) -> Dict[str, str]:
        for line in self.f:
            yield self.process_line(line.strip())


def read_txt(path: Union[str, Path]) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def read_lines(path: Union[str, Path]) -> Sequence[str]:
    """Reads lines from a file, outputting a sequence."""
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f]


def stream_lines(path: Union[str, Path]) -> Generator[str, None, None]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield line.strip()


def read_tsv_dict(
    path: Union[str, Path],
    field_names: Sequence[str],
    delimiter: str = "\t",
    load_to_memory: bool = False,
    skip_header: bool = False,
    use_custom_reader: bool = False,
) -> Iterable[Dict[str, str]]:
    if load_to_memory:
        return read_tsv_dict_ram(
            path=path,
            field_names=field_names,
            delimiter=delimiter,
            skip_header=skip_header,
            use_custom_reader=use_custom_reader,
        )
    else:
        return read_tsv_dict_stream(
            path=path,
            field_names=field_names,
            delimiter=delimiter,
            skip_header=skip_header,
            use_custom_reader=use_custom_reader,
        )


def read_tsv_dict_ram(
    path: Union[str, Path],
    field_names: Sequence[str],
    delimiter: str = "\t",
    skip_header: bool = False,
    use_custom_reader: bool = False,
) -> Iterable[Dict[str, str]]:
    with open(path, encoding="utf-8") as fin:
        if use_custom_reader:
            reader = CustomDictReader(
                f=fin, field_names=field_names, delimiter=delimiter
            )
        else:
            reader = csv.DictReader(f=fin, fieldnames=field_names, delimiter=delimiter)
        lines = [ld for ld in reader]
        return lines[1:] if skip_header else lines


def read_tsv_dict_stream(
    path: Union[str, Path],
    field_names: Sequence[str],
    delimiter: str = "\t",
    skip_header: bool = False,
    use_custom_reader: bool = False,
) -> Iterable[Dict[str, str]]:
    with open(path, encoding="utf-8") as fin:
        if use_custom_reader:
            reader = CustomDictReader(
                f=fin, field_names=field_names, delimiter=delimiter
            )
        else:
            reader = csv.DictReader(f=fin, fieldnames=field_names, delimiter=delimiter)

        if skip_header:
            next(reader, None)
        for line in reader:
            yield line


def read_tsv_list(
    path: Union[str, Path],
    delimiter: str = "\t",
    load_to_memory: bool = False,
    skip_header: bool = False,
) -> Iterable[Sequence[str]]:
    if load_to_memory:
        return read_tsv_list_ram(
            path=path, delimiter=delimiter, skip_header=skip_header
        )
    else:
        return read_tsv_list_stream(
            path=path, delimiter=delimiter, skip_header=skip_header
        )


def read_tsv_list_ram(
    path: Union[str, Path],
    delimiter: str = "\t",
    skip_header: bool = False,
) -> Iterable[Sequence[str]]:
    with open(path, encoding="utf-8") as fin:
        r = csv.reader(fin, delimiter=delimiter)
        lines = [ld for ld in r]
        return lines[1:] if skip_header else lines


def read_tsv_list_stream(
    path: Union[str, Path],
    delimiter: str = "\t",
    skip_header: bool = False,
) -> Iterable[Sequence[str]]:
    with open(path, encoding="utf-8") as fin:
        r = csv.reader(fin, delimiter=delimiter)
        for line in r:
            yield line


def write_lines(
        path: Union[str, Path], lines=Iterable[str], check_empty: bool = True, should_strip: bool = False
) -> None:
    with open(path, mode="w", encoding="utf-8") as fout:
        for line in lines:
            if should_strip:
                line = line.strip()
            if not check_empty or line.strip():
                fout.write(f"{line}\n")


@attr.s(auto_attribs=True)
class TOMLConfigReader:
    def __call__(self, toml_path: Union[str, Path]) -> Dict[str, Any]:
        config_dict = toml.load(toml_path)

        if not self.validate_schema(config_dict):
            raise ValueError("Invalid TOML config!")

        return config_dict

    # TODO: maybe implement this?
    def validate_schema(self, d: dict) -> bool:
        return True


def print_a_few_lines(
    line_iter: Iterable[Any], n_lines: int = 5, msg: str = "Here are a few lines:"
) -> None:
    print("\n" + 70 * "=")
    print(f"{msg}\n")

    with open("/dev/null", "w") as dev_null:
        for ix, line in enumerate(line_iter):
            print(line, file=(sys.stdout if ix < n_lines else dev_null))
    print("\n" + 70 * "=")


def get_line_from_dict(d):
    return d["src"]["text"], d["tgt"]["text"]


def maybe_search(pattern: str, string: str, guess: int = 5) -> int:
    guess = 5  # this is the normal case with 3-letter ISO code
    try:
        assert string[guess] == ">"

        return guess
    except (IndexError, AssertionError, ValueError):
        return string.index(">")


def read_text(path: str) -> TextIO:
    return open(path, encoding="utf-8")


def create_symlink(link_path: Union[str, Path], dest_path: Union[str, Path]) -> None:
    Path(link_path).expanduser().absolute().symlink_to(
        target=Path(dest_path).expanduser().absolute()
    )


def move(source: Union[str, Path], destination: Union[str, Path]) -> None:
    Path(source).expanduser().absolute().rename(
        target=Path(destination).expanduser().absolute()
    )
