import itertools
from io import StringIO
from typing import Union, Iterable, Dict, Any, Sequence, Generator
from pathlib import Path
import csv

import attr
import toml


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


def read_lines(path: Union[str, Path]) -> Iterable[str]:
    """Reads lines from a file, outputting an iterable."""
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
    path: Union[str, Path], lines=Iterable[str], check_empty: bool = True
) -> None:
    with open(path, mode="w", encoding="utf-8") as fout:
        for line in lines:
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
