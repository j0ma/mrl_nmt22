from typing import Union, Iterable, Dict, Any, Sequence, Generator
from pathlib import Path
import csv

import attr
import toml


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
) -> Iterable[Dict[str, str]]:
    with open(path, encoding="utf-8") as fin:
        dr = csv.DictReader(f=fin, fieldnames=field_names, delimiter=delimiter)

        if load_to_memory:
            return [ld for ld in dr]
        else:
            for line in dr:
                yield line


def read_tsv_list(
    path: Union[str, Path],
    delimiter: str = "\t",
    load_to_memory: bool = False,
) -> Iterable[Sequence[str]]:
    with open(path, encoding="utf-8") as fin:
        r = csv.reader(fin, delimiter=delimiter)

        if load_to_memory:
            return [ld for ld in r]
        else:
            for line in r:
                yield line


def write_lines(path: Union[str, Path], lines=Iterable[str]) -> None:
    with open(path, mode="w", encoding="utf-8") as fout:
        for line in lines:
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
