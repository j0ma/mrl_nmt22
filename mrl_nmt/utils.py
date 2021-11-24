from typing import Union, Iterable, Dict, Any
from pathlib import Path

import attr
import toml


def read_txt(path: Union[str, Path]) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def read_lines(path: Union[str, Path], load_to_memory: bool = False) -> Iterable[str]:
    """Reads lines from a file, outputting an iterable.

    Optionally, loads the lines into memory.
    """
    with open(path, encoding="utf-8") as f:
        lines = (line.strip() for line in f)
        return [line for line in lines] if load_to_memory else lines


def write_lines(path: Union[str, Path], lines=Iterable[str]) -> None:
    with open(path, mode="w", encoding="utf-8") as fout:
        fout.writelines(lines)


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
