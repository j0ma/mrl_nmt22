from typing import List, Union, Iterable
from pathlib import Path
import abc

import sacremoses as sm


class Tokenizer(abc.ABC):
    def tokenize(self, line: str) -> List[str]:
        raise NotImplementedError


class SentenceLoader:
    def __init__(
        self, tokenizer: Union[Tokenizer, sm.MosesTokenizer], encoding="utf-8"
    ) -> None:
        self.tokenizer = tokenizer
        self.encoding = encoding

    def read_sentences(self, file_path: Path) -> Iterable[List[str]]:
        """Streams data from file_path, tokenizing them on the fly"""
        with open(file_path, encoding=self.encoding) as f_in:
            for line in f_in:
                yield self.tokenizer.tokenize(line)
