#!/usr/bin/env python

# analyze_split_overlap.py

from collections import Counter
import itertools as it

import mrl_nmt.utils as u
from tqdm import tqdm
from rich import print as rprint
import click

MAX_LEN = 10

def pretty_print_set(x, max_len=MAX_LEN):
    lst = list(x)[:MAX_LEN]
    if not lst:
        rprint(lst)
    else:
        rprint("[{}, ...]".format(", ".join(lst)))


def tokenize(line):
    """Override later?"""
    return line.split(" ")


def get_char_vocab(p):
    rprint(f"Getting character vocab from {p}")
    vocab = set()
    for line in tqdm(u.stream_lines(p)):
        vocab.update(c for c in line)
    return vocab


def get_subword_vocab(p):
    rprint(f"Getting subword vocab from {p}")
    vocab = set()
    for line in tqdm(u.stream_lines(p)):
        vocab.update(tokenize(line))
    return vocab


def iou(a, b):
    return len(a & b) / len(a | b)


@click.command()
@click.option(
    "--train",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
)
@click.option(
    "--dev", type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True)
)
@click.option(
    "--test",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
)
def main(train, dev, test):

    train_chars = get_char_vocab(train)
    dev_chars = get_char_vocab(dev)
    test_chars = get_char_vocab(test)

    train_subwords = get_subword_vocab(train)
    dev_subwords = get_subword_vocab(dev)
    test_subwords = get_subword_vocab(test)

    rprint(f"Character IOU overlap (train-dev): {round(iou(train_chars, dev_chars), 3)}")
    rprint(f"Character IOU overlap (train-test): {round(iou(train_chars, test_chars), 3)}")

    oov_chars_dev = dev_chars - train_chars
    oov_chars_test = test_chars - train_chars

    rprint(f"{len(oov_chars_dev)} OOV characters in [cyan]dev")
    pretty_print_set(oov_chars_dev)
    rprint(f"{len(oov_chars_test)} OOV characters in [cyan]test")
    pretty_print_set(oov_chars_test)

    rprint(f"Subword IOU overlap (train-dev): {round(iou(train_subwords, dev_subwords), 3)}")
    rprint(
        f"Subword IOU overlap (train-test): {round(iou(train_subwords, test_subwords), 3)}"
    )

    oov_subwords_dev = dev_subwords - train_subwords
    oov_subwords_test = test_subwords - train_subwords

    rprint(f"{len(oov_subwords_dev)} OOV subwords in [cyan]dev")
    pretty_print_set(oov_subwords_dev)
    rprint(f"{len(oov_subwords_test)} OOV subwords in [cyan]test")
    pretty_print_set(oov_subwords_test)


if __name__ == "__main__":
    main()
