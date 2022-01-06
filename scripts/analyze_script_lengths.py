#!/usr/bin/env python

# analyze_script_lengths.py

from collections import Counter
import itertools as it

import mrl_nmt.utils as u
from tqdm import tqdm
import click

def tokenize(line):
    """Override later?"""
    return line.split(" ")

def find_longest_lines(src_path, tgt_path, print_longest, with_counters):

    max_src = 0
    max_tgt = 0
    src_lines = u.stream_lines(src_path)
    tgt_lines = u.stream_lines(tgt_path)
    n_src_lines = 0; n_tgt_lines = 0
    if with_counters:
        src_line_counter = Counter()
        tgt_line_counter = Counter()
    for ix, (src_line, tgt_line) in enumerate(tqdm(it.zip_longest(src_lines, tgt_lines))):
        n_src_lines += int(src_line is not None)
        n_tgt_lines += int(tgt_line is not None)
        old_max_src = max_src
        old_max_tgt = max_tgt
        src_len = len(tokenize(src_line))
        tgt_len = len(tokenize(tgt_line))
        max_src = max(max_src, src_len)
        max_tgt = max(max_tgt, tgt_len)
        if with_counters:
            src_line_counter[src_len] += 1
            tgt_line_counter[tgt_len] += 1

        if old_max_src != max_src:
            longest_src_line = src_line
            longest_src_ix = ix
        if old_max_tgt != max_tgt:
            longest_tgt_line = tgt_line
            longest_tgt_ix = ix
    print(f"Lines read: {n_src_lines} (src), {n_tgt_lines} (tgt)")
    print(f"Longest line lengths: {max_src} (src), {max_tgt} (tgt)")
    print(f"Longest lines at index: {longest_src_ix} (src), {longest_tgt_ix} (tgt)")

    if print_longest:
        print(longest_src_line)
        print(longest_tgt_line)
    
    if with_counters:
        for side, counts in [("src", src_line_counter), ("tgt", tgt_line_counter)]:
            print(f"# Line count distribution ({side})")
            for length, n in sorted(counts.most_common(), key=lambda t: t[0], reverse=True):
                print(f"{length}\t{n}")

            print("\n\n")

@click.command()
@click.option("--src", type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True))
@click.option("--tgt", type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True))
@click.option("--print-longest-line", is_flag=True)
@click.option("--with-distributions", is_flag=True)
def main(src, tgt, print_longest_line, with_distributions):
    
    find_longest_lines(src, tgt, print_longest=print_longest_line, with_counters=with_distributions)
    if not print_longest_line:
        print("Run with --print-longest-line to see longest lines")


if __name__ == "__main__":
    main()
