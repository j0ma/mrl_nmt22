#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Note: originally taken from fairseq with slight modifications (mainly adding rich)

import argparse
import fileinput
import hashlib
import sys
from multiprocessing import Pool

from rich.progress import track

def get_hashes_and_lines(raw_line):
    _hash = hashlib.md5(raw_line).hexdigest()

    return _hash, raw_line


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("files", nargs="*", help="input files")
    args = parser.parse_args()

    seen = set()
    with fileinput.input(args.files, mode="rb") as h:
        pool = Pool(args.workers)
        results = pool.imap_unordered(get_hashes_and_lines, h, 1000)

        for i, (_hash, raw_line) in track(enumerate(results), total=100):
            if _hash not in seen:
                seen.add(_hash)
                sys.stdout.buffer.write(raw_line)

    print(file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
