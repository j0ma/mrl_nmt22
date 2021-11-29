from pathlib import Path
import unittest
import sys

import mrl_nmt.preprocessing as pp

# Paths to Flores 101 dev data
SWE_DEV = Path("data/flores101_swe.dev")
ENG_FIN_DEV = Path("data/flores101_eng_fin_dev.tsv")
ENG_FIN_DEV_WITHHEADER = Path("data/flores101_eng_fin_dev_withheader.tsv")

# Constants
N_LINES = 997


class TestLoadedTextFileRAM(unittest.TestCase):
    """Loading Flores 101 Swedish data to memory works."""

    def setUp(self):
        self.swe_in_ram_src = pp.LoadedTextFile(
            path=SWE_DEV, side="src", load_to_memory=True, language="swe"
        )
        self.swe_in_ram_tgt = pp.LoadedTextFile(
            path=SWE_DEV, side="tgt", load_to_memory=True, language="swe"
        )

    def test_right_number_of_lines(self):
        self.assertEqual(N_LINES, len(self.swe_in_ram_src.lines))

    def test_right_number_of_dicts(self):
        self.assertEqual(N_LINES, len(self.swe_in_ram_src.lines_as_dicts))

    def test_can_load_lines_twice(self):
        """The lines can be iterated over several times, e.g. for getting bigrams."""

        print("\n" + 70 * "=")
        print("Here are a few line bigrams:\n")

        with open("/dev/null", "w") as dev_null:
            for ix, (first, second) in enumerate(
                zip(self.swe_in_ram_src.lines, self.swe_in_ram_src.lines[1:])
            ):
                print(
                    {"first": first, "second": second},
                    file=(sys.stdout if ix < 5 else dev_null),
                )
        print("\n" + 70 * "=")

    def test_src_has_only_source(self):
        for line_dict in self.swe_in_ram_src.lines_as_dicts:
            self.assertTrue(line_dict["src"] is not None)
            self.assertTrue(line_dict["tgt"] is None)

    def test_tgt_has_only_target(self):
        for line_dict in self.swe_in_ram_tgt.lines_as_dicts:
            self.assertTrue(line_dict["tgt"] is not None)
            self.assertTrue(line_dict["src"] is None)


class TestLoadedTextFileStream(unittest.TestCase):
    """Loading Flores 101 Swedish data as a stream works."""

    def setUp(self):
        self.swe_streaming_src = pp.LoadedTextFile(
            path=SWE_DEV, side="src", load_to_memory=False, language="swe"
        )
        self.swe_streaming_tgt = pp.LoadedTextFile(
            path=SWE_DEV, side="tgt", load_to_memory=False, language="swe"
        )

    def test_right_number_of_lines(self):
        line_count = 0

        for line in self.swe_streaming_src.stream_lines:
            line_count += 1

        self.assertEqual(N_LINES, line_count)

    def test_right_number_of_dicts(self):

        line_count = 0

        for line in self.swe_streaming_src.lines_as_dicts:
            line_count += 1

        self.assertEqual(N_LINES, line_count)

    def test_no_len_stream_lines(self):
        """Streams do not support len() calls."""

        with self.assertRaises(TypeError):
            print(f"There are {len(self.swe_streaming_src.stream_lines)} lines.")

    def test_no_len_dicts(self):
        """Streams do not support len() calls."""
        with self.assertRaises(TypeError):
            print(f"There are {len(self.swe_streaming_src.lines_as_dicts)} lines.")

    def test_can_stream_lines_twice(self):
        """The lines can be iterated several times, e.g. for getting bigrams."""
        first_iter = self.swe_streaming_src.stream_lines
        second_iter = self.swe_streaming_src.stream_lines

        # throw away first value
        next(second_iter)

        print("\n" + 70 * "=")
        print("Here are a few line bigrams:\n")

        with open("/dev/null", "w") as dev_null:
            for ix, (first, second) in enumerate(zip(first_iter, second_iter)):
                print(
                    {"first": first, "second": second},
                    file=(sys.stdout if ix < 5 else dev_null),
                )
        print("\n" + 70 * "=")

    def test_src_has_only_source(self):
        for line_dict in self.swe_streaming_src.lines_as_dicts:
            self.assertTrue(line_dict["src"] is not None)
            self.assertTrue(line_dict["tgt"] is None)

    def test_tgt_has_only_target(self):
        for line_dict in self.swe_streaming_tgt.lines_as_dicts:
            self.assertTrue(line_dict["tgt"] is not None)
            self.assertTrue(line_dict["src"] is None)


class TestLoadedTSVFileIndexing(unittest.TestCase):
    def test_int_columns_with_fieldnames_error(self):
        with self.assertRaises(AssertionError):
            _ = pp.LoadedTSVFile(
                path=ENG_FIN_DEV,
                src_column=0,
                tgt_column=1,
                language="multi",
                src_language="eng",
                tgt_language="fin",
                fieldnames=["eng", "fin"],
                side="both",
            )

    def test_str_columns_without_fieldnames_error(self):
        with self.assertRaises(AssertionError):
            _ = pp.LoadedTSVFile(
                path=ENG_FIN_DEV,
                src_column="eng",
                tgt_column="fin",
                language="multi",
                src_language="eng",
                tgt_language="fin",
                side="both",
            )

    def test_int_columns_without_fieldnames_noerror(self):
        _ = pp.LoadedTSVFile(
            path=ENG_FIN_DEV,
            src_column=0,
            tgt_column=1,
            language="multi",
            src_language="eng",
            tgt_language="fin",
            side="both",
        )

    def test_str_columns_with_fieldnames_noerror(self):
        _ = pp.LoadedTSVFile(
            path=ENG_FIN_DEV,
            src_column="eng",
            tgt_column="fin",
            language="multi",
            src_language="eng",
            tgt_language="fin",
            fieldnames=["eng", "fin"],
            side="both",
        )
