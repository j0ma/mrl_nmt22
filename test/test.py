from pathlib import Path
import unittest
import sys

import mrl_nmt.preprocessing as pp

# if we're not in test/ append it to the paths
prefix = Path("") if str(Path.cwd()).endswith("/test") else Path("test")

# Paths to Flores 101 dev data
SWE_DEV = prefix / Path("data/flores101_swe.dev")
FIN_DEV = prefix / Path("data/flores101_fin.dev")
ENG_FIN_DEV = prefix / Path("data/flores101_eng_fin_dev.tsv")
ENG_FIN_DEV_WITHHEADER = prefix / Path("data/flores101_eng_fin_dev_withheader.tsv")

# Paths to XLF data
CS_EN_XLF_STUB_PATH = prefix / Path("data/rapid_2019_cs_en_stub.xlf")


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


class TestXLIFFFile(unittest.TestCase):
    def setUp(self) -> None:
        self.xliff = pp.LoadedXLIFFFile(
            path=str(CS_EN_XLF_STUB_PATH),
            src_language="cs",
            tgt_language="en",
        )

    def test_can_print_lines(self) -> None:

        print("\n" + 70 * "=")
        print("Here are a few lines:\n")

        with open("/dev/null", "w") as dev_null:
            for ix, line in enumerate(self.xliff.lines_as_dicts):
                print(line, file=(sys.stdout if ix < 5 else dev_null))
        print("\n" + 70 * "=")


class TestCorpusSplit(unittest.TestCase):
    def setUp(self) -> None:
        self.fin = pp.LoadedTextFile(
            path=FIN_DEV, side="src", language="fi", load_to_memory=False
        )
        self.swe = pp.LoadedTextFile(
            path=SWE_DEV, side="tgt", language="sv", load_to_memory=False
        )
        self.swe_src = pp.LoadedTextFile(
            path=SWE_DEV, side="src", language="sv", load_to_memory=False
        )

    def test_can_align_src_tgt(self) -> None:

        cs = pp.CorpusSplit.from_src_tgt(src=self.fin, tgt=self.swe, split="train")

        print("\n" + 70 * "=")
        print("Here are a few lines:\n")

        with open("/dev/null", "w") as dev_null:
            for ix, line in enumerate(cs.lines):
                print(line, file=(sys.stdout if ix < 5 else dev_null))
        print("\n" + 70 * "=")

    def test_src_tgt_side_mismatch_raises_error(self):
        """An error is raised if src and tgt don't have correct sides."""

        with self.assertRaises(AssertionError):
            cs = pp.CorpusSplit.from_src_tgt(src=self.swe, tgt=self.fin, split="train")

    def test_from_several_files(self):

        files = [self.fin, self.swe_src]
        cs = pp.CorpusSplit.from_several_files(text_files=files, split="train")
        print(cs)

    def test_from_several_files_side_mismatch_raises_error(self):

        files = [self.fin, self.swe]
        with self.assertRaises(AssertionError):
            cs = pp.CorpusSplit.from_several_files(text_files=files, split="train")


if __name__ == "__main__":
    unittest.main(verbosity=1234)
