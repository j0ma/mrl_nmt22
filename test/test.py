import os
from pathlib import Path
import unittest
import itertools as it

from tqdm import tqdm

import mrl_nmt.utils as u
import mrl_nmt.preprocessing as pp
import mrl_nmt.preprocessing.corpora
from mrl_nmt.preprocessing import ops

# if we're not in test/ append it to the paths
prefix = Path("") if str(Path.cwd()).endswith("/test") else Path("test")

# Paths to sample sentences
SAMPLE_SENTENCES_TXT = prefix / Path("data/sample_sentences.txt")
SAMPLE_TSV = prefix / Path("data/sample.tsv")
SAMPLE_NOHEADER_TSV = prefix / Path("data/sample_noheader.tsv")

# Paths to Flores 101 dev data
SWE_DEV = prefix / Path("data/flores101_swe.dev")
FIN_DEV = prefix / Path("data/flores101_fin.dev")
ENG_DEV = prefix / Path("data/flores101_eng.dev")
TUR_DEV = prefix / Path("data/flores101_tur.dev")
ENG_FIN_DEV = prefix / Path("data/flores101_eng_fin_dev.tsv")
ENG_FIN_DEV_WITHHEADER = prefix / Path("data/flores101_eng_fin_dev_withheader.tsv")

# Paths to XLF data
CS_EN_XLF_STUB_PATH = prefix / Path("data/rapid_2019_cs_en_stub.xlf")

# Paths to TMX data
EN_FI_TMX_STUB_PATH = prefix / Path("data/rapid2016.en-fi_stub.tmx")

# Case: local development
local_development_path = "/data/datasets/mrl-nmt"
FULL_DATA_PATH = Path(local_development_path).expanduser()

# Case: server development
server_development_path = "~/datasets/mrl_nmt22/"
if not FULL_DATA_PATH.exists():
    print(f"FULL_DATA_PATH not found under {local_development_path}")
    FULL_DATA_PATH = Path(server_development_path).expanduser()

# Fallback: try to pick from environment
if not FULL_DATA_PATH.exists():
    print(f"FULL_DATA_PATH not found under {server_development_path}")
    try:
        FULL_DATA_PATH = Path(os.environ.get("MRL_FULL_DATA_PATH")).expanduser()
    except:
        print(
            "WARNING: Could not resolve $MRL_FULL_DATA_PATH. "
            "Ignoring tests that use it."
        )
        FULL_DATA_PATH = Path("/dev/null")

FULL_EN_CS_PATH = FULL_DATA_PATH / "cs" / "en-cs"
FULL_EN_TR_PATH = FULL_DATA_PATH / "tr" / "en-tr"
FULL_EN_UZ_PATH = FULL_DATA_PATH / "uz" / "en-uz"

# Constants
N_LINES_FLORES101_DEV = 997
N_LINES_PARACRAWL = 14083311
N_LINES_EN_TR_TRAIN = 35879592

# Toggle this to enable testing with large files
SKIP_LARGE_TESTS = True


class TestTextUtils(unittest.TestCase):
    def test_can_read_txt_into_memory(self):
        sample_sents = u.read_lines(path=SAMPLE_SENTENCES_TXT)
        self.assertEqual(3, len(sample_sents))
        u.print_a_few_lines(sample_sents)

    def test_can_stream_txt(self):
        sample_sents = u.stream_lines(path=SAMPLE_SENTENCES_TXT)

        # generators/streams have no length
        with self.assertRaises(TypeError):
            assert 3 == len(sample_sents)

        u.print_a_few_lines(sample_sents)

    def test_can_read_tsv_into_memory(self):
        sample_tsv = u.read_tsv_dict(
            path=SAMPLE_TSV,
            field_names=["language", "sentence"],
            load_to_memory=True,
            skip_header=True,
        )
        self.assertEqual(list, type(sample_tsv))
        self.assertEqual(3, len(sample_tsv))

    def test_can_stream_tsv(self):
        sample_tsv = u.read_tsv_dict(
            path=SAMPLE_TSV,
            field_names=["language", "sentence"],
            load_to_memory=False,
            skip_header=True,
        )

        # the stream should consist of 3 lines
        line_count = 0
        for _ in sample_tsv:
            line_count += 1
        self.assertEqual(3, line_count)

        # the stream should also not have a len() attribute
        with self.assertRaises(TypeError):
            assert 3 == len(sample_tsv)


class TestLoadedTextFileRAM(unittest.TestCase):
    """Loading Flores 101 Swedish data to memory works."""

    def setUp(self):
        self.swe_in_ram_src = mrl_nmt.preprocessing.corpora.LoadedTextFile(
            path=SWE_DEV, side="src", load_to_memory=True, language="swe"
        )
        self.swe_in_ram_tgt = mrl_nmt.preprocessing.corpora.LoadedTextFile(
            path=SWE_DEV, side="tgt", load_to_memory=True, language="swe"
        )

    def test_right_number_of_lines(self):
        self.assertEqual(N_LINES_FLORES101_DEV, len(self.swe_in_ram_src.lines))

    def test_right_number_of_dicts(self):
        self.assertEqual(N_LINES_FLORES101_DEV, len(self.swe_in_ram_src.lines_as_dicts))

    def test_can_load_lines_twice(self):
        """The lines can be iterated over several times, e.g. for getting bigrams."""

        print("\n" + 70 * "=")
        print("Here are a few line bigrams:\n")

        u.print_a_few_lines(
            msg="Here area few line bigrams:",
            line_iter=(
                {"first": first, "second": second}
                for first, second in enumerate(
                    zip(self.swe_in_ram_src.lines, self.swe_in_ram_src.lines[1:])
                )
            ),
        )

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
        self.swe_streaming_src = mrl_nmt.preprocessing.corpora.LoadedTextFile(
            path=SWE_DEV, side="src", load_to_memory=False, language="swe"
        )
        self.swe_streaming_tgt = mrl_nmt.preprocessing.corpora.LoadedTextFile(
            path=SWE_DEV, side="tgt", load_to_memory=False, language="swe"
        )

    def test_right_number_of_lines(self):
        line_count = 0

        for line in self.swe_streaming_src.stream_lines:
            line_count += 1

        self.assertEqual(N_LINES_FLORES101_DEV, line_count)

    def test_right_number_of_dicts(self):

        line_count = 0

        for line in self.swe_streaming_src.lines_as_dicts:
            line_count += 1

        self.assertEqual(N_LINES_FLORES101_DEV, line_count)

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

        u.print_a_few_lines(
            msg="Here are a few line bigrams:",
            line_iter=(
                {"first": first, "second": second}
                for first, second in enumerate(zip(first_iter, second_iter))
            ),
        )

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
            _ = mrl_nmt.preprocessing.corpora.LoadedTSVFile(
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
            _ = mrl_nmt.preprocessing.corpora.LoadedTSVFile(
                path=ENG_FIN_DEV,
                src_column="eng",
                tgt_column="fin",
                language="multi",
                src_language="eng",
                tgt_language="fin",
                side="both",
            )

    def test_int_columns_without_fieldnames_noerror(self):
        _ = mrl_nmt.preprocessing.corpora.LoadedTSVFile(
            path=ENG_FIN_DEV,
            src_column=0,
            tgt_column=1,
            language="multi",
            src_language="eng",
            tgt_language="fin",
            side="both",
        )

    def test_str_columns_with_fieldnames_noerror(self):
        _ = mrl_nmt.preprocessing.corpora.LoadedTSVFile(
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
        self.xliff = mrl_nmt.preprocessing.corpora.LoadedXLIFFFile(
            path=str(CS_EN_XLF_STUB_PATH),
            src_language="cs",
            tgt_language="en",
        )

    def test_can_print_lines(self) -> None:
        u.print_a_few_lines(self.xliff.lines_as_dicts)


class TestTMXFile(unittest.TestCase):
    def test_can_print_lines_stream(self) -> None:
        self.tmx_file = mrl_nmt.preprocessing.corpora.LoadedTMXFile(
            path=str(EN_FI_TMX_STUB_PATH),
            src_language="en",
            tgt_language="fi",
            load_to_memory=False,
        )
        lines = self.tmx_file.lines_as_dicts
        u.print_a_few_lines(lines)

    def test_can_print_lines_ram(self) -> None:
        self.tmx_file = mrl_nmt.preprocessing.corpora.LoadedTMXFile(
            path=str(EN_FI_TMX_STUB_PATH),
            src_language="en",
            tgt_language="fi",
            load_to_memory=True,
        )
        lines = self.tmx_file.lines_as_dicts
        u.print_a_few_lines(lines)

    # TODO: more tests


class TestCorpusSplit(unittest.TestCase):
    def setUp(self) -> None:
        self.fin = mrl_nmt.preprocessing.corpora.LoadedTextFile(
            path=FIN_DEV, side="src", language="fi", load_to_memory=False
        )
        self.swe = mrl_nmt.preprocessing.corpora.LoadedTextFile(
            path=SWE_DEV, side="tgt", language="sv", load_to_memory=False
        )
        self.swe_src = mrl_nmt.preprocessing.corpora.LoadedTextFile(
            path=SWE_DEV, side="src", language="sv", load_to_memory=False
        )

    def test_can_align_src_tgt(self) -> None:

        cs = mrl_nmt.preprocessing.corpora.CorpusSplit.from_src_tgt(
            src=self.fin, tgt=self.swe, split="train"
        )
        u.print_a_few_lines(line_iter=cs.lines)

    def test_src_tgt_side_mismatch_raises_error(self):
        """An error is raised if src and tgt don't have correct sides."""

        with self.assertRaises(AssertionError):
            _ = mrl_nmt.preprocessing.corpora.CorpusSplit.from_src_tgt(
                src=self.swe, tgt=self.fin, split="train"
            )

    def test_stack_text_files(self):

        files = [self.fin, self.swe_src]
        _ = mrl_nmt.preprocessing.corpora.CorpusSplit.stack_text_files(
            text_files=files, split="train"
        )

    def test_stack_text_files_side_mismatch_raises_error(self):

        files = [self.fin, self.swe]
        with self.assertRaises(AssertionError):
            _ = mrl_nmt.preprocessing.corpora.CorpusSplit.stack_text_files(
                text_files=files, split="train"
            )


class TestPreprocessingOps(unittest.TestCase):
    def setUp(self) -> None:
        self.fin = mrl_nmt.preprocessing.corpora.LoadedTextFile(
            path=FIN_DEV, side="src", language="fi", load_to_memory=False
        )
        self.swe = mrl_nmt.preprocessing.corpora.LoadedTextFile(
            path=SWE_DEV, side="src", language="sv", load_to_memory=False
        )
        self.swe_tgt = mrl_nmt.preprocessing.corpora.LoadedTextFile(
            path=SWE_DEV, side="tgt", language="sv", load_to_memory=False
        )
        self.eng = mrl_nmt.preprocessing.corpora.LoadedTextFile(
            path=ENG_DEV, side="tgt", language="en", load_to_memory=False
        )
        self.fin_corpus = mrl_nmt.preprocessing.CorpusSplit.from_text_file(
            text_file=self.fin, split="dev"
        )
        self.swe_tgt_corpus = mrl_nmt.preprocessing.CorpusSplit.from_text_file(
            text_file=self.swe_tgt, split="dev"
        )

    def test_convert_to_chars(self):
        self.fin_chars = ops.convert_to_chars(
            corpus=self.fin_corpus, side=self.fin.side
        )
        u.print_a_few_lines(self.fin_chars.lines)

    def test_duplicate_lines(self):
        self.fin_5x_round_robin = ops.duplicate_lines(
            mrl_nmt.preprocessing.CorpusSplit.from_text_file(
                text_file=self.fin, split="train"
            ),
            n=5,
        )
        self.fin_5x_repeat_each = ops.duplicate_lines(
            mrl_nmt.preprocessing.CorpusSplit.from_text_file(
                text_file=self.fin, split="train"
            ),
            n=5,
            round_robin=False,
        )

        n_lines_round_robin = 0
        n_lines_repeat_each = 0

        for rr_line, re_line in it.zip_longest(
            self.fin_5x_round_robin.lines, self.fin_5x_repeat_each.lines
        ):
            n_lines_round_robin += int(bool(rr_line))
            n_lines_repeat_each += int(bool(re_line))

        self.assertEqual(n_lines_round_robin, n_lines_repeat_each)

    def test_create_multilingual_corpus(self):

        en_corpus = ops.duplicate_lines(
            mrl_nmt.preprocessing.CorpusSplit.from_text_file(
                text_file=self.eng, split="train"
            ),
            2,
        )
        multi_corpus = mrl_nmt.preprocessing.CorpusSplit.stack_text_files(
            text_files=[self.fin, self.swe], split="train"
        )

        combined_corpus = mrl_nmt.preprocessing.CorpusSplit.from_src_tgt(
            src=multi_corpus, tgt=en_corpus, split="train"
        )

    def test_load_commoncrawl(self):
        mrl_nmt.preprocessing.ops.load_commoncrawl(
            folder=prefix / "data", src_language="cs", tgt_language="en"
        )

    def test_sentencepiece_fi_sv(self):
        """SentencePiece Processing works on dev set for FI-SV"""
        sp_vocab_size = 777
        fi_model = Path("/tmp/fi_model.bin", is_file=True)
        sv_model = Path("/tmp/sv_model.bin", is_file=True)

        # align two loaded text files into source and target
        corpus = pp.CorpusSplit.from_src_tgt(
            src=self.fin, tgt=self.swe_tgt, split="dev", verbose=True
        )

        # train src side model, process text and save
        corpus = pp.ops.process_with_sentencepiece(
            corpus=corpus,
            side="src",
            vocab_size=sp_vocab_size,
            use_pretrained_model=False,
            model_file=fi_model,
        )

        # train tgt side model, process text and save
        corpus = pp.ops.process_with_sentencepiece(
            use_pretrained_model=False,
            corpus=corpus,
            side="tgt",
            vocab_size=sp_vocab_size,
            model_file=sv_model,
        )
        print("Model trained and saved.")

        # load again and process using pretrained model
        # (this is redundant in practice, ofc)
        new_corpus = pp.CorpusSplit.from_src_tgt(
            src=self.fin, tgt=self.swe_tgt, split="dev", verbose=True
        )
        print(f"Re-applying src side processing with {fi_model}.")
        new_corpus = pp.ops.process_with_sentencepiece(
            corpus=new_corpus,
            side="src",
            vocab_size=sp_vocab_size,
            use_pretrained_model=True,
            model_file=fi_model,
        )
        print(f"Re-applying tgt side processing with {sv_model}.")
        new_corpus = pp.ops.process_with_sentencepiece(
            corpus=new_corpus,
            side="tgt",
            vocab_size=sp_vocab_size,
            use_pretrained_model=True,
            model_file=sv_model,
        )

        new_corpus.write_to_disk(folder=Path("/tmp/"), prefix="", skip_upon_fail=True)

        u.print_a_few_lines(
            "Source: {}\nTarget: {}".format(d["tgt"]["text"], d["src"]["text"])
            for d in corpus.lines
        )

    @unittest.skipIf(
        condition=not FULL_EN_CS_PATH.exists(),
        reason=f"{FULL_EN_CS_PATH} not found!",
    )
    def test_load_paracrawl(self):
        pc = mrl_nmt.preprocessing.ops.load_paracrawl(
            folder=FULL_EN_CS_PATH, foreign_language="cs"
        )
        line_count = 0
        for line in pc.lines:
            if line_count < 5:
                print(line)
            line_count += 1

        self.assertEqual(N_LINES_PARACRAWL, line_count)

    @unittest.skipIf(
        condition=not FULL_EN_CS_PATH.exists(),
        reason=f"{FULL_EN_CS_PATH} not found!",
    )
    def test_stack_corpus_splits_num_lines_simple(self):
        def load(input_base_folder):
            news_commentary_train = pp.ops.load_news_commentary(
                folder=input_base_folder,
                src_language="cs",
                tgt_language="en",
                split=split,
            )
            rapid_train = pp.ops.load_rapid_2019_xlf(
                folder=input_base_folder,
                src_language="cs",
                tgt_language="en",
                split=split,
            )
            return news_commentary_train, rapid_train

        input_base_folder = FULL_DATA_PATH / "cs" / "en-cs"
        split = "train"
        news_commentary_train, rapid_train = load(input_base_folder)
        line_count_news = sum(1 for _ in news_commentary_train.lines)
        line_count_rapid = sum(1 for _ in rapid_train.lines)
        self.assertGreater(line_count_news, 0)
        self.assertGreater(line_count_rapid, 0)

        train = pp.corpora.CorpusSplit.stack_corpus_splits(
            corpus_splits=[c for c in load(input_base_folder)],
            split="train",
        )
        line_count_combined = sum(1 for _ in train.lines)
        self.assertEqual(line_count_combined, line_count_news + line_count_rapid)


class TestParallelCorpusStats(unittest.TestCase):
    @unittest.skipIf(
        condition=(not TUR_DEV.exists()),
        reason="EN-TR dev data not found.",
    )
    def test_en_tr_dev_size(self):
        """EN-TR dev data contains the correct number of lines, regardless of preprocessing"""

        levels = ("word", "char", "sentencepiece", "morph")
        sp_conf = {
            "src": {
                "vocab_size": 200,
                "use_pretrained_model": False,
                "model_file": "entr.en.bin",
            },
            "tgt": {
                "vocab_size": 200,
                "use_pretrained_model": False,
                "model_file": "entr.tr.bin",
            },
        }

        for src_lvl, tgt_lvl in zip(levels, levels):
            if src_lvl != "morph" and tgt_lvl != "morph":
                en_tr_corpus = ops.process_tr(
                    input_base_folder=prefix / "data",
                    split="dev",
                    en_output_level=src_lvl,
                    tr_output_level=tgt_lvl,
                    prefix="flores101_",
                    sentencepiece_config=sp_conf,
                )
                line_count = 0
                for _ in tqdm(en_tr_corpus.lines):
                    line_count += 1

                self.assertEqual(N_LINES_FLORES101_DEV, line_count)
            else:
                # NOTE: this will fail once morph processing implemented
                with self.assertRaises(NotImplementedError):
                    en_tr_corpus = ops.process_tr(
                        input_base_folder=prefix / "data",
                        split="dev",
                        en_output_level=src_lvl,
                        tr_output_level=tgt_lvl,
                        prefix="flores101_",
                        sentencepiece_config=sp_conf,
                    )

    @unittest.skipIf(condition=SKIP_LARGE_TESTS, reason="SKIP_LARGE_TESTS=True")
    def test_en_tr_train_size(self):
        """EN-TR train data contains the correct number of lines, regardless of preprocessing"""

        levels = ("word", "char", "sentencepiece", "morph")
        sp_conf = {
            "src": {
                "vocab_size": 2000,
                "use_pretrained_model": False,
                "model_file": "entr.en.bin",
                "input_sentence_size": 500000,
                "shuffle_input_sentence": True,
            },
            "tgt": {
                "vocab_size": 2000,
                "use_pretrained_model": False,
                "model_file": "entr.tr.bin",
                "input_sentence_size": 500000,
                "shuffle_input_sentence": True,
            },
        }

        for src_lvl, tgt_lvl in zip(levels, levels):
            if src_lvl != "morph" and tgt_lvl != "morph":
                en_tr_corpus = ops.process_tr(
                    input_base_folder=FULL_EN_TR_PATH,
                    split="train",
                    en_output_level=src_lvl,
                    tr_output_level=tgt_lvl,
                    sentencepiece_config=sp_conf,
                )
                line_count = 0
                for _ in tqdm(en_tr_corpus.lines):
                    line_count += 1

                self.assertEqual(N_LINES_FLORES101_DEV, line_count)
            else:
                # NOTE: this will fail once morph processing implemented
                with self.assertRaises(NotImplementedError):
                    en_tr_corpus = ops.process_tr(
                        input_base_folder=FULL_EN_TR_PATH,
                        split="train",
                        en_output_level=src_lvl,
                        tr_output_level=tgt_lvl,
                        sentencepiece_config=sp_conf,
                    )


if __name__ == "__main__":
    unittest.main(verbosity=1234)
