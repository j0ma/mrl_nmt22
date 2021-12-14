import os
import unittest
from pathlib import Path

import psutil as psu

import mrl_nmt.scoring
import scripts.evaluate as e

# if we're not in test/ append it to the paths
prefix = Path("") if str(Path.cwd()).endswith("/test") else Path("test")

# Paths to sample sentences
SAMPLE_SENTENCES_TXT = prefix / Path("data/sample_sentences.txt")
SAMPLE_TSV = prefix / Path("data/sample.tsv")
SAMPLE_NOHEADER_TSV = prefix / Path("data/sample_noheader.tsv")

# Paths to Flores 101 dev data
ENG_FIN_FAKE_MT_OUTPUT = prefix / Path("data/flores101_eng_fin_fake_output.tsv")


class TestEvaluateScript(unittest.TestCase):
    def test_eng_fin_fake_mt_output1(self):
        """Mock a MT evaluation scenario by taking the Flores-101 dev data
        for EN-FI and repeating FI side for hypotheses & references.

        Call evaluate just as if calling scripts/evaluate.py as a script.
        """

        print("Test: String output")
        e.evaluate(
            references_path="",
            hypotheses_path="",
            source_path="",
            combined_tsv_path=str(ENG_FIN_FAKE_MT_OUTPUT),
            score_output_path="/dev/stdout",
            output_as_tsv=False,
            output_as_json=False,
            src_language="en",
            tgt_language="fi",
            skip_header=True,
            remove_sentencepiece=False,
            remove_char=False,
        )

        print("Test: TSV output")
        e.evaluate(
            references_path="",
            hypotheses_path="",
            source_path="",
            combined_tsv_path=str(ENG_FIN_FAKE_MT_OUTPUT),
            score_output_path="/dev/stdout",
            output_as_tsv=True,
            output_as_json=False,
            src_language="en",
            tgt_language="fi",
            skip_header=True,
            remove_sentencepiece=False,
            remove_char=False,
        )

    def test_eng_fin_fake_mt_output2(self):
        """Mock a MT evaluation scenario by taking the Flores-101 dev data
        for EN-FI and repeating FI side for hypotheses & references.

        Use the classes from scripts/evaluate.py and check the output values make sense.
        """

        results = mrl_nmt.scoring.ExperimentResults.from_tsv(
            tsv_path=str(ENG_FIN_FAKE_MT_OUTPUT),
            src_language="en",
            tgt_language="fi",
            skip_header=True,
            remove_sentencepiece=False,
            remove_char=False,
            metrics_to_compute=e.DEFAULT_METRICS,
        )

        for lang in results.languages | {"global"}:
            try:
                _res = results.by_language(lang)
                for m in _res.metrics:
                    self.assertEqual(
                        100.0, m.value, msg=f"Expected {m.name} == 100, got {m.value}."
                    )
            except KeyError:
                pass


if __name__ == "__main__":
    unittest.main(verbosity=1234)
