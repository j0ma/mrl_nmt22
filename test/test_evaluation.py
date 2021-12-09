import os
import unittest
from pathlib import Path

import psutil as psu

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
        )

    def test_eng_fin_fake_mt_output2(self):
        """Mock a MT evaluation scenario by taking the Flores-101 dev data
        for EN-FI and repeating FI side for hypotheses & references.

        Use the classes from scripts/evaluate.py and check the output values make sense.
        """

        results = e.ExperimentResults.from_tsv(
            tsv_path=str(ENG_FIN_FAKE_MT_OUTPUT),
            src_language="en",
            tgt_language="fi",
            skip_header=True,
        )

        global_results = results.metrics_dict["global"]
        fi_results = results.metrics_dict["fi"]
        assert (
            global_results.metrics.bleu == 100
        ), f"Expected global BLEU = 100, got BLEU = {global_results.metrics.bleu}"
        assert (
            fi_results.metrics.bleu == 100
        ), f"Expected Finnish BLEU = 100, got BLEU = {fi_results.metrics.bleu}"

        # These will fail once CHRF3 is implemented
        with self.assertRaises(AssertionError):
            assert (
                global_results.metrics.chrf3 == 100
            ), f"Expected global CHRF3 = 100, got CHRF3 = {global_results.metrics.chrf3}"
            assert (
                fi_results.metrics.chrf3 == 100
            ), f"Expected Finnish CHRF3 = 100, got CHRF3 = {fi_results.metrics.chrf3}"


if __name__ == "__main__":
    unittest.main(verbosity=1234)
