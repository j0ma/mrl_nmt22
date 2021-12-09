import os
from pathlib import Path
import unittest
import itertools as it
import psutil as psu

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
N_LINES_EN_UZ_TRAIN = 529574

# Set SKIP_LARGE_TESTS=True to always enable large tests
RAM_GB_THRESHOLD = 30
giga_ram = psu.virtual_memory().total // 10 ** 9
if giga_ram <= RAM_GB_THRESHOLD:
    SKIP_LARGE_TESTS = True
    print(
        f"Skipping large tests due to insufficient RAM: {round(giga_ram)} GB <= {RAM_GB_THRESHOLD} GB"
    )
else:
    SKIP_LARGE_TESTS = False


class TestEvaluateScript(unittest.TestCase):
    raise NotImplementedError("TODO")


if __name__ == "__main__":
    unittest.main(verbosity=1234)
