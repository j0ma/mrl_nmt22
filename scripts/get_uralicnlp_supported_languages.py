import sys
import csv

import pycountry as pc
from uralicNLP import uralicApi

# save keys here as constants in case they change later
MORPH = "morph"
DICT = "dictionary"
CG = "cg"

# delimiter as well
DELIM = "\t"


def get_human_readable_language(lang_code: str) -> str:
    return pc.languages.lookup(lang_code).name


def main():
    supported = uralicApi.supported_languages()

    all_lang_codes = set()

    for category, lang_codes in supported.items():
        all_lang_codes = all_lang_codes | set(lang_codes)

    output_rows = [
        {
            "lang_code": lang_code,
            "language": get_human_readable_language(lang_code),
            "morphology": lang_code in supported[MORPH],
            "dictionary": lang_code in supported[DICT],
            "constraint_grammar": lang_code in supported[CG],
        }

        for lang_code in all_lang_codes
    ]

    output_field_names = list(output_rows[0].keys())

    with sys.stdout as fout:
        dw = csv.DictWriter(fout, fieldnames=output_field_names, delimiter=DELIM)
        dw.writeheader()

        for row in output_rows:
            dw.writerow(row)


if __name__ == "__main__":
    main()
