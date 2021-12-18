import sys
import requests
import zipfile
import os
import argparse

from pathlib import Path

# Based on download_data.py from
# https://github.com/turkic-interlingua/til-mt/blob/master/til_corpus/download_data.py

base_url = "gs://til-corpus/corpus"

# all the langs present in the corpus
langs = [
    "alt",
    "az",
    "ba",
    "cjs",
    "crh",
    "cv",
    "gag",
    "kaa",
    "kjh",
    "kk",
    "krc",
    "kum",
    "ky",
    "sah",
    "slr",
    "tk",
    "tr",
    "tt",
    "tyv",
    "ug",
    "uum",
    "uz",
    "en",
    "ru",
]

# downloads a zip file given a url
def download_url(url, save_path):
    try:
        os.system(f"gsutil -m cp -r {url} {save_path}")
    except CommandException as e:
        print("Requested test files are not found!")


def unzip(path_to_zip_file, directory_to_extract_to):
    with zipfile.ZipFile(path_to_zip_file, "r") as zip_ref:
        zip_ref.extractall(directory_to_extract_to)


def create_folder(base_path, source, target, split):

    folder_path = Path(f"{base_path}/{source}-{target}/{split}")

    if not folder_path.exists():
        folder_path.mkdir(exist_ok=True, parents=True)

        return True
    else:
        print(f"{split} data already exists! Skipping...")

        return False


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source_language",
        default=None,
        type=str,
        required=True,
        help="2-letter code of the source language",
    )
    parser.add_argument(
        "--target_language",
        default=None,
        type=str,
        required=True,
        help="2-letter code of the target language",
    )
    parser.add_argument(
        "--split",
        default="all",
        type=str,
        required=False,
        help='which split to download. Options: \{"train", "dev", "test"\}',
    )
    parser.add_argument(
        "--download_folder",
        type=str,
        required=True,
        help="Where to download files. Final data will be under <download_folder>/<src>-<tgt>",
    )
    args = parser.parse_args()

    source = args.source_language
    target = args.target_language
    download_folder = Path(args.download_folder).expanduser().resolve()

    # check if the language code exists in the corpus

    if source not in langs or target not in langs:
        print("Language code error!!!\nLanguage code must be one of these: ")
        print(langs)
        quit()

    if args.split not in ["train", "dev", "test", "all"]:
        print("Splits should be one of the following: [train, dev, test, all]")
        quit()

    splits = ["train", "dev", "test"] if args.split == "all" else [args.split]

    for split in splits:
        create_folder(download_folder, source, target, split)

        # base save path
        save_path = download_folder / f"{source}-{target}"

        if not save_path.exists():
            save_path.mkdir(parents=True, exist_ok=True)

        save_path_str = str(save_path)

        if split == "test":
            if create_folder(download_folder, source, target, f"{split}/bible"):
                test_bible_path = f"{base_url}/test/bible/{source}-{target}"
                test_bible_download_path = f"{save_path_str}/test/bible/"
                print(f"Downloading bible test files to {test_bible_download_path}")
                download_url(test_bible_path, test_bible_download_path)

            if create_folder(download_folder, source, target, f"{split}/ted"):
                test_ted_path = f"{base_url}/test/ted/{source}-{target}"
                test_ted_download_path = f"{save_path_str}/test/ted/"
                print(f"Downloading ted talk test files to {test_ted_path}")
                download_url(test_ted_path, test_ted_download_path)

            if create_folder(download_folder, source, target, f"{split}/x-wmt"):
                test_x_wmt_path = f"{base_url}/test/x-wmt/{source}-{target}"
                test_x_wmt_download_path = f"{save_path_str}/test/x-wmt/"
                print(f"Downloading x-wmt test files to {test_x_wmt_path}")
                download_url(test_x_wmt_path, test_x_wmt_download_path)

        else:
            download_path = f"{base_url}/{split}/{source}-{target}"
            split_folder = f"{save_path_str}/{split}/"
            print(f"Downloading {split} files to {split_folder}")
            download_url(download_path, split_folder)
