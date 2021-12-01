from html import unescape
from pathlib import Path
from typing import Union
from xml.etree import ElementTree as ET

import attr
import typing

from mtdata.iso.bcp47 import bcp47
from mtdata.utils import IO
from mtdata.main import lang_pair


@attr.s(auto_attribs=True)
class TMXHandler:
    """
    Quick TMX handler, based largely on the TMX parsing implementation in mtdata.

    With legacy_mode=True, output is written to self.output_file.
    With legacy_mode=False, output is yielded as an iterable of dicts.
        The individual dicts look like this:
        {
            <src_language>: <src_line>
            <tgt_language>: <tgt_line>
        }

    and are intended for downstream processing with things that expect
    outputs similar to csv.DictWriter(fieldnames=[src_language, tgt_language])
    """

    src_language: str
    tgt_language: str
    input_file: Union[str, Path]
    output_file: typing.Optional[Union[str, Path]] = None

    legacy_mode: bool = False

    def __attrs_post_init__(self):
        if self.output_file:
            assert self.legacy_mode
            self.output = IO.writer(self.output_file)

    def parse_tmx(self, data):
        context = ET.iterparse(data, events=["end"])
        tus = (el for event, el in context if el.tag == "tu")
        count = 0
        for tu in tus:
            lang_seg = {}
            for tuv in tu.findall("tuv"):
                lang = [v for k, v in tuv.attrib.items() if k.endswith("lang")]
                seg = tuv.findtext("seg")
                if lang and seg:
                    lang = bcp47(lang[0])
                    seg = unescape(seg.strip()).replace("\n", " ").replace("\t", " ")
                    lang_seg[lang] = seg
            yield lang_seg
            count += 1
            tu.clear()

    def read_tmx(self, path: Union[Path, str], langs=None):
        """
        reads a TMX file as records
        :param path: path to .tmx file
        :param langs: (lang1, lang2) codes eg (de, en); when it is None the code tries to auto detect
        :return: stream of (text1, text2)
        """
        passes = 0
        fails = 0
        with IO.reader(path) as data:
            recs = self.parse_tmx(data)
            for lang_seg in recs:
                if langs is None:
                    if len(lang_seg) == 2:
                        langs = tuple(lang_seg.keys())
                    else:
                        raise Exception(
                            f"Language autodetect for TMX only supports 2 languages,"
                            f" but provided with {lang_seg.keys()} in TMX {path}"
                        )
                if langs[0] in lang_seg and langs[1] in lang_seg:
                    yield lang_seg[langs[0]], lang_seg[langs[1]]
                    passes += 1
                else:
                    fails += 1
        if passes == 0:
            if fails == 0:
                raise Exception(f"Empty TMX {path}")
            raise Exception(f"Nothing for {langs[0]}-{langs[1]} in TMX {path}")

    def process(self):
        langpair = lang_pair(f"{self.src_language}-{self.tgt_language}")
        recs = self.read_tmx(self.input_file, langs=langpair)
        if self.legacy_mode:
            for rec in recs:
                rec = [l.replace("\t", " ") for l in rec]
                self.output.write("\t".join(rec) + "\n")
            return self.output
        else:
            for rec in recs:
                rec = [l.replace("\t", " ") for l in rec]
                ld = {
                    lang: line
                    for lang, line in zip((self.src_language, self.tgt_language), rec)
                }
                yield ld
