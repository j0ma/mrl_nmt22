from typing import List, TextIO, Dict, Tuple, Set
from tqdm import tqdm
import sacrebleu
import click
import attr
import csv
import sys

from nltk.translate.chrf_score import corpus_chrf
import mrl_nmt.utils as u

"""evaluate.py

Computes evaluation metrics for translated sentences:
    - BLEU
    - CHRF3
"""


def maybe_search(pattern: str, string: str, guess: int = 5) -> int:
    guess = 5  # this is the normal case with 3-letter ISO code
    try:
        assert string[guess] == ">"

        return guess
    except (IndexError, AssertionError, ValueError):
        return string.index(">")


def read_text(path: str) -> TextIO:
    return open(path, encoding="utf-8")


@attr.s(auto_attribs=True)
class Postprocessor:

    remove_sentencepiece: bool = False
    remove_char: bool = False

    def __attrs_post_init__(self):
        assert not (
            self.remove_sentencepiece and self.remove_char
        ), "Can only convert one of {SP, char} to word-level"
        if self.remove_sentencepiece:
            self.postprocess = self.bpe_to_words()
        elif self.remove_char:
            self.postprocess = self.chars_to_words()
        else:
            self.postprocess = lambda s: s

    def __call__(self, s: str) -> str:
        return self.postprocess(str)

    def bpe_to_words(self, s: str) -> str:
        return s.replace(" ", "").replace(u.SP_BOW_SYMBOL, " ")

    def chars_to_words(self, s: str) -> str:
        return s.replace(" ", "").replace(u.SPACE_SYMBOL, " ")


@attr.s(kw_only=True)  # kw_only ensures we are explicit
class TranslationOutput:
    """Represents a single translation output, consisting of a
    source language and line, reference translation and a model hypothesis.
    """

    src_language: str = attr.ib(default="en")
    tgt_language: str = attr.ib()
    reference: str = attr.ib()
    hypothesis: str = attr.ib()
    source: str = attr.ib(default="")


@attr.s(kw_only=True)
class TranslationMetrics:
    """Score container for a collection of translation results."""

    chrf3: float = attr.ib(factory=float)
    bleu: float = attr.ib(factory=float)
    rounding: int = attr.ib(default=5)
    src_language: str = attr.ib(default="")
    tgt_language: str = attr.ib(default="")

    def __attrs_post_init__(self) -> None:
        self.chrf3 = round(self.chrf3, self.rounding)
        self.bleu = round(self.bleu, self.rounding)

    def format(self) -> str:
        return f"CHRF3\t{self.chrf3:.4f}\nBLEU\t{self.bleu:.4f}\n\n"


def chrf_score(
    system_outputs: List[TranslationOutput],
    min_len: int = 1,
    max_len: int = 6,
    beta: int = 3,
    ignore_whitespace: bool = False,
) -> float:
    hypotheses = [o.hypothesis for o in system_outputs]
    references = [o.reference for o in system_outputs]
    score = corpus_chrf(
        references=references,
        hypotheses=hypotheses,
        min_len=min_len,
        max_len=max_len,
        beta=beta,
        ignore_whitespace=ignore_whitespace,
    )
    return score


def bleu_score(system_outputs: List[TranslationOutput]) -> float:
    hypotheses = [o.hypothesis for o in system_outputs]
    references = [[o.reference for o in system_outputs]]
    bleu_obj = sacrebleu.corpus_bleu(hypotheses, references, force=True)

    return bleu_obj.score / 100.0  # divide to normalize


@attr.s(kw_only=True)
class TranslationResults:
    system_outputs: List[TranslationOutput] = attr.ib(factory=list)
    metrics: TranslationMetrics = attr.ib(factory=TranslationMetrics)

    def __attrs_post_init__(self) -> None:
        self.metrics = self.compute_metrics()

    def compute_metrics(self) -> TranslationMetrics:

        unique_src_languages: Set[str] = set()
        unique_tgt_languages: Set[str] = set()
        for o in self.system_outputs:
            unique_src_languages.add(o.src_language)
            unique_tgt_languages.add(o.tgt_language)

        if len(unique_tgt_languages) > 1:
            tgt_language = "global"
        else:
            tgt_language = list(unique_tgt_languages)[0]

        if len(unique_src_languages) > 1:
            src_language = "global"
        else:
            src_language = list(unique_src_languages)[0]

        bleu = 100 * bleu_score(self.system_outputs)
        chrf3 = 100 * chrf_score(self.system_outputs)

        metrics = TranslationMetrics(
            chrf3=chrf3, bleu=bleu, src_language=src_language, tgt_language=tgt_language
        )

        return metrics


@attr.s(kw_only=True)
class ExperimentResults:
    system_outputs: List[TranslationOutput] = attr.ib(factory=list)
    languages: Set[str] = attr.ib(factory=set)
    grouped: bool = attr.ib(default=True)
    metrics_dict: Dict[str, TranslationResults] = attr.ib(factory=dict)

    def __attrs_post_init__(self) -> None:
        self.metrics_dict = self.compute_metrics_dict()

    def compute_metrics_dict(self) -> Dict[str, TranslationResults]:

        # first compute global metrics
        metrics = {"global": TranslationResults(system_outputs=self.system_outputs)}

        # then compute one for each target lang

        for lang in tqdm(self.languages, total=len(self.languages)):
            filtered_outputs = [
                o for o in self.system_outputs if o.tgt_language == lang
            ]
            if filtered_outputs:
                metrics[lang] = TranslationResults(system_outputs=filtered_outputs)

        return metrics

    @classmethod
    def outputs_from_paths(
        cls,
        references_path: str,
        hypotheses_path: str,
        source_path: str,
        src_language: str = "en",
        infer_src_language: bool = False,
        tgt_language: str = "",
        infer_tgt_language: bool = False,
        remove_sentencepiece: bool = False,
        remove_char: bool = False,
    ) -> Tuple[List[TranslationOutput], Set[str]]:

        postprocess = Postprocessor(
            remove_sentencepiece=remove_sentencepiece, remove_char=remove_char
        )

        if not src_language:
            infer_src_language = True
        if not tgt_language:
            infer_tgt_language = True

        with read_text(hypotheses_path) as hyp, read_text(
            references_path
        ) as ref, read_text(source_path) as src:
            system_outputs = []
            languages: Set[str] = set()

            for hyp_line, ref_line, src_line in zip(hyp, ref, src):

                # figure out language
                if infer_src_language:
                    src_language_ix = maybe_search(
                        pattern=">", string=src_line, guess=4
                    )
                    src_language = (
                        # this will throw ValueError if incorrectly formatted
                        src_line[:src_language_ix]
                        .replace("<", "")
                        .replace(">", "")
                    ).strip()
                if infer_tgt_language:
                    tgt_language_ix = maybe_search(
                        pattern=">", string=hyp_line, guess=4
                    )
                    tgt_language = (
                        # this will throw ValueError if incorrectly formatted
                        hyp_line[:tgt_language_ix]
                        .replace("<", "")
                        .replace(">", "")
                    ).strip()

                # record observed languages
                languages.update((src_language, tgt_language))

                # TODO: source side postprocessing
                hypothesis = postprocess(hyp_line.strip())
                reference = postprocess(ref_line.strip())
                source = src_line.strip()
                system_outputs.append(
                    TranslationOutput(
                        src_language=src_language,
                        tgt_language=tgt_language,
                        reference=reference,
                        hypothesis=hypothesis,
                        source=source,
                    )
                )

            return system_outputs, languages

    @classmethod
    def outputs_from_combined_tsv(
        cls,
        combined_tsv_path: str,
        src_language: str = "en",
        infer_src_language: bool = False,
        tgt_language: str = "",
        infer_tgt_language: bool = False,
        skip_header: bool = False,
        remove_sentencepiece: bool = False,
        remove_char: bool = False,
    ) -> Tuple[List[TranslationOutput], Set[str]]:

        if not src_language:
            infer_src_language = True
        if not tgt_language:
            infer_tgt_language = True

        columns = ["hyp", "ref", "src"]
        output_rows = u.read_tsv_dict(
            path=combined_tsv_path,
            field_names=columns,
            delimiter="\t",
            skip_header=skip_header,
            load_to_memory=True,
        )

        for row in output_rows:
            if infer_src_language:
                src = row["src"]
                closing_bracket_ix = maybe_search(pattern=">", string=src, guess=4)
                row["src_language"] = (
                    # ugly line to infer language from a tag like <eng>
                    src[:closing_bracket_ix]
                    .replace("<", "")
                    .replace(">", "")
                    .strip()
                )
            if infer_tgt_language:
                tgt = row["hyp"]
                closing_bracket_ix = maybe_search(pattern=">", string=tgt, guess=4)
                row["tgt_language"] = (
                    # ugly line to infer language from a tag like <eng>
                    tgt[:closing_bracket_ix]
                    .replace("<", "")
                    .replace(">", "")
                    .strip()
                )

        languages: Set[str] = set()
        system_outputs = []
        for row in tqdm(output_rows):
            src_lang = row.get("src_language", src_language)
            tgt_lang = row.get("tgt_language", tgt_language)
            languages.update((src_lang, tgt_lang))
            system_outputs.append(
                TranslationOutput(
                    src_language=src_lang,
                    tgt_language=tgt_lang,
                    reference=row["ref"],
                    hypothesis=row["hyp"],
                    source=row["src"],
                )
            )

        return system_outputs, languages

    @classmethod
    def from_paths(
        cls,
        references_path: str,
        hypotheses_path: str,
        source_path: str,
        grouped: bool = True,
        src_language: str = "",
        tgt_language: str = "",
        remove_sentencepiece: bool = False,
        remove_char: bool = False,
    ):
        system_outputs, languages = cls.outputs_from_paths(
            references_path,
            hypotheses_path,
            source_path,
            src_language=src_language,
            tgt_language=tgt_language,
            remove_sentencepiece=remove_sentencepiece,
            remove_char=remove_char,
        )

        return cls(
            system_outputs=system_outputs,
            grouped=grouped,
            languages=languages,
        )

    @classmethod
    def from_tsv(
        cls,
        tsv_path: str,
        grouped: bool = True,
        src_language: str = "",
        tgt_language: str = "",
        skip_header: bool = False,
        remove_sentencepiece: bool = False,
        remove_char: bool = False,
    ):
        system_outputs, languages = cls.outputs_from_combined_tsv(
            tsv_path,
            src_language=src_language,
            tgt_language=tgt_language,
            skip_header=skip_header,
            remove_sentencepiece=remove_sentencepiece,
            remove_char=remove_char,
        )

        return cls(
            system_outputs=system_outputs,
            grouped=grouped,
            languages=languages,
        )

    def as_dict_rows(self):
        _languages = self.languages | set(["global"])

        rows = []
        for lang in _languages:

            try:
                row = attr.asdict(self.metrics_dict[lang].metrics)
                row = {
                    "BLEU": row["bleu"],
                    "CHRF3": row["chrf3"],
                    "Source": row["src_language"],
                    "Target": row["tgt_language"],
                }
                rows.append(row)
            except KeyError:
                pass

        return rows


def evaluate(
    references_path: str,
    hypotheses_path: str,
    source_path: str,
    combined_tsv_path: str,
    score_output_path: str,
    output_as_tsv: bool,
    output_as_json: bool,
    src_language: str,
    tgt_language: str,
    skip_header: bool,
    remove_sentencepiece: bool,
    remove_char: bool,
):

    if combined_tsv_path:
        results = ExperimentResults.from_tsv(
            tsv_path=combined_tsv_path,
            src_language=src_language,
            tgt_language=tgt_language,
            skip_header=skip_header,
            remove_sentencepiece=remove_sentencepiece,
            remove_char=remove_char,
        )
    else:
        results = ExperimentResults.from_paths(
            references_path=references_path,
            hypotheses_path=hypotheses_path,
            source_path=source_path,
            src_language=src_language,
            tgt_language=tgt_language,
            remove_sentencepiece=remove_sentencepiece,
            remove_char=remove_char,
        )

    with (
        open(score_output_path, "w", encoding="utf-8")
        if score_output_path
        else sys.stdout
    ) as score_out_file:
        if output_as_tsv:
            result_rows = results.as_dict_rows()
            dw = csv.DictWriter(
                f=score_out_file,
                fieldnames=["BLEU", "CHRF3", "Source", "Target"],
                delimiter="\t",
            )
            dw.writeheader()
            for row in result_rows:
                dw.writerow(row)

        else:
            for lang in results.languages:
                if lang in results.metrics_dict:
                    score_out_file.write(f"{lang}:\n")
                    score_out_file.write(
                        results.metrics_dict.get(lang).metrics.format()
                    )

            # finally write out global
            score_out_file.write("global:\n")
            score_out_file.write(results.metrics_dict.get("global").metrics.format())


@click.command()
@click.option("--references-path", "--gold-path", "--ref", "--gold", default="")
@click.option("--hypotheses-path", "--hyp", default="")
@click.option("--source-path", "--src", default="")
@click.option("--combined-tsv-path", "--tsv", default="")
@click.option("--score-output-path", "--score", default="/dev/stdout")
@click.option("--output-as-tsv", is_flag=True)
@click.option("--output-as-json", is_flag=True)
@click.option(
    "--src-language",
    help="Optionally specify a global source language (default: en)",
    default="en",
)
@click.option(
    "--tgt-language",
    help="Optionally specify a global target language (default: inferred)",
    default="",
)
@click.option(
    "--skip-header-in-tsv",
    is_flag=True,
    help="Skip header when parsing input TSV",
    default=False,
)
@click.option(
    "--remove-sentencepiece",
    is_flag=True,
    help="Convert sentencepiece => words before scoring",
    default=False,
)
@click.option(
    "--remove-char",
    is_flag=True,
    help="Convert chars => words before scoring",
    default=False,
)
def main(
    references_path: str,
    hypotheses_path: str,
    source_path: str,
    combined_tsv_path: str,
    score_output_path: str,
    output_as_tsv: bool,
    output_as_json: bool,
    src_language: str,
    tgt_language: str,
    skip_header_in_tsv: bool,
    remove_sentencepiece: bool,
    remove_char: bool,
):
    evaluate(
        references_path,
        hypotheses_path,
        source_path,
        combined_tsv_path,
        score_output_path,
        output_as_tsv,
        output_as_json,
        src_language,
        tgt_language,
        skip_header=skip_header_in_tsv,
        remove_sentencepiece=remove_sentencepiece,
        remove_char=remove_char,
    )


if __name__ == "__main__":
    m
