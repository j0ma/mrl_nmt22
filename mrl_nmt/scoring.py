from typing import Optional, Callable, Iterable, List, Dict, Union, Tuple, Set

import attr
import sacrebleu
from nltk.translate.chrf_score import corpus_chrf
from tqdm import tqdm

from mrl_nmt import utils as u
from mrl_nmt.preprocessing import Postprocessor


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
    verbose: bool = attr.ib(default=False)

    def __attrs_post_init__(self):
        if self.verbose:
            print(f"[TranslationOutput] Source: {self.source}")
            print(f"[TranslationOutput] Reference: {self.reference}")
            print(f"[TranslationOutput] Hypothesis: {self.hypothesis}")


@attr.s(auto_attribs=True)
class Metric:
    name: str = "MetricNeedsAName"
    value: float = 0.0

    callable: Optional[Callable] = None

    def round(self, rounding: int):
        self.value = round(self.value, rounding)

    def compute_value(self, system_outputs: Iterable[TranslationOutput]) -> None:
        if (not self.callable) or self.value:
            return

        self.value = self.callable(system_outputs)


@attr.s(auto_attribs=True)
class BLEUMetric(Metric):

    tokenize: str = "13a"
    ignore_case: bool = False

    def __attrs_post_init__(self):
        self.callable = self._compute_bleu
        self.name = f"BLEU-{self.tokenize}-{'lc' if self.ignore_case else 'mixed'}"

    def _compute_bleu(self, system_outputs: List[TranslationOutput]) -> float:
        hypotheses = [o.hypothesis for o in system_outputs]
        references = [[o.reference for o in system_outputs]]
        bleu = sacrebleu.BLEU(lowercase=self.ignore_case, tokenize=self.tokenize)

        score = bleu.corpus_score(hypotheses, references).score
        print(f"SacreBLEU signature: {bleu.get_signature()}")

        return score


@attr.s(auto_attribs=True)
class CHRFMetric(Metric):

    min_len: int = 1
    max_len: int = 6
    beta: int = 3
    ignore_whitespace: bool = False

    def __attrs_post_init__(self):
        self.callable = self._compute_chrf
        self.name = f"CHRF{self.beta}-min{self.min_len}-max{self.max_len}"

    def _compute_chrf(
        self,
        system_outputs: List[TranslationOutput],
    ) -> float:
        hypotheses = [o.hypothesis for o in system_outputs]
        references = [o.reference for o in system_outputs]
        score = corpus_chrf(
            references=references,
            hypotheses=hypotheses,
            min_len=self.min_len,
            max_len=self.max_len,
            beta=self.beta,
            ignore_whitespace=self.ignore_whitespace,
        )
        return 100 * score


@attr.s(kw_only=True)
class TranslationMetrics:
    """Container for a set of scores for some system outputs.

    Metrics to compute must be Metric objects and should be passed
    in as the metrics argument to the initializer.
    """

    system_outputs: List[TranslationOutput] = attr.ib(factory=list)
    metrics: Iterable[Metric] = attr.ib(factory=list)

    rounding: int = attr.ib(default=5)

    def __attrs_post_init__(self) -> None:

        self.src_language, self.tgt_language = self.infer_languages()

        self.metric_cache = {}
        for metric in self.metrics:
            metric.round(self.rounding)
            metric.compute_value(self.system_outputs)
            self.metric_cache[metric.name] = metric

    def __getitem__(self, item: str):
        return self.metric_cache[item]

    def format(self) -> str:
        return "\n".join(f"{m.name}\t{m.value}" for m in self.metrics) + "\n\n"

    def format_as_dict(self) -> Dict[str, Union[str, float]]:
        out = {m.name: m.value for m in self.metrics}
        out["Source"] = self.src_language
        out["Target"] = self.tgt_language
        return out

    @property
    def metric_names(self) -> List[str]:
        return list({m.name for m in self.metrics})

    def infer_languages(self) -> Tuple[str, str]:
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

        return src_language, tgt_language


@attr.s(kw_only=True)
class ExperimentResults:
    system_outputs: List[TranslationOutput] = attr.ib(factory=list)
    metrics_to_compute: Iterable[Metric] = attr.ib(factory=list)
    languages: Set[str] = attr.ib(factory=set)
    grouped: bool = attr.ib(default=True)

    def __attrs_post_init__(self) -> None:
        self.metrics_dict = self.compute_metrics_dict()

    def by_language(self, language: str) -> TranslationMetrics:
        try:
            return self.metrics_dict[language]
        except KeyError:
            raise KeyError(f"Metrics not found for language: {language}")

    @property
    def metric_names(self) -> List[str]:
        names = set()
        for results in self.metrics_dict.values():
            names.update(results.metric_names)
        return list(names)

    def compute_metrics_dict(self) -> Dict[str, TranslationMetrics]:

        # first compute global metrics
        metrics = {
            "global": TranslationMetrics(
                system_outputs=self.system_outputs, metrics=self.metrics_to_compute
            )
        }

        # then compute one for each target lang

        for lang in tqdm(self.languages, total=len(self.languages)):
            filtered_outputs = [
                o for o in self.system_outputs if o.tgt_language == lang
            ]
            if filtered_outputs:
                metrics[lang] = TranslationMetrics(
                    system_outputs=filtered_outputs, metrics=self.metrics_to_compute
                )

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

        with u.read_text(hypotheses_path) as hyp, u.read_text(
            references_path
        ) as ref, u.read_text(source_path) as src:
            system_outputs = []
            languages: Set[str] = set()

            for hyp_line, ref_line, src_line in zip(hyp, ref, src):

                # figure out language
                if infer_src_language:
                    src_language_ix = u.maybe_search(
                        pattern=">", string=src_line, guess=4
                    )
                    src_language = (
                        # this will throw ValueError if incorrectly formatted
                        src_line[:src_language_ix]
                        .replace("<", "")
                        .replace(">", "")
                    ).strip()
                if infer_tgt_language:
                    tgt_language_ix = u.maybe_search(
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

                hypothesis = postprocess(hyp_line.strip())
                reference = ref_line.strip()
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

        postprocess = Postprocessor(
            remove_sentencepiece=remove_sentencepiece, remove_char=remove_char
        )

        if not src_language:
            infer_src_language = True
        if not tgt_language:
            infer_tgt_language = True

        columns = ["ref", "hyp", "src", "ref_detok"]
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
                closing_bracket_ix = u.maybe_search(pattern=">", string=src, guess=4)
                row["src_language"] = (
                    # ugly line to infer language from a tag like <eng>
                    src[:closing_bracket_ix]
                    .replace("<", "")
                    .replace(">", "")
                    .strip()
                )
            if infer_tgt_language:
                tgt = row["hyp"]
                closing_bracket_ix = u.maybe_search(pattern=">", string=tgt, guess=4)
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
                    reference=row["ref_detok"],
                    hypothesis=postprocess(row["hyp"]),
                    source=row["src"],
                )
            )

        return system_outputs, languages

    @classmethod
    def from_paths(
        cls,
        metrics_to_compute: Iterable[Metric],
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
            metrics_to_compute=metrics_to_compute,
        )

    @classmethod
    def from_tsv(
        cls,
        metrics_to_compute: Iterable[Metric],
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
            metrics_to_compute=metrics_to_compute,
        )
