from typing import List

import click
import csv
import sys

from mrl_nmt.scoring import BLEUMetric, CHRFMetric, ExperimentResults, CHRFMetricSB

"""evaluate.py

Computes evaluation metrics for translated sentences:
    - Various BLEU variants
    - CHRF3
"""

DEFAULT_METRICS = [
    BLEUMetric(tokenize="13a", ignore_case=False),
    BLEUMetric(tokenize="intl", ignore_case=False),
    BLEUMetric(tokenize="char", ignore_case=False),
    BLEUMetric(tokenize="spm", ignore_case=False),
    BLEUMetric(tokenize="none", ignore_case=False),
    BLEUMetric(tokenize="13a", ignore_case=True),
    BLEUMetric(tokenize="intl", ignore_case=True),
    BLEUMetric(tokenize="char", ignore_case=True),
    BLEUMetric(tokenize="spm", ignore_case=True),
    BLEUMetric(tokenize="none", ignore_case=True),
    CHRFMetric(beta=3),
    CHRFMetricSB(),
]


def evaluate(
    references_path: str,
    hypotheses_path: str,
    source_path: str,
    combined_tsv_path: str,
    score_output_path: str,
    output_as_tsv: bool,
    src_language: str,
    tgt_language: str,
    skip_header: bool,
    remove_sentencepiece: bool,
    remove_char: bool,
    metrics_to_compute: List[str] = None,
):
    if not metrics_to_compute:
        metrics_to_compute = DEFAULT_METRICS

    if combined_tsv_path:
        results = ExperimentResults.from_tsv(
            tsv_path=combined_tsv_path,
            src_language=src_language,
            tgt_language=tgt_language,
            skip_header=skip_header,
            remove_sentencepiece=remove_sentencepiece,
            remove_char=remove_char,
            metrics_to_compute=metrics_to_compute,
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
            metrics_to_compute=metrics_to_compute,
        )

    with (
        open(score_output_path, "w", encoding="utf-8")
        if score_output_path
        else sys.stdout
    ) as score_out_file:
        if output_as_tsv:
            result_header = results.metric_names + ["Source", "Target"]
            dw = csv.DictWriter(
                f=score_out_file,
                fieldnames=result_header,
                delimiter="\t",
            )

            def result_rows():
                for language, metrics in results.metrics_dict.items():
                    metrics_dict = metrics.format_as_dict()
                    if language == "global":
                        metrics_dict["Source"] = "global"
                        metrics_dict["Target"] = "global"
                    yield metrics_dict

            dw.writeheader()
            for row in result_rows():
                dw.writerow(row)

        else:
            for lang in results.languages:
                if lang in results.metrics_dict:
                    for metric_name, metric_value in (
                        results.metrics_dict.get(lang).format_as_dict().items()
                    ):
                        if metric_name not in ["Source", "Target"]:
                            score_out_file.write(
                                f"EVAL_SCALAR: {lang}-{metric_name}\t{metric_value}\n"
                            )

            # finally write out global
            for metric_name, metric_value in (
                results.metrics_dict.get("global").format_as_dict().items()
            ):
                if metric_name not in ["Source", "Target"]:
                    score_out_file.write(
                        f"EVAL_SCALAR: global-{metric_name}\t{metric_value}\n"
                    )


@click.command()
@click.option("--references-path", "--gold-path", "--ref", "--gold", default="")
@click.option("--hypotheses-path", "--hyp", default="")
@click.option("--source-path", "--src", default="")
@click.option("--combined-tsv-path", "--tsv", default="")
@click.option("--score-output-path", "--score", default="/dev/stdout")
@click.option("--output-as-tsv", is_flag=True)
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
        src_language,
        tgt_language,
        skip_header=skip_header_in_tsv,
        remove_sentencepiece=remove_sentencepiece,
        remove_char=remove_char,
    )


if __name__ == "__main__":
    main()
