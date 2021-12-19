import csv
import itertools as it
from collections import defaultdict
from pathlib import Path

import mrl_nmt.preprocessing as pp
import mrl_nmt.utils as u
from tqdm import tqdm
import click


def fetch_lines(
    references_path,
    references_clean_path,
    hypotheses_path,
    source_path,
):
    with u.read_text(hypotheses_path) as hyp_lines, u.read_text(
        references_path
    ) as ref_lines, u.read_text(source_path) as src_lines, u.read_text(
        references_clean_path
    ) as ref_clean_lines:
        for ref, hyp, src, ref_clean in it.zip_longest(
            ref_lines, hyp_lines, src_lines, ref_clean_lines
        ):
            yield ref, hyp, src, ref_clean


@click.command()
@click.option(
    "--references-path",
    "--gold-path",
    "--ref",
    "--gold",
    help="Path to potentially preprocessed references (used in training)",
)
@click.option(
    "--references-clean-path",
    "--gold-clean-path",
    "--ref-clean",
    "--gold-clean",
    help="Path to untouched references (used in final eval, probably at most tokenized)",
)
@click.option(
    "--hypotheses-path",
    "--hyp",
    help="Path to hypotheses/model outputs (probably as subwords, at least tokenized)",
)
@click.option(
    "--source-path",
    "--src",
    help="Path to source side inputs (probably as subwords, at least tokenized)",
)
@click.option(
    "--remove-processing-src",
    help="Processing to remove from source (e.g. sentencepiece)",
    type=click.Choice(pp.PROCESSING_TYPES),
)
@click.option(
    "--remove-processing-hyp",
    help="Processing to remove from hypotheses (e.g. sentencepiece)",
    type=click.Choice(pp.PROCESSING_TYPES),
)
@click.option(
    "--remove-processing-ref",
    help="Processing to remove from references (e.g. sentencepiece)",
    type=click.Choice(pp.PROCESSING_TYPES),
)
@click.option(
    "--remove-processing-ref-clean",
    help="Processing to remove from clean references (most likely nothing)",
    type=click.Choice(pp.PROCESSING_TYPES),
)
@click.option(
    "--detokenize-src",
    help="Detokenize source lines. (model output, most likely yes)",
    type=bool,
)
@click.option(
    "--detokenize-ref",
    help="Detokenize reference lines? (model output, most likely yes)",
    type=bool,
)
@click.option(
    "--detokenize-hyp",
    help="Detokenize hypothesis lines? (model output, most likely yes)",
    type=bool,
)
@click.option(
    "--detokenize-ref-clean",
    help="Detokenize clean references? (most likely no)",
    type=bool,
)
@click.option("--source-output-path", type=click.Path())
@click.option("--hypotheses-output-path", type=click.Path())
@click.option("--references-output-path", type=click.Path())
@click.option("--references-clean-output-path", type=click.Path())
def main(
    references_path,
    references_clean_path,
    hypotheses_path,
    source_path,
    remove_processing_hyp,
    remove_processing_src,
    remove_processing_ref,
    remove_processing_ref_clean,
    detokenize_hyp,
    detokenize_src,
    detokenize_ref,
    detokenize_ref_clean,
    source_output_path,
    hypotheses_output_path,
    references_output_path,
    references_clean_output_path,
):
    sides = ["hyp", "src", "ref", "ref_clean"]
    processing_to_remove = [
        remove_processing_hyp,
        remove_processing_src,
        remove_processing_ref,
        remove_processing_ref_clean,
    ]
    whether_to_detokenize = [
        detokenize_hyp,
        detokenize_src,
        detokenize_ref,
        detokenize_ref_clean,
    ]
    postprocessors = {
        side: pp.Postprocessor.from_strings(processing_to_remove, detokenize)

        for side, processing_to_remove, detokenize in zip(
            sides, processing_to_remove, whether_to_detokenize
        )
    }
    outputs = defaultdict(list)

    print("[postprocess] Postprocessing lines:")

    for ref_hyp_src_refclean in tqdm(
        fetch_lines(
            references_path,
            references_clean_path,
            hypotheses_path,
            source_path,
        )
    ):
        for side, line in zip(sides, ref_hyp_src_refclean):
            postprocess = postprocessors[side]
            processed_line = postprocess(line)
            outputs[side].append(processed_line)

    print("[postprocess] Outputting lines:")
    output_paths = [
        hypotheses_output_path,
        source_output_path,
        references_output_path,
        references_clean_output_path,
    ]
    assert all(
        bool(p) for p in output_paths
    ), f"Need all output paths, got: {output_paths}"

    for side, output_path in zip(sides, output_paths):
        print(f"[postprocess] Writing: {side} to {output_path}")
        u.write_lines(path=Path(output_path), lines=outputs[side])


if __name__ == "__main__":
    main()
