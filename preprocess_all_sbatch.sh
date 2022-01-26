#!/usr/bin/env bash

set -euo pipefail

export MRL_NMT_ENV_NAME="fairseq-py3.8"

#for yml_file in $(ls scripts/slurm/en_sp*/en*.yaml | rg -v "en_sp32k_iu" | rg -v "en_sp32k_uz")
#do

    #export MRL_NMT_YAML_CONFIG="$(pwd)/${yml_file}"

    #lang_inferred=$(basename $yml_file | cut -d'_' -f3)
    #sbatch --mail-user=jonnesaleva@brandeis.edu --mail-type=ALL --job-name="pp${lang_inferred}" scripts/slurm/slurm_preprocess_yaml_config.sh
#done

for yml_file in $(ls scripts/slurm/en_spmbart_*/*.yaml)
do
    export MRL_NMT_YAML_CONFIG="$(pwd)/${yml_file}"

    lang_inferred=$(basename $yml_file | cut -d'_' -f3)
    sbatch --mail-user=jonnesaleva@brandeis.edu --mail-type=ALL --job-name="bart-${lang_inferred}" scripts/slurm/slurm_preprocess_yaml_config.sh
done

squeue -u $(whoami)
