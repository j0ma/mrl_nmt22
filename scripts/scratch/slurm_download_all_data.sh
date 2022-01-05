#!/usr/bin/env bash
#
#SBATCH --ncpus-per-task=64
#SBATCH --mem-per-task=64G
#SBATCH --ntasks=1
#SBATCH --job-name=download_all_data
#SBATCH --output=/scratch0/jonnesaleva/download_all_training_data.out
#SBATCH --account=guest
#SBATCH --partition=guest-compute
#SBATCH --qos=low

set -euo pipefail
source /home/$(whoami)/miniconda3/etc/profile.d/conda.sh

conda_env_name=${mrl_nmt_env_name:-fairseq-py3.8}
conda activate $conda_env_name

for til_lang in "uz" "tr"
do
    guild run data:download_til -y --background \
        corpus_name=default-train \
        source_language=en \
        target_language="${til_lang}" \
        download_folder="${mrl_nmt_download_folder}/${til_lang}"
done

for mtdata_lang in "cs" "de" "ru" "fi" "et" "iu" "vi"
do
    guild run data:download_mtdata -y --background \
        corpus_name=default-train \
        source_language=en \
        target_language="${mtdata_lang}" \
        download_folder="${mrl_nmt_download_folder}/${mtdata_lang}"
done

# TODO: other corpora for other languages (e.g. newstest)
