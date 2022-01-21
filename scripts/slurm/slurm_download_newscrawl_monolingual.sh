#!/usr/bin/env bash
#
#SBATCH --cpus-per-task=6
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --output=/scratch0/jonnesaleva/download_newscrawl.out
#SBATCH --account=guest
#SBATCH --partition=guest-compute
#SBATCH --qos=low

set -euo pipefail
source /home/$(whoami)/miniconda3/etc/profile.d/conda.sh

conda_env_name=${MRL_NMT_ENV_NAME:-fairseq-py3.8}
conda activate $conda_env_name

echo "Base download folder: ${MRL_NMT_DOWNLOAD_FOLDER}"

for lang in "cs" "de" "ru" "fi" "et" "tr"
do
    guild run data:download_newscrawl_mono -y \
        --tag "newscrawl-${lang}" \
        corpus_name="${lang}_mono" \
        language="${lang}" \
        download_folder="${MRL_NMT_DOWNLOAD_FOLDER}/${lang}/mono"
done
