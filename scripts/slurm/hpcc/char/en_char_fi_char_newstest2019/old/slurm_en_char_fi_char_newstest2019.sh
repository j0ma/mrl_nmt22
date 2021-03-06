#!/usr/bin/env bash
#
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --job-name=en_char_fi_char_newstest2019
#SBATCH --output=/scratch0/jonnesaleva/en_char_fi_char_newstest2019.out
#SBATCH --account=guest
#SBATCH --partition=guest-gpu
#SBATCH --qos=low-gpu
#SBATCH --gres=gpu:V100:10   # Request V100 GPUs

bash scripts/slurm/en_char_fi_char_newstest2019.sh \
    en_char_fi_char_slurm slurmformer \
    /home/jonnesaleva/datasets/mrl_nmt22/processed/en-fi/en_char_fi_char/default-train/en-fi.dev.detok.fi \
    /home/jonnesaleva/datasets/mrl_nmt22/processed/en-fi/en_char_fi_char/default-train \
    ./data-bin/en-fi/en_char_fi_char/default-train/ \
    /home/jonnesaleva/datasets/mrl_nmt22/processed/en-fi/en_char_fi_char/newstest-2019/en-fi.train.detok.fi \
    /home/jonnesaleva/datasets/mrl_nmt22/processed/en-fi/en_char_fi_char/newstest-2019 \
    ./data-bin/en-fi/en_char_fi_char/newstest-2019/
