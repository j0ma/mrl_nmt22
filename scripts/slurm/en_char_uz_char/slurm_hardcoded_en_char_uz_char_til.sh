#!/usr/bin/env bash
#
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --job-name=uz_hard
#SBATCH --output=/scratch0/jonnesaleva/hardcoded_en_char_uz_char_til.out
#SBATCH --account=guest
#SBATCH --partition=guest-gpu
#SBATCH --qos=low-gpu
#SBATCH --export=ALL

export MRL_NMT_EXPERIMENT_NAME=en_char_uz_char_slurm
export MRL_NMT_MODEL_NAME=slurmformer
export MRL_NMT_REFERENCES_FILE=/home/jonnesaleva/datasets/mrl_nmt22/processed/en-uz/en_char_uz_char/default-train/en-uz.test.detok.uz
export MRL_NMT_RAW_DATA_FOLDER=/home/jonnesaleva/datasets/mrl_nmt22/processed/en-uz/en_char_uz_char/default-train
export MRL_NMT_BIN_DATA_FOLDER=/home/jonnesaleva/mrl_nmt22/data-bin/en_char_uz_char/default-train/
export MRL_NMT_ENV_NAME=fairseq-py3.8
export MRL_NMT_EXPERIMENTS_FOLDER=/home/jonnesaleva/mrl_nmt22/experiments
export MRL_NMT_CHECKPOINTS_FOLDER=/home/jonnesaleva/mrl_nmt22/checkpoints

bash scripts/slurm/inner_en_char_uz_char_til.sh
