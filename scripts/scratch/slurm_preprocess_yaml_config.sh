#!/usr/bin/env bash
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --job-name=preprocess_yaml_config
#SBATCH --output=/scratch0/jonnesaleva/preprocess_yaml_config_%j.out
#SBATCH --account=guest
#SBATCH --partition=guest-gpu
#SBATCH --qos=low-gpu
#SBATCH --gres=low-gpu
#SBATCH --export=ALL,MRL_NMT_YAML_CONFIG
#SBATCH --gres=gpu:V100:1   # Request V100 GPUs

set -euo pipefail

module load anaconda

# Script that runs a given preprocessing op given a YAML config.
# Meant to be used inside SLURM.

# Set up Conda environment 
conda_env_name="${MRL_NMT_ENV_NAME:-fairseq-py3.8}"
source /home/$(whoami)/miniconda3/etc/profile.d/conda.sh
conda activate $conda_env_name

python scripts/text_processing/preprocess.py \
    --yaml-config ${MRL_NMT_YAML_CONFIG}
    --use-gpu --gpu-devices "${CUDA_VISIBLE_DEVICES}"
