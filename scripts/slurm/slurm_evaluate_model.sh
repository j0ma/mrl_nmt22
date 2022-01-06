#!/usr/bin/env bash
#
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --job-name=uz32
#SBATCH --output=/scratch0/jonnesaleva/en_sp32k_uz_sp32k_til.out
#SBATCH --account=guest
#SBATCH --partition=guest-gpu
#SBATCH --qos=low-gpu
#SBATCH --export=ALL

# Script that bundles together experiment creation, training and evaluation 

# Parse arguments from environment 
experiment_name="${MRL_NMT_EXPERIMENT_NAME}"
model_name="${MRL_NMT_MODEL_NAME}"
references_file="${MRL_NMT_REFERENCES_FILE}"
raw_data_folder="${MRL_NMT_RAW_DATA_FOLDER}"
bin_data_folder="${MRL_NMT_BIN_DATA_FOLDER}"
conda_env_name="${MRL_NMT_ENV_NAME}"
experiments_prefix="${MRL_NMT_EXPERIMENTS_FOLDER}"
checkpoints_prefix="${MRL_NMT_CHECKPOINTS_FOLDER}"
gpu="$CUDA_VISIBLE_DEVICES"

### Source necessary function definitions
source scripts/slurm/slurm_functions.sh

### Create experiment folder & train/eval folders for default corpus
prep && train #&& evaluate
