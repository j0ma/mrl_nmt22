#!/usr/bin/env bash
#
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --account=guest
#SBATCH --partition=guest-gpu
#SBATCH --qos=low-gpu
#SBATCH --export=ALL
#SBATCH --output=/scratch0/jonnesaleva/eval.out

set -euo pipefail

# Script that bundles together experiment creation, training and evaluation 

# Parse arguments from environment 
src_lang="${MRL_NMT_SRC_LANG}"
tgt_lang="${MRL_NMT_TGT_LANG}"
experiment_name="${MRL_NMT_EXPERIMENT_NAME}"
model_name="${MRL_NMT_MODEL_NAME}"
eval_name="${MRL_NMT_EVAL_NAME}"
eval_model_checkpoint="${MRL_NMT_EVAL_MODEL_CHECKPOINT}"
references_clean_file="${MRL_NMT_REFERENCES_FILE}"
raw_data_folder="${MRL_NMT_RAW_DATA_FOLDER}"
bin_data_folder="${MRL_NMT_BIN_DATA_FOLDER}"
conda_env_name="${MRL_NMT_ENV_NAME}"
experiments_prefix="${MRL_NMT_EXPERIMENTS_FOLDER}"
checkpoints_prefix="${MRL_NMT_CHECKPOINTS_FOLDER}"
gpu="$CUDA_VISIBLE_DEVICES"
eval_mode="${MRL_NMT_EVAL_MODE}"
remove_preprocessing_hypotheses="${MRL_NMT_REMOVE_PREPROCESSING_HYPOTHESES}"
remove_preprocessing_references="${MRL_NMT_REMOVE_PREPROCESSING_REFERENCES}"
remove_preprocessing_source="${MRL_NMT_REMOVE_PREPROCESSING_SOURCE}"
remove_preprocessing_references_clean="${MRL_NMT_REMOVE_PREPROCESSING_REFERENCES_CLEAN}"
detokenize_hypotheses="${MRL_NMT_DETOKENIZE_HYPOTHESES}"
detokenize_references="${MRL_NMT_DETOKENIZE_REFERENCES}"
detokenize_source="${MRL_NMT_DETOKENIZE_SOURCE}"
detokenize_references_clean="${MRL_NMT_DETOKENIZE_REFERENCES_CLEAN}"

### Source necessary function definitions
source scripts/slurm/slurm_functions.sh

activate_conda_env $conda_env_name

### Create experiment folder & train/eval folders for default corpus
prep_eval \
    $experiment_name \
    $model_name \
    $eval_name \
    $eval_model_checkpoint \
    $raw_data_folder \
    $bin_data_folder \
    $references_clean_file \
    $experiments_prefix \
    $checkpoints_prefix

evaluate \
    $experiment_name \
    $src_lang \
    $tgt_lang \
    $model_name \
    $references_clean_file \
    $remove_preprocessing_hypotheses \
    $remove_preprocessing_references \
    $remove_preprocessing_source \
    $remove_preprocessing_references_clean \
    $detokenize_hypotheses \
    $detokenize_references \
    $detokenize_source \
    $detokenize_references_clean \
    "${CUDA_VISIBLE_DEVICES}" \
    $eval_mode \
    $eval_name
