#!/usr/bin/env bash

set -euo pipefail

# Script that bundles together experiment creation, training and evaluation 

# Parse arguments from environment (passed by Guild)
experiment_name="${MRL_NMT_EXPERIMENT_NAME}"
model_name="${MRL_NMT_MODEL_NAME}"
references_file="${MRL_NMT_REFERENCES_FILE}"
raw_data_folder="${MRL_NMT_RAW_DATA_FOLDER}"
bin_data_folder="${MRL_NMT_BIN_DATA_FOLDER}"
conda_env_name="${MRL_NMT_ENV_NAME}"
experiments_prefix="${MRL_NMT_EXPERIMENTS_FOLDER}"
checkpoints_prefix="${MRL_NMT_CHECKPOINTS_FOLDER}"
gpu="$CUDA_VISIBLE_DEVICES"

# Set up Conda environment
source /home/$(whoami)/miniconda3/etc/profile.d/conda.sh
conda activate $conda_env_name

# Experiment folder creation

### Create experiment folder & train/eval folders for default corpus
python scripts/create_experiment.py \
    --experiment-name $experiment_name \
    --references-file $references_file \
    --raw-data-folder $raw_data_folder \
    --bin-data-folder $bin_data_folder \
    --model-name $model_name \
    --experiments-prefix $experiments_prefix \
    --checkpoints-prefix $checkpoints_prefix

which python
conda info
pwd

# Train + eval using Guild
train () {
    echo "TRAINING"
    guild run nmt:train_transformer -y \
    experiment_name=$experiment_name \
    model_name=$model_name \
    src_lang=en tgt_lang=uz  \
    max_tokens=10000 batch_size=96 max_updates=1500000  \
    gpu_device="${gpu}" \
    validate_interval_updates=25000
    save_interval_updates=500000
}

evaluate () {
    echo "EVALUATION"
    guild run nmt:evaluate_transformer -y \
        experiment_name=$experiment_name \
        src_lang=en tgt_lang=uz \
        model_name=$model_name eval_name="eval_${model_name}" \
        references_clean_file=$references_file \
        remove_preprocessing_hypotheses=char \
        remove_preprocessing_references=char  \
        remove_preprocessing_source=char \
        remove_preprocessing_references_clean=none \
        detokenize_hypotheses=no \
        detokenize_references=no  \
        detokenize_source=no \
        detokenize_references_clean=no \
        gpu_device="${gpu}" mode="test"
}

train && evaluate
