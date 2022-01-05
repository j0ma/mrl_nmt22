#!/usr/bin/env bash
#
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --job-name=en_char_fi_char_newstest2019
#SBATCH --output=/scratch0/jonnesaleva/en_char_uz_char_til.out
#SBATCH --account=guest
#SBATCH --partition=guest-gpu
#SBATCH --qos=low-gpu
#SBATCH --export=ALL

set -euo pipefail

# Script that bundles together experiment creation, training and evaluation 

# Parse arguments from environment (passed by Guild)
experiment_name="${MRL_NMT_EXPERIMENT_NAME}"
model_name="${MRL_NMT_MODEL_NAME}"
references_file="${MRL_NMT_REFERENCES_FILE}"
raw_data_folder="${MRL_NMT_RAW_DATA_FOLDER}"
bin_data_folder="${MRL_NMT_BIN_DATA_FOLDER}"
gpu="$CUDA_VISIBLE_DEVICES"

usage () {
    echo """
    Arguments: 
    experiment_name
    model_name
    references_file
    raw_data_folder
    bin_data_folder
    """
}

[ "$#" -lt 5 ] && usage && exit 1

# Set up Conda environment
source /home/$(whoami)/miniconda3/etc/profile.d/conda.sh
conda activate $conda_env_name

# Experiment folder creation

### Create experiment folder & train/eval folders for default corpus
python scripts/create_experiment.py \
    --experiment-name $experiment_name \
    --references-file $references \
    --raw-data-folder $raw_data_folder \
    --bin-data-folder $bin_data_folder \
    --model-name $model_name

# Train + eval using Guild
guild run nmt:train_transformer -y \
    experiment_name=$experiment_name \
    model_name=$model_name \
    src_lang=en tgt_lang=uz  \
    max_tokens=10000 batch_size=96 max_updates=1500000  \
    gpu_device="${gpu}" \
    validate_interval_updates=25000
    save_interval_updates=500000

guild run nmt:evaluate_transformer -y \
    experiment_name=$experiment_name \
    src_lang=en tgt_lang=uz \
    model_name=$model_name eval_name=eval-$model_name \
    references_clean_file=$references \
    remove_preprocessing_hypotheses=char \
    remove_preprocessing_references=char  \
    remove_preprocessing_source=char \
    remove_preprocessing_references_clean=none \
    detokenize_hypotheses=no \
    detokenize_references=no  \
    detokenize_source=no \
    detokenize_references_clean=no \
    gpu_device="${gpu}" mode=test
