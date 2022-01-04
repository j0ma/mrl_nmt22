#!/usr/bin/env bash
set -euo pipefail

# Parse arguments
experiment_name=$1
model_name=$2
references_default_train=$3
raw_data_folder_default_train=$4
bin_data_folder_default_train=$5
references_newstest2019=$6
raw_data_folder_newstest2019=$7
bin_data_folder_newstest2019=$8

usage () {
    echo """
    Arguments: 
    experiment_name
    model_name
    references_default_train
    raw_data_folder_default_train
    bin_data_folder_default_train
    references_newstest2019
    raw_data_folder_newstest2019
    bin_data_folder_newstest2019
    """
}

[ "$#" -lt 8 ] && usage && exit 1

# Set up Conda environment
conda_env_name=torch-rtx-3090
source /home/$(whoami)/miniconda3/etc/profile.d/conda.sh
conda activate $conda_env_name

# Experiment folder creation

### Create experiment folder & train/eval folders for default corpus
python scripts/create_experiment.py \
    --experiment-name $experiment_name \
    --references-file $references_default_train \
    --raw-data-folder $raw_data_folder_default_train \
    --bin-data-folder $bin_data_folder_default_train \
    --model-name $model_name

### Create eval folder for newstest2019
python scripts/create_experiment.py \
    --experiment-name $experiment_name \
    --eval-only --eval-name newstest-2019 \
    --eval-model-checkpoint "./experiments/$experiment_name/eval/eval_${model_name}/checkpoint" \
    --references-file $references_newstest2019 \
    --raw-data-folder $raw_data_folder_newstest2019 \
    --bin-data-folder $bin_data_folder_newstest2019 \

# Train + eval using Guild
guild run nmt:train_transformer -y \
    experiment_name=$experiment_name \
    model_name=$model_name \
    src_lang=en tgt_lang=fi  \
    max_tokens=4096 max_updates=100  \
    gpu_device="0,1" validate_interval_updates=50

guild run nmt:evaluate_transformer -y \
    experiment_name=$experiment_name \
    src_lang=en tgt_lang=fi \
    model_name=$model_name eval_name=newstest-2019 \
    references_clean_file=$references_newstest2019 \
    remove_preprocessing_hypotheses=char \
    remove_preprocessing_references=char  \
    remove_preprocessing_source=char \
    remove_preprocessing_references_clean=none \
    gpu_device="0,1" mode=train
