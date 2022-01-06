#!/usr/bin/env bash
#
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --job-name=tr32
#SBATCH --output=/scratch0/jonnesaleva/en_sp32k_tr_sp32k_til.out
#SBATCH --account=guest
#SBATCH --partition=guest-gpu
#SBATCH --qos=low-gpu
#SBATCH --export=ALL

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

echo $experiment_name
echo $model_name
echo $references_file
echo $raw_data_folder
echo $bin_data_folder
echo $conda_env_name
echo $experiments_prefix
echo $checkpoints_prefix
echo $gpu

# Set up Conda environment
source /home/$(whoami)/miniconda3/etc/profile.d/conda.sh
conda activate $conda_env_name

# Experiment folder creation

### Create experiment folder & train/eval folders for default corpus
prep () {
    python scripts/create_experiment.py \
        --experiment-name $experiment_name \
        --references-file $references_file \
        --raw-data-folder $raw_data_folder \
        --bin-data-folder $bin_data_folder \
        --model-name $model_name \
        --experiments-prefix $experiments_prefix \
        --checkpoints-prefix $checkpoints_prefix
}

# Train + eval using Guild

train () {
    guild run nmt:train_transformer -y \
        experiment_name=$experiment_name \
        model_name=$model_name \
        src_lang=en tgt_lang=tr  \
        max_tokens=10000 batch_size=96 max_updates=450000  \
        gpu_device="${gpu}" \
        validate_interval_updates=10000 \
        save_interval_updates=50000 \
        encoder_embedding_dim=512 \
        decoder_embedding_dim=512 \
        lr=0.0003 p_dropout=0.2 \
        encoder_layers=6 encoder_attention_heads=8 encoder_hidden_size=2048 \
        decoder_layers=6 decoder_attention_heads=8 decoder_hidden_size=2048 
}

evaluate () {
    guild run nmt:evaluate_transformer -y \
        experiment_name=$experiment_name \
        src_lang=en tgt_lang=tr \
        model_name=$model_name eval_name="eval_${model_name}" \
        references_clean_file=$references_file \
        remove_preprocessing_hypotheses=sentencepiece \
        remove_preprocessing_references=sentencepiece  \
        remove_preprocessing_source=sentencepiece \
        remove_preprocessing_references_clean=none \
        detokenize_hypotheses=no \
        detokenize_references=no  \
        detokenize_source=no \
        detokenize_references_clean=no \
        gpu_device="${gpu}" mode="test"
}

prep && train #&& evaluate
