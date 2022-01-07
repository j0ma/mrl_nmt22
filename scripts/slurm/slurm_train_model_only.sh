#!/usr/bin/env bash
#
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --account=guest
#SBATCH --partition=guest-gpu
#SBATCH --qos=low-gpu
#SBATCH --export=ALL

# Parse arguments from environment 
batch_size="${MRL_NMT_BATCH_SIZE}"
bin_data_folder="${MRL_NMT_BIN_DATA_FOLDER}"
checkpoints_prefix="${MRL_NMT_CHECKPOINTS_FOLDER}"
conda_env_name="${MRL_NMT_ENV_NAME}"
decoder_attention_heads="${MRL_DECODER_ATTENTION_HEADS}"
decoder_embedding_dim="${MRL_DECODER_EMBEDDING_DIM}"
decoder_hidden_size="${MRL_NMT_DECODER_HIDDEN_SIZE}"
decoder_layers="${MRL_NMT_DECODER_LAYERS}"
detokenize_hypotheses="${MRL_NMT_DETOKENIZE_HYPOTHESES}"
detokenize_references_clean="${MRL_NMT_DETOKENIZE_REFERENCES_CLEAN}"
detokenize_references="${MRL_NMT_DETOKENIZE_REFERENCES}"
detokenize_source="${MRL_NMT_DETOKENIZE_SOURCE}"
encoder_attention_heads="${MRL_ENCODER_ATTENTION_HEADS}"
encoder_embedding_dim="${MRL_ENCODER_EMBEDDING_DIM}"
encoder_hidden_size="${MRL_NMT_ENCODER_HIDDEN_SIZE}"
encoder_layers="${MRL_NMT_ENCODER_LAYERS}"
eval_name="${MRL_NMT_EVAL_NAME}"
experiment_name="${MRL_NMT_EXPERIMENT_NAME}"
experiments_prefix="${MRL_NMT_EXPERIMENTS_FOLDER}"
gpu="$CUDA_VISIBLE_DEVICES"
lr="${MRL_NMT_LEARNING_RATE}"
max_tokens="${MRL_NMT_MAX_TOKENS}"
max_updates="${MRL_NMT_MAX_UPDATES}"
model_name="${MRL_NMT_MODEL_NAME}"
mode="${MRL_NMT_EVAL_MODE}"
raw_data_folder="${MRL_NMT_RAW_DATA_FOLDER}"
references_clean_file="${MRL_NMT_REFERENCES_FILE}"
remove_preprocessing_hypotheses="${MRL_NMT_REMOVE_PREPROCESSING_HYPOTHESES}"
remove_preprocessing_references_clean="${MRL_NMT_REMOVE_PREPROCESSING_REFERENCES_CLEAN}"
remove_preprocessing_references="${MRL_NMT_REMOVE_PREPROCESSING_REFERENCES}"
remove_preprocessing_source="${MRL_NMT_REMOVE_PREPROCESSING_SOURCE}"
save_interval_updates="${MRL_NMT_SAVE_INTERVAL_UPDATES}"
src_lang="${MRL_NMT_SRC_LANG}"
tgt_lang="${MRL_NMT_TGT_LANG}"
validate_interval_updates="${MRL_NMT_VALIDATE_INTERVAL_UPDATES}"
p_dropout="${MRL_NMT_P_DROPOUT}"

source scripts/slurm/slurm_functions.sh

# Set up Conda environment
activate_conda_env $conda_env_name

_train () {
    train \
        $experiment_name \
        $model_name \
        $src_lang \
        $tgt_lang \
        $max_tokens \
        $batch_size \
        $max_updates \
        $gpu \
        $validate_interval_updates \
        $save_interval_updates \
        $encoder_embedding_dim \
        $decoder_embedding_dim \
        $lr \
        $encoder_layers \
        $encoder_attention_heads \
        $encoder_hidden_size \
        $decoder_layers \
        $decoder_attention_heads \
        $decoder_hidden_size \
        $p_dropout
}


_train 
