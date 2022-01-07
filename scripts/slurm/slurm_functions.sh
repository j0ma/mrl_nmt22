activate_conda_env () {
    local conda_env_name=${1:-fairseq-py3.8}
    source /home/$(whoami)/miniconda3/etc/profile.d/conda.sh
    conda activate $conda_env_name
}

prep_train () {

    local experiment_name=$1
    local references_file=$2
    local raw_data_folder=$3
    local bin_data_folder=$4
    local model_name=$5
    local experiments_prefix=$6
    local checkpoints_prefix=$7

    python scripts/create_experiment.py \
        --experiment-name $experiment_name \
        --references-file $references_file \
        --raw-data-folder $raw_data_folder \
        --bin-data-folder $bin_data_folder \
        --model-name $model_name \
        --experiments-prefix $experiments_prefix \
        --checkpoints-prefix $checkpoints_prefix
}

prep_eval () {

    local experiment_name=$1
    local model_name=$2
    local eval_name=$3
    local eval_model_checkpoint=$4
    local raw_data_folder=$5
    local bin_data_folder=$6
    local references_clean_file=$7
    local experiments_prefix=$8
    local checkpoints_prefix=$9

    python scripts/create_experiment.py \
        --experiment-name $experiment_name \
        --model-name $model_name \
        --eval-name $eval_name \
        --eval-only \
        --eval-model-checkpoint $eval_model_checkpoint \
        --references-file $references_clean_file \
        --raw-data-folder $raw_data_folder \
        --bin-data-folder $bin_data_folder \
        --experiments-prefix $experiments_prefix \
        --checkpoints-prefix $checkpoints_prefix
}


train () {

	local experiment_name=$1
	local model_name=$2
	local src_lang=$3
	local tgt_lang=$4
	local max_tokens=$5
	local batch_size=$6
	local max_updates=$7
	local gpu_device=$8
	local validate_interval_updates=$9
	local save_interval_updates=${10}
	local encoder_embedding_dim=${11}
	local decoder_embedding_dim=${12}
	local lr=${13}
	local encoder_layers=${14}
	local encoder_attention_heads=${15}
	local encoder_hidden_size=${16}
	local decoder_layers=${17}
	local decoder_attention_heads=${18}
	local decoder_hidden_size=${19}
    local p_dropout=${20}

    guild run nmt:train_transformer -y \
        experiment_name=$experiment_name \
        model_name=$model_name \
        src_lang=$src_lang \
        tgt_lang=$tgt_lang \
        max_tokens=$max_tokens \
        batch_size=$batch_size \
        max_updates=$max_updates \
        gpu_device=$gpu_device \
        validate_interval_updates=$validate_interval_updates \
        save_interval_updates=$save_interval_updates \
        encoder_embedding_dim=$encoder_embedding_dim \
        decoder_embedding_dim=$decoder_embedding_dim \
        lr=$lr p_dropout=$p_dropout \
        encoder_layers=$encoder_layers \
        encoder_attention_heads=$encoder_attention_heads \
        encoder_hidden_size=$encoder_hidden_size \
        decoder_layers=$decoder_layers \
        decoder_attention_heads=$decoder_attention_heads \
        decoder_hidden_size=$decoder_hidden_size
    }

evaluate () {

	local experiment_name=$1
	local src_lang=$2
	local tgt_lang=$3
	local model_name=$4
	local references_clean_file=$5
	local remove_preprocessing_hypotheses=$6
	local remove_preprocessing_references=$7
	local remove_preprocessing_source=$8
	local remove_preprocessing_references_clean=$9
	local detokenize_hypotheses=${10}
	local detokenize_references=${11}
	local detokenize_source=${12}
	local detokenize_references_clean=${13}
	local gpu_device=${14}
    local mode=${15}
    local eval_name=${16}


    guild run nmt:evaluate_transformer -y \
        experiment_name=$experiment_name \
        src_lang=$src_lang tgt_lang=$tgt_lang \
        model_name=$model_name eval_name=$eval_name \
        references_clean_file=$references_clean_file \
        remove_preprocessing_hypotheses=$remove_preprocessing_hypotheses \
        remove_preprocessing_references=$remove_preprocessing_references  \
        remove_preprocessing_source=$remove_preprocessing_source \
        remove_preprocessing_references_clean=$remove_preprocessing_references_clean \
        detokenize_hypotheses=$detokenize_hypotheses \
        detokenize_references=$detokenize_references \
        detokenize_source=$detokenize_source \
        detokenize_references_clean=$detokenize_references_clean \
        gpu_device="${gpu}" mode=${mode}
}
