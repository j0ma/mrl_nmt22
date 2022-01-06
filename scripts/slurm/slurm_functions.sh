activate_conda_env () {
    local conda_env_name=${1:-fairseq-py3.8}
    source /home/$(whoami)/miniconda3/etc/profile.d/conda.sh
    conda activate $conda_env_name
}

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


train () {
    guild run nmt:train_transformer -y \
        experiment_name=$experiment_name \
        model_name=$model_name \
        src_lang=en tgt_lang=uz  \
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
        src_lang=en tgt_lang=uz \
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
