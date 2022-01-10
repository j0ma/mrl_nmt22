export MRL_DECODER_ATTENTION_HEADS=8
export MRL_DECODER_EMBEDDING_DIM=512
export MRL_ENCODER_ATTENTION_HEADS=8
export MRL_ENCODER_EMBEDDING_DIM=512
export MRL_NMT_BATCH_SIZE=128
export MRL_NMT_CHECKPOINTS_FOLDER=/home/jonnesaleva/mrl_nmt22/checkpoints
export MRL_NMT_DECODER_HIDDEN_SIZE=2048
export MRL_NMT_DECODER_LAYERS=6
export MRL_NMT_DETOKENIZE_HYPOTHESES=yes
export MRL_NMT_DETOKENIZE_REFERENCES_CLEAN=yes
export MRL_NMT_DETOKENIZE_REFERENCES=yes
export MRL_NMT_DETOKENIZE_SOURCE=yes
export MRL_NMT_ENCODER_HIDDEN_SIZE=2048
export MRL_NMT_ENCODER_LAYERS=6
export MRL_NMT_ENV_NAME=fairseq-py3.8
export MRL_NMT_EXPERIMENT_NAME=en_sp32k_fi_sp32k_slurm
export MRL_NMT_EXPERIMENTS_FOLDER=/home/jonnesaleva/mrl_nmt22/experiments
export MRL_NMT_LEARNING_RATE=0.0003
export MRL_NMT_MAX_TOKENS=8192
export MRL_NMT_MAX_UPDATES=500000
export MRL_NMT_MODEL_NAME=slurmformer_clipnorm0.1_500k
export MRL_NMT_P_DROPOUT=0.2
export MRL_NMT_REMOVE_PREPROCESSING_HYPOTHESES=sentencepiece
export MRL_NMT_REMOVE_PREPROCESSING_REFERENCES_CLEAN=none
export MRL_NMT_REMOVE_PREPROCESSING_REFERENCES=sentencepiece
export MRL_NMT_REMOVE_PREPROCESSING_SOURCE=sentencepiece
export MRL_NMT_SAVE_INTERVAL_UPDATES=50000
export MRL_NMT_SRC_LANG=en
export MRL_NMT_TGT_LANG=fi
export MRL_NMT_VALIDATE_INTERVAL_UPDATES=10000

# corpus: default-train
export MRL_NMT_BIN_DATA_FOLDER=/home/jonnesaleva/mrl_nmt22/data-bin/en-fi/en_sp32k_fi_sp32k/default-train/
export MRL_NMT_EVAL_MODE="dev"
export MRL_NMT_EVAL_MODEL_CHECKPOINT="/home/jonnesaleva/mrl_nmt22/experiments/en_sp32k_fi_sp32k_slurm/eval/eval_slurmformer_clipnorm0.1_500k/checkpoint"
export MRL_NMT_EVAL_NAME="eval_slurmformer_clipnorm0.1_500k"
export MRL_NMT_RAW_DATA_FOLDER=/home/jonnesaleva/datasets/mrl_nmt22/processed/en-fi/en_sp32k_fi_sp32k/default-train
export MRL_NMT_REFERENCES_FILE=/home/jonnesaleva/datasets/mrl_nmt22/processed/en-fi/en_sp32k_fi_sp32k/default-train/en-fi.dev.detok.fi

# uncomment to evaluate on newstest-2019
#export MRL_NMT_BIN_DATA_FOLDER=/home/jonnesaleva/mrl_nmt22/data-bin/en-fi/en_sp32k_fi_sp32k/newstest-2019/
#export MRL_NMT_EVAL_MODE="train"
#export MRL_NMT_EVAL_MODEL_CHECKPOINT="/home/jonnesaleva/mrl_nmt22/experiments/en_sp32k_fi_sp32k_slurm/eval/newstest-2019/checkpoint"
#export MRL_NMT_EVAL_NAME="newstest-2019"
#export MRL_NMT_RAW_DATA_FOLDER=/home/jonnesaleva/datasets/mrl_nmt22/processed/en-fi/en_sp32k_fi_sp32k/newstest-2019
#export MRL_NMT_REFERENCES_FILE=/home/jonnesaleva/datasets/mrl_nmt22/processed/en-fi/en_sp32k_fi_sp32k/newstest-2019/en-fi.train.detok.fi
