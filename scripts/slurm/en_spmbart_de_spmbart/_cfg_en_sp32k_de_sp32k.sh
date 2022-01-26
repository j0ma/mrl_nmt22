export MRL_DECODER_ATTENTION_HEADS=8
export MRL_DECODER_EMBEDDING_DIM=512
export MRL_ENCODER_ATTENTION_HEADS=8
export MRL_ENCODER_EMBEDDING_DIM=512
export MRL_NMT_BATCH_SIZE=256 # larger batch size
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
export MRL_NMT_EXPERIMENT_NAME=en_sp32k_de_sp32k_slurm
export MRL_NMT_EXPERIMENTS_FOLDER=/home/jonnesaleva/mrl_nmt22/experiments
export MRL_NMT_LEARNING_RATE=0.001
export MRL_NMT_MAX_TOKENS=8192
export MRL_NMT_MAX_UPDATES=20000
export MRL_NMT_MODEL_NAME=slurmformer_clipnorm0.1_20k_lr0.001 
export MRL_NMT_PATIENCE=10  # longer patience
export MRL_NMT_P_DROPOUT=0.2
export MRL_NMT_REMOVE_PREPROCESSING_HYPOTHESES=sentencepiece
export MRL_NMT_REMOVE_PREPROCESSING_REFERENCES_CLEAN=none
export MRL_NMT_REMOVE_PREPROCESSING_REFERENCES=sentencepiece
export MRL_NMT_REMOVE_PREPROCESSING_SOURCE=sentencepiece
export MRL_NMT_SAVE_INTERVAL_UPDATES=50000
export MRL_NMT_SRC_LANG=en
export MRL_NMT_TGT_LANG=de
export MRL_NMT_VALIDATE_INTERVAL_UPDATES=500 

# corpus: default-train
#export MRL_NMT_BIN_DATA_FOLDER=/home/jonnesaleva/mrl_nmt22/data-bin/en-de/en_sp32k_de_sp32k/default-train/
#export MRL_NMT_EVAL_MODE="test"
#export MRL_NMT_EVAL_MODEL_CHECKPOINT="/home/jonnesaleva/mrl_nmt22/experiments/en_sp32k_de_sp32k_slurm/train/slurmformer_clipnorm0.1_20k_lr0.001/checkpoints/checkpoint_best.pt"
#export MRL_NMT_EVAL_NAME="newstest2020"
#export MRL_NMT_RAW_DATA_FOLDER=/home/jonnesaleva/datasets/mrl_nmt22/processed/en-de/en_sp32k_de_sp32k/default-train
#export MRL_NMT_REFERENCES_FILE=/home/jonnesaleva/datasets/mrl_nmt22/processed/en-de/en_sp32k_de_sp32k/default-train/en-de.test.detok.de

# uncomment to evaluate on WMT-early
#export MRL_NMT_BIN_DATA_FOLDER=/home/jonnesaleva/mrl_nmt22/data-bin/en-de/en_sp32k_de_sp32k/wmt-early/
#export MRL_NMT_EVAL_MODE="train"
#export MRL_NMT_EVAL_MODEL_CHECKPOINT="/home/jonnesaleva/mrl_nmt22/experiments/en_sp32k_de_sp32k_slurm/train/slurmformer_clipnorm0.1_20k_lr0.001/checkpoints/checkpoint_best.pt"
#export MRL_NMT_EVAL_NAME="wmt-early"
#export MRL_NMT_RAW_DATA_FOLDER=/home/jonnesaleva/datasets/mrl_nmt22/processed/en-de/en_sp32k_de_sp32k/wmt-early
#export MRL_NMT_REFERENCES_FILE=/home/jonnesaleva/datasets/mrl_nmt22/processed/en-de/en_sp32k_de_sp32k/wmt-early/en-de.train.detok.de

# uncomment to evaluate on WMT-late test, i.e. newstest18
export MRL_NMT_BIN_DATA_FOLDER=/home/jonnesaleva/mrl_nmt22/data-bin/en-de/en_sp32k_de_sp32k/wmt-late/
export MRL_NMT_EVAL_MODE="test"
export MRL_NMT_EVAL_MODEL_CHECKPOINT="/home/jonnesaleva/mrl_nmt22/experiments/en_sp32k_de_sp32k_slurm/train/slurmformer_clipnorm0.1_20k_lr0.001/checkpoints/checkpoint_best.pt"
export MRL_NMT_EVAL_NAME="newstest2018"
export MRL_NMT_RAW_DATA_FOLDER=/home/jonnesaleva/datasets/mrl_nmt22/processed/en-de/en_sp32k_de_sp32k/wmt-late
export MRL_NMT_REFERENCES_FILE=/home/jonnesaleva/datasets/mrl_nmt22/processed/en-de/en_sp32k_de_sp32k/wmt-late/en-de.test.detok.de
