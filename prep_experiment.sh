#!/usr/bin/env bash
set -euo pipefail

# Creates a custom folder for an experiment and 
# symlinks the relevant folders (checkpoint, data, data-bin)

EXPERIMENT_NAME="${1}"
RAW_DATA_FOLDER="${2}"
BIN_DATA_FOLDER="${3}"
MODEL_NAME="${4}"
WORKING_DIR=$(pwd)

echo "Preparing folders for experiment: '${EXPERIMENT_NAME}'"

# create experiment folder
EXPERIMENT_FOLDER="${WORKING_DIR}/experiments/${EXPERIMENT_NAME}"
mkdir -p "${EXPERIMENT_FOLDER}"


# symlink data folders
echo "Symlinking data folders..."
ln -vs "${RAW_DATA_FOLDER}" "${EXPERIMENT_FOLDER}/raw_data"
ln -vs "${BIN_DATA_FOLDER}" "${EXPERIMENT_FOLDER}/binarized_data"

# create path for checkpoint folder
DATE_SLUG=$(date -u +"%Y-%m-%d-%H%M")

# no need to save these since guild will save them
CHECKPOINT_FOLDER="${WORKING_DIR}/checkpoints/${EXPERIMENT_NAME}-${DATE_SLUG}"

# create checkpoint folder
echo "Creating folder for checkpoints..."
mkdir -p --verbose "${CHECKPOINT_FOLDER}"

# symlink checkpoint folder in experiment folder
echo "Symlinking checkpoint folder..."
MODEL_FOLDER="${EXPERIMENT_FOLDER}/${MODEL_NAME}"
mkdir -p --verbose $MODEL_FOLDER
ln -vs "${CHECKPOINT_FOLDER}" "${MODEL_FOLDER}/checkpoints"
