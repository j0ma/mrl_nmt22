#!/usr/bin/env bash
set -euo pipefail

src_lang=$1
tgt_lang=$2
data_folder=$3
data_bin_folder=$4

default_n_cpus=$(nproc)
n_workers=${5:-$default_n_cpus}
#gpu_id=${5:-0}

train_prefix="${data_folder}/${src_lang}-${tgt_lang}.train"
dev_prefix="${data_folder}/${src_lang}-${tgt_lang}.dev"

# test set disabled for now
#test_prefix="${data_folder}/${src_lang}-${tgt_lang}.test"

#CUDA_VISIBLE_DEVICES=${gpu_id} 
fairseq-preprocess \
    --source-lang "${src_lang}" \
    --target-lang "${tgt_lang}" \
    --trainpref "${train_prefix}" \
    --validpref "${dev_prefix}" \
    --destdir "${data_bin_folder}" \
    --cpu --workers "${n_workers}"

    # test sets disabled for now
    #--testpref "${test_prefix}" \
