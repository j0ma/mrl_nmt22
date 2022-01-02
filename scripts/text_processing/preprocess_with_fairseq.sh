#!/usr/bin/env bash
set -euo pipefail

src_lang=$1
tgt_lang=$2
data_folder=$3
data_bin_folder=$4

src_dict=$5
tgt_dict=$6

# need these to be able to reuse token-to-ix mappings
if [ -z "${src_dict}" ]; then
    src_dict_flag=""
else
    src_dict_flag="--srcdict=${src_dict}"
fi
if [ -z "${tgt_dict}" ]; then
    tgt_dict_flag=""
else
    tgt_dict_flag="--tgtdict=${tgt_dict}"
fi

# disabled for now since never used
#default_n_cpus=$(nproc)
#n_workers=${5:-$default_n_cpus}
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
    --cpu --workers "$(nproc)" \
    ${src_dict_flag} ${tgt_dict_flag}

    # test sets disabled for now
    #--testpref "${test_prefix}" \
