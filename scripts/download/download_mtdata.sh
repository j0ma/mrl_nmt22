#!/usr/bin/env bash

# Downloads data for English - <lang> using MTData
# Note: meant to be run from the base dir of the repo (or by Guild)

set -euo pipefail

mtdata_download () {

    local -r corpus_name="$1"
    shift
    local -r src_lang="$1"
    shift
    local -r tgt_lang="$1"
    shift
    local -r destination="$1"
    shift

    # go to directory with recipes
    pushd config/data_download

    # actually perform download
    mtdata get-recipe \
        -ri "mrl-${src_lang}-${tgt_lang}-${corpus_name}" \
        -o "${destination}"
    
    # go back
    popd
}

corpus_name=$1
src_lang=$2
tgt_lang=$3
data_folder=$4

destination="${data_folder}/${src_lang}-${tgt_lang}/${corpus_name}/download"

mkdir -vp $destination

mtdata_download $corpus_name $src_lang $tgt_lang $destination
