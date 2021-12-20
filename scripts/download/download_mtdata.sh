#!/usr/bin/env bash

# Downloads data for English - <lang> using MTData
# Note: meant to be run from the base dir of the repo (or by Guild)

set -euo pipefail

mtdata_download () {

    local -r src_lang="$1"
    shift
    local -r tgt_lang="$1"
    shift
    local -r destination="$1"
    shift

    # go to directory with recipes
    pushd examples/data_download

    # actually perform download
    mtdata get-recipe \
        -ri "mrl-${src_lang}-${tgt_lang}" \
        -o "${destination}"
    
    # go back
    popd
}

src_lang=$1
tgt_lang=$2
data_folder=$3

destination="${data_folder}/${src_lang}-${tgt_lang}/download"

mkdir -vp $destination

mtdata_download $src_lang $tgt_lang $destination
