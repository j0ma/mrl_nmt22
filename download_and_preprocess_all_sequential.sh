#!/usr/bin/env bash

set -euo pipefail

download_data () {
    local language=$1

    download_folder_stub="$HOME/datasets/mrl_nmt22"
    download_folder="${download_folder_stub}/${language}"
    case "$language"
    in
        cs) 
            corpora=("default-train" "wmt-late"  "wmt-early")
            download_cmd="download_mtdata"
            ;;
        de) 
            corpora=("default-train" "wmt-late"  "wmt-early")
            download_cmd="download_mtdata"
            ;;
        fi) 
            corpora=("default-train" "newstest-2018"  "newstest-2019")
            download_cmd="download_mtdata"
            ;;
        iu) 
            corpora=("hansard" "wmt20")
            download_cmd="download_mtdata"
            ;;
        ru) 
            corpora=("default-train" "wmt-18-20")
            download_cmd="download_mtdata"
            ;;
        et) 
            corpora=("default-train") 
            download_cmd="download_mtdata"
            ;;
        uz) 
            corpora=("default-train") 
            download_cmd="download_til"
            ;;
        tr) 
            corpora=("default-train") 
            download_cmd="download_til"
            ;;
        vi) 
            echo "INFO: Vietnamese PhoMT corpus must be downloaded manually." && exit ;;
        *) echo "Unsupported language: ${languge}" && exit ;;
    esac

    for corpus in "${corpora[@]}"
    do
        guild run "data:${download_cmd}" \
            -y --background \
            corpus_name="$corpus" \
            source_language=en \
            target_language="$language" \
            download_folder="$download_folder"
    done
}

languages=("cs" "de" "ru" "fi" "iu" "et" "tr" "uz" "vi")

#for language in "${languages[@]}" 
#do
    #download_data $language 
#done

# do the preprocessing per the yaml file
for yml_file in $(ls ./scripts/slurm/en_*/en*.yaml | rg -v "en_sp32k_iu" | rg -v "en_sp32k_uz")
do
    python scripts/text_processing/preprocess.py --yaml-config $yml_file
done

