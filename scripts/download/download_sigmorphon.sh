#!/usr/bin/env bash

# Utility script to download Newscrawl data using GNU parallel

set -exuo pipefail
declare -A available_languages

[ "$#" -lt "1" ] && echo "Error: Must specify output folder!" && exit 1

output_folder=$1

conllu_to_json=$(pwd)/scripts/download/conllu_to_json.py

available_languages=(
    [cs]="Czech" [de]="German" [fi]="Finnish"
    [ru]="Russian" [tr]="Turkish" [et]="Estonian"
    [vi]="Vietnamese"    
)

main () {

   local output_folder=$1

   staging=$(mktemp -d) 
   git clone https://github.com/sigmorphon/2019.git $staging/sigmorphon

   cd $staging/sigmorphon/task2
    
   for lang_code in "${!available_languages[@]}"
   do
       lang="${available_languages[$lang_code]}"
       output_folder_lang="${output_folder}/${lang_code}/sigmorphon"
       mkdir -p $output_folder_lang
       for split in "train" "dev" "test"
       do
           ls ./*$lang*/*.conllu \
               | rg -v "covered" | rg "${split}" \
               | parallel -t python $conllu_to_json -i {} "|" jq -c ">>" "${output_folder_lang}/${lang_code}_sigmorphon_${split}.jsonl"
       done
   done || rm -rf $staging

   rm -rf $staging
}

main $output_folder
