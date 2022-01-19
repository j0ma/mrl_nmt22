#!/usr/bin/env bash

# Utility script to download Newscrawl data using GNU parallel

set -exuo pipefail

[ "$#" -lt "1" ] && echo "Error: Must specify output folder!" && exit 1

output_folder=$1

available_languages=(
    "Czech" "German" "Finnish"
    "Russian" "Turkish" "Estonian"
    "Vietnamese"    
)

main () {

   local output_folder=$1

   staging=$(mktemp -d) 
   git clone https://github.com/sigmorphon/2019.git $staging/sigmorphon

   cd $staging/sigmorphon/task2
    
   for lang in "${available_languages[@]}"
   do
       output_folder_lang="${output_folder}/${language}/sigmorphon"
       mkdir -p $output_folder_lang
       ls ./*$lang*/*.conllu | parallel -t python -i {} "|" jq -c ">>" "${output_folder_lang}/${language}_sigmorphon.jsonl"
   done

}

main $output_folder
