#!/usr/bin/env bash

# Utility script to download Newscrawl data using GNU parallel

set -euo pipefail

[ "$#" -lt "2" ] && echo "Error: Must specify language and output folder/file!" && exit 1

LANGUAGE=$1
OUTPUT_FOLDER=$2
DEFAULT_FNAME="${LANGUAGE}_mono"
OUTPUT_FILE="${OUTPUT_FOLDER}/${3:-$DEFAULT_FNAME}"

get_newscrawl_urls () {

    local language=$1
    temp_file=$(mktemp)

    wget --quiet -k -l 0 -O $temp_file \
        "http://data.statmt.org/news-crawl/${language}/" \
        2>/dev/null

    rg -o "https://data\.statmt\.org/news-crawl.*\.gz\b" \
        < $temp_file | sed "s/\">.*$//g"
}

main () {

    local language=$1
    local output_file=$2

    mkdir -vp $(dirname $output_file)

    get_newscrawl_urls "${language}" | parallel -t wget -O - {} "|" gunzip -c ">>" "${output_file}"

    echo "Lines written:"
    wc -l "${output_file}" | sed ':a;s/\B[0-9]\{3\}\>/,&/;ta'

}

main $LANGUAGE $OUTPUT_FILE
