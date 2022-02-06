#!/usr/bin/env bash

# Utility script to download Newscrawl data using GNU parallel

set -euo pipefail

[ "$#" -lt "2" ] && echo "Error: Must specify language and output folder/file!" && exit 1

LANGUAGE=$1
OUTPUT_FOLDER=$2
DEFAULT_FNAME="${LANGUAGE}_mono"
OUTPUT_FILE="${OUTPUT_FOLDER}/${DEFAULT_FNAME}"
CONV_TO_UTF=${3:-yes}
DEFAULT_ENC=${4:-ISO-8859-1}
SUBSAMPLE=${5:-yes}
SUBSAMPLE_SIZE=${6:-5000000}

get_newscrawl_urls () {

    local language=$1
    temp_file=$(mktemp)

    wget --quiet -k -l 0 -O $temp_file \
        "http://data.statmt.org/news-crawl/${language}/" \
        2>/dev/null

    rg -o "https://data\.statmt\.org/news-crawl.*\.gz\b" \
        < $temp_file | sed "s/\">.*$//g"
}
convert_to_utf8 () {
    local orig_output_file=$1
    local utf8_output_file=$2
    local orig_enc=$3
    iconv -f "${orig_enc}" -t UTF-8 $orig_output_file > $utf8_output_file
    rm -v "${orig_output_file}" && mv -v "${utf8_output_file}" "${orig_output_file}"
}

main () {

    local language=$1
    local output_file=$2

    local iso_to_utf=$3
    local orig_enc=$4
    
    local subsample=$5
    local subsample_size=$6

    mkdir -vp $(dirname $output_file)

    get_newscrawl_urls "${language}" | parallel -t wget -O - {} "|" gunzip -c ">>" "${output_file}"

    if [ "${subsample}"="yes" ]
    then
        local subsampled_output_file="${output_file}.sub"
        shuf -n ${subsample_size} < $output_file > $subsampled_output_file
        local output_file=${subsampled_output_file}
    fi

    if [ "${iso_to_utf}"="yes" ]
    then
        local utf8_output_file="${output_file}.utf8"
        convert_to_utf8 $output_file $utf8_output_file $orig_enc
    fi

    echo "Lines written:"
    wc -l "${output_file}" | sed ':a;s/\B[0-9]\{3\}\>/,&/;ta'

}

main \
    $LANGUAGE \
    $OUTPUT_FILE \
    $CONV_TO_UTF \
    $DEFAULT_ENC \
    $SUBSAMPLE \
    $SUBSAMPLE_SIZE
