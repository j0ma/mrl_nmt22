#!/usr/bin/env bash

# Utility script to download Newscrawl data using GNU parallel

set -euo pipefail

[ "$#" -lt "1" ] && echo "Error: Must specify output folder!" && exit 1

OUTPUT_FOLDER=$1
LANGUAGE=iu
DEFAULT_FNAME="${LANGUAGE}_mono"
OUTPUT_FILE="${OUTPUT_FOLDER}/${3:-$DEFAULT_FNAME}"

main () {

    local language=$1
    local output_file=$2

    mkdir -vp $(dirname $output_file)

    wget --quiet -O - http://web-language-models.s3.amazonaws.com/wmt20/deduped/iu.xz | xz -d | tqdm > "${output_file}"

    echo "Lines written:"
    wc -l "${output_file}" | sed ':a;s/\B[0-9]\{3\}\>/,&/;ta'

}

main $LANGUAGE $OUTPUT_FILE
