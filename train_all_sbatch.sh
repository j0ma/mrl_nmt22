#!/usr/bin/env bash

base_path="scripts/slurm/"

folders=(
    en_sp4k_uz_sp4k
    en_sp32k_vi_sp32k
    en_sp32k_uz_sp32k
    en_sp32k_tr_sp32k
    en_sp32k_iu_sp32k
    en_sp32k_fi_sp32k
    en_sp32k_et_sp32k
    en_sp32k_de_sp32k
    en_sp1k_iu_sp1k
    #en_sp32k_ru_sp32k
    #en_sp32k_cs_sp32k
)

for folder in "${folders[@]}"
do
    cfg_file=$(ls ${base_path}/${folder}/_cfg* | head -n 1)
    echo $cfg_file
    source $cfg_file
    job_name=$(echo $folder | cut -d_ -f3,4 | tr _ -) 
    echo $job_name
    sbatch \
        --mail-user=jonnesaleva@brandeis.edu \
        --mail-type=ALL \
        --gres=gpu:V100:5 \
        --export=ALL \
        --job-name=$job_name\
            ${base_path}/slurm_train_evaluate_new_model.sh 
done

