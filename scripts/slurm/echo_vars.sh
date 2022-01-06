#!/usr/bin/env bash
#
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --ntasks=1
#SBATCH --job-name=echo_vars_slurm
#SBATCH --output=/scratch0/jonnesaleva/echo_vars_slurm.out
#SBATCH --account=guest
#SBATCH --partition=guest-compute
#SBATCH --qos=low
#SBATCH --export=TEST_ASDF,TEST_FDSA

# Set up Conda environment
source /home/$(whoami)/miniconda3/etc/profile.d/conda.sh
conda activate fairseq-py3.8

guild run slurm:echo_something msg=henlo -y
guild operations

opsnames=$(guild operations)

echo "Guild ops: ${opsnames}"

echo "TEST_ASDF = ${TEST_ASDF}"
echo "TEST_FDSA = ${TEST_FDSA}"

ls -l .
