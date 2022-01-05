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

echo "TEST_ASDF = ${TEST_ASDF}"
echo "TEST_FDSA = ${TEST_FDSA}"
