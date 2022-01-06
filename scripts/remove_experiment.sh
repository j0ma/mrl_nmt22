#!/usr/bin/env bash

set -euo pipefail

# Script to remove an experiment. 
# Meant to be executed by Guild.

experiment_name=$1

rm -vrf {experiments,checkpoints}/*${experiment_name}*
