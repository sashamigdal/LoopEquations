#!/bin/bash

# Usage: submit_plot.sh <M> <run id>

export M=$1
export RUN_ID=$2

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"

jobid=$(sbatch --parsable plot.sh)
echo Submitted binning job ${jobid}
