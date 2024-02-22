#!/bin/bash

NJOBS=480
export M=2000000
export T=524288
export RUN_ID=0

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"

jobid=$(sbatch --array=1-${NJOBS} -t 01:00:00 --parsable sort.sh)
echo Submitted sorting job ${jobid}
