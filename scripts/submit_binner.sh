#!/bin/bash

# Usage: submit_binner.sh <M> <run id>

NJOBS=480
export M=$1
export T=524288
export RUN_ID=$2

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"

# 1. Keep only DS & indices
jobid=$(sbatch --array=1-${NJOBS} --parsable index.sh)
echo Submitted index job ${jobid}
# 2. Distributed merge sort
# 3. Reverse permutation
# 4. Split indices
# 5. Bin each file
# 6. Collect bins
