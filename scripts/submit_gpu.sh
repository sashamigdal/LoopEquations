#!/bin/bash

# 2x4, 6x2, 10x1 (jobs x GPUs) are OK.

#Export Parameters
NJOBS=480
export NCPUS=8
NGPUS=1

# Memory size, in GB:
export MEM_SIZE=20
export M=2000000
export T=524288
export RUN_ID=0

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"

jobid=$(sbatch -p nvidia -q nvidia-xxl --array=1-${NJOBS} --cpus-per-task=${NCPUS} --mem=${MEM_SIZE}GB --gres=gpu:a100:${NGPUS} -t 10-00:00:00 --parsable run_gpu.sh)
echo Submitted GPU jobs ${jobid}

#jobid2=$(sbatch --dependency=afterok:${jobid} --parsable plot.sh)
#echo Submitted binning job ${jobid2}
