#!/bin/bash

# 2x4, 6x2, 10x1 (jobs x GPUs) are OK.

#Export Parameters
NJOBS=10
export NCPUS=8
NGPUS=2

# Memory size, in GB:
export MEM_SIZE=16
export M=250000000
export T=524288
export RUN_ID=1

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"

jobid=$(sbatch -p nvidia -q nvidia-xxl --array=1-${NJOBS} --cpus-per-task=${NCPUS} --mem=${MEM_SIZE}GB --gres=gpu:a100:${NGPUS} -t 3-00:00:00 --parsable run_gpu.sh)
echo Submitted GPU jobs ${jobid}

jobid2=$(sbatch --dependency=afterok:${jobid} --parsable plot.sh)
echo Submitted binning job ${jobid2}
