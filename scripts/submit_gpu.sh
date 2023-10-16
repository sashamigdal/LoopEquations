#!/bin/bash

# 2x4, 6x2, 10x1 (jobs x GPUs) are OK.

#Export Parameters
NJOBS=1
export NCPUS=12
NGPUS=1

# Memory size, in GB:
export MEM_SIZE=16
export MU=1e-7
export T=1000
PROJECT_DIR=/scratch/${USER}/LoopEquations
COMMANDS=list_of_commands.txt

#if test -f "${COMMANDS}"; then
#    rm ${COMMANDS}
#fi

# echo "${PROJECT_DIR}/scripts/run_gpu.sh" >> ${COMMANDS}
# slurm_parallel_ja_submit.sh -N 1 -t 00:20:00 ${COMMANDS}

jobid=$(sbatch --array=1-${NJOBS} --cpus-per-task=${NCPUS} --mem=${MEM_SIZE}GB  -p nvidia --gres=gpu:${NGPUS} -t 00:30:00 --parsable run_gpu.sh)
echo Submitted jobs ${jobid}
