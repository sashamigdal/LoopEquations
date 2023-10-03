#!/bin/bash

#Export Parameters
NJOBS=1
export NCPUS=12

# Memory size, in GB:
export MEM_SIZE=16
export MU=1e-7
export T=1000


jobid=$(sbatch --cpus-per-task=${NCPUS} --mem=${MEM_SIZE}GB  -p nvidia --gres=gpu:1 --parsable run_gpu.sh)
echo Submitted jobs ${jobid}

