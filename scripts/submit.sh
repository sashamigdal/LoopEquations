#!/bin/bash

#Export Parameters
NJOBS=100
export NCPUS=128

# Memory size, in GB:
export MEM_SIZE=240
export M=100000000
export T=100000

jobid=$(sbatch --array=1-${NJOBS} --cpus-per-task=${NCPUS} --mem=${MEM_SIZE}GB --parsable run.sh)

SBATCH -p preempt
SBATCH -C dalmac

sbatch -d afterok:$jobid plot.sh
