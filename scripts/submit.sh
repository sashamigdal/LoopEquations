#!/bin/bash

#Export Parameters
NJOBS=10
export NCPUS=128

# Memory size, in GB:
export MEM_SIZE=480
export M=50000000
export T=1000

jobid=$(sbatch --array=1-${NJOBS} --cpus-per-task=${NCPUS} --mem=${MEM_SIZE}GB --parsable run.sh)

sbatch -d afterok:$jobid plot.sh
