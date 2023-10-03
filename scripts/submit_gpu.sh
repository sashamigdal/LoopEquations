#!/bin/bash

#Export Parameters
NJOBS=100
export NCPUS=128

# Memory size, in GB:
export MEM_SIZE=240
export MU=1e-7
export T=1000

jobid=$(sbatch --array=1-${NJOBS} --cpus-per-task=${NCPUS} --mem=${MEM_SIZE}GB --parsable run.sh)
echo Submitted jobs ${jobid}
