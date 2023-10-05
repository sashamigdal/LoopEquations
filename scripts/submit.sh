#!/bin/bash

#Export Parameters
NJOBS=1
export NCPUS=1

# Memory size, in GB:
export MEM_SIZE=4
export MU=1e-6
export T=1000

jobid=$(sbatch --cpus-per-task=${NCPUS} --mem=${MEM_SIZE}GB --parsable run.sh)
echo Submitted jobs ${jobid}

# sbatch -d afterok:$jobid plot.sh
#sbatch plot.sh
