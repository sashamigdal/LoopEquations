#!/bin/bash

#SBATCH --mail-type=ALL
#SBATCH --mail-user=am10485@nyu.e
#SBATCH --cpus-per-task=32
#SBATCH --output=/scratch/am10485/slurm-logs/slurm-%A_%a.out
#SBATCH --time=1-00:00:00
#SBATCH --mem=100GB

echo ${SLURM_ARRAY_JOB_ID}


NODE=$SLURM_ARRAY_TASK_ID
CPU=${SLURM_CPUS_PER_TASK}

cd /home/am10485/LoopEquations

/home/am10485/LoopEquations/scripts/python \
    -u VorticityCorrelation.py -M $M -T $T -CPU $CPU -C $NODE
