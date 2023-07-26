#!/bin/bash

#SBATCH --mail-type=ALL
#SBATCH --mail-user=am10485@nyu.e
#SBATCH --output=/scratch/am10485/slurm-logs/slurm-%A_%a.out
#SBATCH --time=1-00:00:00

echo SLURM_ARRAY_JOB_ID = ${SLURM_ARRAY_JOB_ID}
echo SLURM_ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}
echo SLURM_CPUS_PER_TASK = ${SLURM_CPUS_PER_TASK}

NODE=$SLURM_ARRAY_TASK_ID

PROJECT_DIR=/scratch/${USER}/LoopEquations

echo PROJECT_DIR = ${PROJECT_DIR}

cd ${PROJECT_DIR}

source /share/apps/NYUAD/miniconda/3-4.11.0/bin/activate
conda activate py39

echo "Starting script on node ${NODE} . . . "
# python -u VorticityCorrelation.py -M ${M} -T ${T} -CPU ${SLURM_CPUS_PER_TASK} -C ${NODE}

python -u VorticityCorrelation.py -M ${M} -T ${T} -CPU 80 -C ${NODE}
