#!/bin/bash

#SBATCH --time=00:20:00
#SBATCH -p preempt

PROJECT_DIR=/scratch/${USER}/LoopEquations
JOB_ID=$1

echo SLURM_ARRAY_JOB_ID = ${SLURM_ARRAY_JOB_ID}
echo SLURM_ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}
echo SLURM_CPUS_PER_TASK = ${SLURM_CPUS_PER_TASK}
echo JOB_ID = ${JOB_ID}
echo PROJECT_DIR = ${PROJECT_DIR}

cd ${PROJECT_DIR}

source /share/apps/NYUAD/miniconda/3-4.11.0/bin/activate py39

# python -u VorticityCorrelation.py -Mu ${MU} -EG E -T ${T} -CPU 1 -C ${JOB_ID} -R0 0.0 -R1 0.01 --serial -STP 1000000
python -u VorticityCorrelation.py -CPU 1 -NLAM 0 -Mu 1e-7 --serial
