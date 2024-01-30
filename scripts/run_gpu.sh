#!/bin/bash

#SBATCH --output=/scratch/am10485/slurm-logs/run_gpu-%A_%a_%N_%J.out

echo SLURM_ARRAY_JOB_ID = ${SLURM_ARRAY_JOB_ID}
echo SLURM_ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}
echo SLURM_CPUS_PER_TASK = ${SLURM_CPUS_PER_TASK}
echo PROJECT_DIR = ${PROJECT_DIR}

cd ${PROJECT_DIR}

source /share/apps/NYUAD/miniconda/3-4.11.0/bin/activate py39

python -u VorticityCorrelation.py -M ${M} -T ${T} -EG E -C ${SLURM_ARRAY_TASK_ID} --compute GPU -NLAM 0 --run ${RUN_ID}
