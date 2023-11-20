#!/bin/bash

PROJECT_DIR=/scratch/${USER}/LoopEquations
JOB_ID=$1

echo SLURM_ARRAY_JOB_ID = ${SLURM_ARRAY_JOB_ID}
echo SLURM_ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}
echo SLURM_CPUS_PER_TASK = ${SLURM_CPUS_PER_TASK}
echo JOB_ID = ${JOB_ID}
echo PROJECT_DIR = ${PROJECT_DIR}

cd ${PROJECT_DIR}

source /share/apps/NYUAD/miniconda/3-4.11.0/bin/activate py39

python -u VorticityCorrelation.py -M ${M} -T ${T} -EG E -C ${JOB_ID} --compute=CPU -CPU 1 -NLAM 0 --run=3 --serial
