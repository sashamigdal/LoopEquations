#!/bin/bash

#SBATCH --mail-type=ALL
#SBATCH --mail-user=am10485@nyu.edu
#SBATCH --cpus-per-task=64
#SBATCH --output=/scratch/am10485/slurm-logs/slurm-%A_%a.out
#SBATCH --time=7-00:00:00
#SBATCH --mem=240GB

PROJECT_DIR=/scratch/${USER}/LoopEquations

echo PROJECT_DIR = ${PROJECT_DIR}

cd ${PROJECT_DIR}

source /share/apps/NYUAD/miniconda/3-4.11.0/bin/activate
conda activate py39

echo "Starting plotting script on node . . . "

python -u VorticityCorrelation.py -M ${M} -T ${T} -CPU ${SLURM_CPUS_PER_TASK} -C 0 -EG E
python -u VorticityCorrelation.py -M ${M} -T ${T} -CPU ${SLURM_CPUS_PER_TASK} -C 0 -EG G
