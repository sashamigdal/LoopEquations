#!/bin/bash

#SBATCH --mail-type=ALL
#SBATCH --mail-user=am10485@nyu.edu
#SBATCH --cpus-per-task=32
#SBATCH --output=/scratch/am10485/slurm-logs/slurm-%A_%a.out
#SBATCH --time=00:30:00
#SBATCH --mem=100GB

echo PROJECT_DIR=${PROJECT_DIR}
cd ${PROJECT_DIR}

source /share/apps/NYUAD/miniconda/3-4.11.0/bin/activate py39

echo "Starting collecting script . . . "
# python -u VorticityCorrelation.py -M ${M} -EG E -C 0 --run 2 --compute GPU
cd CPP/cmake-build-release
./SamplesBinner ${PROJECT_DIR}/plots/VorticityCorr.${M}.GPU.${RUN_ID} 20
