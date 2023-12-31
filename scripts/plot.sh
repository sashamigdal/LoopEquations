#!/bin/bash

#SBATCH --mail-type=ALL
#SBATCH --mail-user=am10485@nyu.edu
#SBATCH --cpus-per-task=1
#SBATCH --output=/scratch/am10485/slurm-logs/slurm-%A_%a.out
#SBATCH --time=00:30:00
#SBATCH --mem=100GB

PROJECT_DIR=/scratch/${USER}/LoopEquations
M=200000000

echo PROJECT_DIR = ${PROJECT_DIR}

cd ${PROJECT_DIR}

source /share/apps/NYUAD/miniconda/3-4.11.0/bin/activate py39

echo "Starting collecting script on node . . . "
# python -u VorticityCorrelation.py -M ${M} -EG E -C 0 --run 2 --compute GPU
cd CPP/cmake-build-release
./SamplesBinner /scratch/am10485/LoopEquations/plots/VorticityCorr.${M}.GPU.0 1000000
