#!/bin/bash

#SBATCH --mail-type=ALL
#SBATCH --mail-user=am10485@nyu.edu
#SBATCH --cpus-per-task=1
#SBATCH --output=/scratch/am10485/slurm-logs/index-%A_%a.out
#SBATCH --time=1-00:00:00
#SBATCH --mem=2GB

echo PROJECT_DIR=${PROJECT_DIR}
cd ${PROJECT_DIR}

echo "Starting collecting . . . "
cd CPP/cmake-build-release
./SamplesBinner --collect ${PROJECT_DIR}/plots/VorticityCorr.${M}.GPU.${RUN_ID}
