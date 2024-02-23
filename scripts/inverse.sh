#!/bin/bash

#SBATCH --mail-type=ALL
#SBATCH --mail-user=am10485@nyu.edu
#SBATCH --cpus-per-task=1
#SBATCH --output=/scratch/am10485/slurm-logs/index-%A_%a.out
#SBATCH --time=01:00:00
#TODO: calculate mem reqs
#SBATCH --mem=64GB

echo PROJECT_DIR=${PROJECT_DIR}
cd ${PROJECT_DIR}

echo "Starting inverse & split . . . "
cd CPP/cmake-build-release
./SamplesBinner --inverse ${PROJECT_DIR}/plots/VorticityCorr.${M}.GPU.${RUN_ID}/Fdata.E.${T}.1.idx $1
