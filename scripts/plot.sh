#!/bin/bash

#SBATCH --mail-type=ALL
#SBATCH --mail-user=am10485@nyu.edu
#SBATCH --cpus-per-task=64
#SBATCH --output=/scratch/am10485/slurm-logs/slurm-%A_%a.out
#SBATCH --time=1-00:00:00
#SBATCH --mem=200GB

PROJECT_DIR=/scratch/${USER}/LoopEquations

echo PROJECT_DIR = ${PROJECT_DIR}

cd ${PROJECT_DIR}

source /share/apps/NYUAD/miniconda/3-4.11.0/bin/activate py39

echo "Starting plotting script on node . . . "
python -u VorticityCorrelation.py -Mu ${MU} -EG E -T ${T} -CPU 1 -C 0 -R0 0.0 -R1 0.01 --serial -STP 0
