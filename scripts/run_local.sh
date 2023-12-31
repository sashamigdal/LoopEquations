#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=am10485@nyu.edu
#SBATCH --cpus-per-task=48
#SBATCH --output=~/slurm/slurm-%A_%a.out
#SBATCH --time=7-00:00:00
#SBATCH --mem=120GB

# echo ${SLURM_ARRAY_JOB_ID}

#/tmpdata
#Removing old slurm files
# find ~/slurm/. -type f ! -name "slurm-${SLURM_ARRAY_JOB_ID}*" -exec rm {} +

#activate conda environment
cd ..
source venv/bin/activate
# module load anaconda3/2020.07
#source /share/apps/NYUAD/miniconda/3-4.11.0/bin/activate
# activate
PYTHON=/opt/anaconda2/bin/python
NODE=1
#NODE=0
${PYTHON} -u VorticityCorrelation.py -M $M -T $T -CPU $CPU -C $NODE
