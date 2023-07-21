#!/bin/bash

#SBATCH --mail-type=ALL
#SBATCH --mail-user=am10485@nyu.edu
#SBATCH --cpus-per-task=32
#SBATCH --output=/scratch/am10485/slurm-logs/slurm-%A_%a.out
#SBATCH --time=7-00:00:00
#SBATCH --mem=100GB  

#echo ${SLURM_ARRAY_JOB_ID}

#/tmpdata
#Removing old slurm files
#find slurm/. -type f ! -name "slurm-${SLURM_ARRAY_JOB_ID}*" -exec rm {} +

#activate conda environment
#module load anaconda3/2020.07
#source /share/apps/NYUAD/miniconda/3-4.11.0/bin/activate
#activate
#CPU=48
#NODE=$SLURM_ARRAY_TASK_ID

NODE=0
CPU=${SLURM_CPUS_PER_TASK}

cd /home/am10485/LoopEquations

/home/am10485/LoopEquations/scripts/python  \
    -u VorticityCorrelation.py -M $M -T $T -CPU $CPU -C $NODE

