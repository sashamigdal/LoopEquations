#!/bin/bash

#Export Parameters
export M=10000000
export T=10000

jobid=$(sbatch --array=1-100 --parsable run.sh)

sbatch -d afterok:$jobid plot.sh