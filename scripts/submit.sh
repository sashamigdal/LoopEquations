#!/bin/bash

#Export Parameters
export M=50000000
export T=1000000

jobid=$(sbatch --array=1-1000 --parsable run.sh)

sbatch -d afterok:$jobid plot.sh
