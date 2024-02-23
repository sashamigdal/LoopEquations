#!/bin/bash

# Usage: submit_binner.sh <M> <run id>

NJOBS=480
export M=$1
export T=524288
export RUN_ID=$2

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"

if false; then
# 1. Keep only DS & indices
if jobid=$(sbatch --wait --array=1-${NJOBS} --parsable index.sh) ; then
  echo "Index job array ${jobid} completed."
else
  echo "FAIL"
  exit 1
fi


# 2. Distributed merge sort
NFILES=${NJOBS}
while [ ${NFILES} -gt 1 ]
do
  if jobid=$(sbatch --wait --array=1-$((NFILES/2)) --parsable merge.sh) ; then
    echo "Merge job array ${jobid} completed. Processed ${NFILES} files."
  else
    echo "FAIL"
    exit 1
  fi

  if [ $((NFILES%2)) -eq 1 ] ; then
    mv ${PROJECT_DIR}/plots/VorticityCorr.${M}.GPU.${RUN_ID}/Fdata.E.${T}.${NFILES}.idx ${PROJECT_DIR}/plots/VorticityCorr.${M}.GPU.${RUN_ID}/Fdata.E.${T}.$((NFILES/2+1)).idx.out
  fi

  rm ${PROJECT_DIR}/plots/VorticityCorr.${M}.GPU.${RUN_ID}/Fdata.E.${T}.*.idx
  rename .idx.out .idx ${PROJECT_DIR}/plots/VorticityCorr.${M}.GPU.${RUN_ID}/Fdata.E.${T}.*.idx.out
  NFILES=$((NFILES/2+NFILES%2))
done

# 3. Inverse permutation and split
if jobid=$(sbatch --wait --parsable inverse.sh ${NJOBS}) ; then
  echo "Inverse & split job ${jobid} completed."
else
  echo "FAIL"
  exit 1
fi
fi

# 4. Bin each file
if jobid=$(sbatch --wait --array=1-${NJOBS} --parsable bin.sh ${NJOBS}) ; then
  echo "Binning job array ${jobid} completed."
else
  echo "FAIL"
  exit 1
fi

# 5. Collect bins
