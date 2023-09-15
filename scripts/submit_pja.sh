#!/bin/bash
# Submits jobs to Parallel Job Array

T=1000000
T_PER_JOB=100
NJOBS=$(( ${T} / ${T_PER_JOB} ))
COMMANDS=list_of_commands.txt

if test -f "${COMMANDS}"; then
    rm ${COMMANDS}
fi

for (( i=1; i<=${NJOBS}; i++ ))
do
   echo "run.sh $i" >> ${COMMANDS}
done

# rm ${COMMANDS} 2> /dev/null

#Export Parameters
export MU=1e-7
export T=${T_PER_JOB}

slurm_parallel_ja_submit.sh -t 02:30:00 ${COMMANDS}
