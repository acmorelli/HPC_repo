#! /bin/bash
#SBATCH -p q_student
#SBATCH -N 20                 # start on 20 nodes
#SBATCH --cpu-freq=High       # set CPU frequency to maximum
#SBATCH --time=15:00           # job will run for 15 minutes (maximum)

# Path of the executable
EXEC_PATH=./build/allgather_merge
# Output folder
OUTFOLDER=./Results-N${SLURM_NNODES}-${SLURM_JOB_ID}


mkdir -p $OUTFOLDER
echo "Job ${SLURM_JOB_ID} started at $(date)" >> $OUTFOLDER/slurm-${SLURM_JOB_ID}.log

# in case your code is very slow, modify the following for loop to perform with only one ntasks, e.g., for ntasks in 32.
for ntasks in 1 10 32; do
  # Base name for output files in this loop iteration
  BASE_OUTNAME=${OUTFOLDER}/${SLURM_NNODES}x${ntasks}_${SLURM_JOB_ID}

  echo "Job: ${SLURM_JOB_ID} - running with $SLURM_NNODES Nodes x $ntasks PPN" > ${BASE_OUTNAME}.out

  srun --ntasks-per-node=$ntasks $EXEC_PATH > ${BASE_OUTNAME}.out 2> ${BASE_OUTNAME}.err

done

echo "Job ${SLURM_JOB_ID} completed at $(date)" >> $OUTFOLDER/slurm-${SLURM_JOB_ID}.log
