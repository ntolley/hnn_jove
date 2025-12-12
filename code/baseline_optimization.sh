#!/bin/bash
#SBATCH -J extrinsic_permutations
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -t 20:00:00
#SBATCH --mem=200G
#SBATCH --account carney-sjones-condo
#SBATCH -p batch
#SBATCH --array=0-19

# Specify an output file
#SBATCH -o /users/ntolley/Jones_Lab/hnn_jove/data/baseline_optimization/job_out/extrinsic_permutations-%j.out
#SBATCH -e /users/ntolley/Jones_Lab/hnn_jove/data/baseline_optimization/job_out/extrinsic_permutations-%j.out

module load anaconda
source ~/.bashrc
conda activate jaxley2

echo "Starting job $SLURM_ARRAY_TASK_ID on $HOSTNAME"
python baseline_optimization.py $SLURM_ARRAY_TASK_ID

export OUTPATH="/users/ntolley/Jones_Lab/hnn_jove/data/baseline_optimization/job_out"
scontrol show job $SLURM_JOB_ID >> ${OUTPATH}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.stats
sacct -j $SLURM_JOB_ID --format=JobID,JobName,MaxRSS,Elapsed >> ${OUTPATH}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.stats