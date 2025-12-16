#!/bin/bash
#SBATCH --partition=Odyssey # Selected partition
#SBATCH --job-name=... # Name for the job
#SBATCH --gres=gpu:a100:1 # Resources asked
#SBATCH --nodelist=sl-mee-br-207 # Node where the job will be executed
#SBATCH --cpus-per-gpu=12    # 12 CPUs for each GPU
#SBATCH --output=job/job_%j.log # %j for jobid

export HOME='/Odyssey/private/ochapron/'
source /Odyssey/private/ochapron/start_conda.sh
conda activate fdv
srun python GS_SSH_year_warped.py