#!/bin/bash -l
#SBATCH --gres=gpu:a40:1
#SBATCH --time=01:00:00
#SBATCH --job-name=testjob_llava
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module load python
conda activate llava

python3 video_tutorial.py
