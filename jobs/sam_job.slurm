#!/bin/bash -l
#
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --partition=rtx3080
#SBATCH --time=24:00:00
#SBATCH --export=NONE
#SBATCH --output=sam_job.out

unset SLURM_EXPORT_ENV

module load python
conda activate pytorch

python3 /home/hpc/rlvl/rlvl102h/Orange-Dataset-Segmentation/sam_trainer.py