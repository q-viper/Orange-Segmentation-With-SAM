#!/bin/bash -l
#
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=24:00:00
#SBATCH --export=NONE
#SBATCH --output=binary_focal_seg.out

unset SLURM_EXPORT_ENV

module load python
conda activate pytorch

python3 /home/hpc/rlvl/rlvl102h/Orange-Dataset-Segmentation/trainer_binary_focal.py