# Task: Orange segmentation with SAM
I did this project while taking a project Representation Learning in Summer Semester at FAU Erlangen Nuremberg.

## Data
Source: [Kaggle](https://www.kaggle.com/datasets/durgapokharel/orange-infection-mask-dataset)

## Training
### U-Net
U-net training is done by using `segmentation-models-pytorch` package. And files `training/trainer_binary_dice.py` and `training/trainer_binary_focal.py` are used to train.

### SAM
Followed [Encord.com](https://encord.com/blog/learn-how-to-fine-tune-the-segment-anything-model-sam/) for finetuning the model and file `training/sam_trainer.py` is the one used to train.

Notebooks were used to get visualizations, zero-shot evaluation and some other experiments. `jobs/*.slurm` were used to run a slurm job on HPC.