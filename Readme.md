# Task: Orange segmentation with SAM
I did this project while taking a project Representation Learning in Summer Semester at FAU Erlangen-Nuremberg.

## Data
Source: [Kaggle](https://www.kaggle.com/datasets/durgapokharel/orange-infection-mask-dataset)

## Training
### U-Net
U-net training is done by using [`segmentation-models-pytorch`](https://github.com/qubvel-org/segmentation_models.pytorch) package. And files [`training/trainer_binary_dice.py`](training/trainer_binary_dice.py) and [`training/trainer_binary_focal.py`](training/trainer_binary_focal.py) are used to train.

### SAM
SAM installation: `pip install git+https://github.com/facebookresearch/segment-anything.git`.

Followed [Encord.com](https://encord.com/blog/learn-how-to-fine-tune-the-segment-anything-model-sam/) for finetuning the model and file [`training/sam_trainer.py`](training/sam_trainer.py) is the one used to train.

Notebooks were used to get visualizations, zero-shot evaluation and some other experiments. `jobs/*.slurm` were used to run a slurm job on HPC.

### SAM 2
I also did inference with SAM2 but was not a easy task. It can be found on [notebooks\SAM_zeroshot_expt.ipynb](notebooks\SAM_zeroshot_expt.ipynb).



## Results
Please follow [assets/PRL_PPT.pdf](assets/PRL_PPT.pdf) for presentation. And [assets/PRL_Report.pdf](assets/PRL_Report.pdf) for report.