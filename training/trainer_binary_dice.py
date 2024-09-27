
import datetime
import os
from pathlib import Path
import cv2
import json
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from segmentation_models_pytorch.utils.base import Metric
import segmentation_models_pytorch.utils.functional as smp_f
import torch.nn.functional as F
import segmentation_models_pytorch.utils as smp_utils
import segmentation_models_pytorch as smp
import json

def compare_results(dataset, model, epoch, device, num_examples=2):
    model.eval()
    with torch.no_grad():
        fig, axes = plt.subplots(num_examples*2, 4, figsize=(12, num_examples * 4))
        fig.suptitle(f'Test Images and Predicted Masks - Epoch {epoch + 1}')

        for i in range(0, 
                       num_examples*2,2):
            images, masks = dataset[i]

            images = images.unsqueeze(0).to(device)
            masks = masks.unsqueeze(0).to(device)
            
            
            # Forward pass through the model
            outputs = model(images)
            one_hot_target = F.one_hot(masks, outputs.size(1))  # Shape: (N, H, W, C)
            masks = one_hot_target.permute(0, 3, 1, 2).contiguous()  # Shape: (N, C, H, W)

            # Convert tensors to numpy arrays for plotting
            image_np = images.squeeze(0).cpu().permute(1, 2, 0).detach().numpy()
            true_mask_np = masks.squeeze(0).cpu().detach().numpy()
            pred_mask_np = 255*(outputs.squeeze(0).cpu().numpy()>0.5)

            # Plot original image, true mask, and predicted mask in the first row
            axes[i, 0].imshow(image_np.astype(np.uint8))
            axes[i, 0].set_title('Original Image')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(true_mask_np[0], cmap='gray')  # Plotting the first mask
            axes[i, 1].set_title('BG')
            axes[i, 1].axis('off')

            axes[i, 2].imshow(true_mask_np[1], cmap='gray')  # Plotting the first mask
            axes[i, 2].set_title('Healthy')
            axes[i, 2].axis('off')

            axes[i, 3].imshow(true_mask_np[2], cmap='gray')  # Plotting the second mask
            axes[i, 3].set_title('Infected')
            axes[i, 3].axis('off')


            # Plot predicted mask in the second row
            axes[i+1, 0].imshow(image_np.astype(np.uint8))
            axes[i+1, 0].set_title('Original Image')
            axes[i+1, 0].axis('off')        
            axes[i + 1, 1].imshow(pred_mask_np[0], cmap='gray')  # Plotting the predicted mask
            axes[i + 1, 1].set_title('Predicted BG')
            axes[i + 1, 1].axis('off')
            axes[i + 1, 2].imshow(pred_mask_np[1], cmap='gray')  # Plotting the predicted mask
            axes[i + 1, 2].set_title('Predicted Healthy')
            axes[i + 1, 2].axis('off')
            axes[i + 1, 3].imshow(pred_mask_np[2], cmap='gray')  # Plotting the predicted mask
            axes[i + 1, 3].set_title('Predicted Infected')
            axes[i + 1, 3].axis('off')

        plt.tight_layout()
        plt.show()
    return fig

def val_iou_classwise(test_loader, device, model, iou_obj):
    model.eval()
    acc_ious = 0
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model.to(device)(images)
            acc_ious+=iou_obj.forward_each_class(outputs, masks)
    acc_ious = acc_ious/len(test_loader)
    return [float(iou) for iou in acc_ious]

def write_checkpoint_logs(run_dir, start_epoch, current_epoch, logs, model):
    log_dir = run_dir/f'{start_epoch}_{current_epoch}_logs.json'
    # Write the dictionary to a JSON file
    with open(log_dir, 'w') as json_file:
        json.dump(logs, json_file, indent=4)
    
    
    if (run_dir/f'{start_epoch}_{current_epoch-1}_logs.json').exists() and current_epoch>(start_epoch+1):
        (run_dir/f'{start_epoch}_{current_epoch-1}_logs.json').unlink()
    
    model_dir = run_dir/f'checkpoint_{current_epoch}.pth'
    torch.save(model, str(model_dir))
    
    if (run_dir/f'checkpoint_{current_epoch-1}.pth').exists():
        (run_dir/f'checkpoint_{current_epoch-1}.pth').unlink()

def log_best(run_dir, start_epoch, current_epoch, model):
    torch.save(model, str(run_dir/f'{start_epoch}_{current_epoch}_best_model.pth'))
    
    if (run_dir/f'{start_epoch}_{current_epoch-1}_best_model.pth').exists() and     current_epoch>(start_epoch+1):
        (run_dir/f'{start_epoch}_{current_epoch-1}_best_model.pth').unlink()
    
class DataLabels:
    def __init__(self, root_dir, labels_file, shuffle_seed=42, split_ratio=0.8, max_samples=-1):
        """
        root_dir: From where all dataset is being read.
        labels_file: Labels file path.
        shuffle_seed: to handle random split.
        split_ratio: first half train, second will be test
        max_samples: max number of samples to take. -1 means all.
        """
        self.root_dir = root_dir
        self.labels_file = labels_file
        self.images_dir = root_dir
        self.shuffle_seed = shuffle_seed
        self.split_ratio = split_ratio
        self.max_samples = max_samples
        
        self.labels, self.label_counts = self.load_labels(labels_file)
        np.random.seed(shuffle_seed)
        fnames = list(self.labels.keys())
        np.random.shuffle(fnames)
        self.idxs = list(range(len(fnames)))
        self.labels = {k: self.labels[k] for k in fnames}
        self.train_idx = self.idxs[:int(len(self.labels) * split_ratio)]
        self.valid_idx = self.idxs[len(self.train_idx):]
        self.train_labels = {fnames[k]: self.labels[fnames[k]] for k in self.train_idx}
        self.valid_labels = {fnames[k]: self.labels[fnames[k]] for k in self.valid_idx}

    def load_labels(self, labels_dir):
        # Load segmentation labels from JSON file
        with open(labels_dir, 'r') as f:
            segmentation_data = json.load(f)

        final_labels = {}
        label_counts = {}
        self.label_names = []
        img_cnts = 0

        for sm in segmentation_data:
            img_name = "IMG" + sm['image'].split('-IMG')[1]
            img_dict = {}

            if sm.get('label') is None:
                continue

            for labels in sm['label']:
                width = labels['original_width']
                height = labels['original_height']
                points = np.array(labels['points']).reshape(-1, 2)
                points[:, 0] = points[:, 0] * width / 100
                points[:, 1] = points[:, 1] * height / 100

                label_names = labels['polygonlabels']

                for label_name in label_names:
                    self.label_names.append(label_name)
                    if img_dict.get(label_name) is None:
                        img_dict[label_name] = []
                    img_dict[label_name].append(points)

            final_labels[img_name] = img_dict
            label_counts[img_name] = {k: len(v) for k, v in img_dict.items()}
            img_cnts += 1
            
            if img_cnts >= self.max_samples and self.max_samples > 0:
                break

        self.label_names = sorted(list(set(self.label_names)))
        return final_labels, label_counts

class DataSet(Dataset):
    def __init__(self, kind=1, data_labels=None, transform=None, 
                 color_space="RGB", multiclass=True):
        """
        kind: 1 for training and 0 for validation.
        data_labels: dictionary containing labels.
        transform: what transformation to apply.
        color_space: rgb/hsv or any other.
        multiclass: if True, target will be N,H,W else, N,C,H,W
        """
        self.data_labels = data_labels
        self.color_space=color_space
        self.transform = transform
        self.multiclass = multiclass
        # 1 for train, 0 for valid
        if kind == 1:
            self.curr_labels = self.data_labels.train_labels
        else:
            self.curr_labels = self.data_labels.valid_labels

    def __len__(self):
        return len(self.curr_labels)

    def __getitem__(self, idx):
        img_name = list(self.curr_labels.keys())[idx]
        img_path = os.path.join(self.data_labels.images_dir, img_name)

        # Load image
        image = Image.open(img_path).convert(self.color_space)
        image = np.array(image)

        # Create masks for each class label
        background = np.zeros(image.shape[:2], dtype=np.uint8) + 255
        healthy = np.zeros(image.shape[:2], dtype=np.uint8)
        infected = np.zeros(image.shape[:2], dtype=np.uint8)
        
        masks_dict = {"Background": background, "Healthy": healthy, "Infected": infected}
        
        for label_name, points_list in self.curr_labels[img_name].items():
            mask = masks_dict.get(label_name)

            # Draw each mask on the blank mask
            for points in points_list:
                points = np.round(points).astype(int)
                mask = cv2.fillPoly(mask, [points], 255)
                background = cv2.fillPoly(background, [points], 0)

            masks_dict[label_name] = mask
        background = background
        
        masks = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        masks[:, :, 0] = masks_dict["Background"]
        masks[:, :, 1] = masks_dict["Healthy"]
        masks[:, :, 2] = masks_dict["Infected"]

        if self.transform:
            augmented = self.transform(image=image, masks=masks)
            image = augmented['image']
            masks = augmented['masks']
        if self.multiclass:
            # for multiclass we need mask in format N, H, W. Each value in it represents class value.
            # class should start from 0.
            nmasks = np.zeros((masks.shape[0], masks.shape[1]))
            nmasks[(masks[:, :, 1] + masks[:, :, 2])>0] = 1
            # nmasks[masks[:, :, 2]>0] = 2
            masks=torch.tensor(nmasks).long()
        return image.float(), masks

def get_train_transforms(height, width):
    return A.Compose([
        A.HorizontalFlip(p=0.5),  # Apply horizontal flip with 50% probability
        A.VerticalFlip(p=0.5),    # Apply vertical flip with 50% probability
        # A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
        A.GaussNoise(p=0.2),
        A.Perspective(p=0.5),
        A.Resize(height, width),  # Resize image at the end
        ToTensorV2()
], additional_targets={'masks': 'mask'})

def get_test_transforms(height, width):
    return A.Compose([
        A.Resize(height, width),
        ToTensorV2()
    ], additional_targets={'masks': 'mask'})


class IoU(Metric):
    __name__ = "iou_score"
    def __init__(
        self, eps=1e-7, threshold=0.5, ignore_channels=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):        
        one_hot_target = F.one_hot(y_gt, y_pr.size(1))  # Shape: (N, H, W, C)
        one_hot_target = one_hot_target.permute(0, 3, 1, 2).contiguous()  # Shape: (N, C, H, W)       
        return smp_f.iou(
            y_pr,
            one_hot_target,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )
    
    def forward_each_class(self, y_pr, y_gt):
        one_hot_target = F.one_hot(y_gt, y_pr.size(1))  # Shape: (N, H, W, C)
        one_hot_target = one_hot_target.permute(0, 3, 1, 2).contiguous()  # Shape: (N, C, H, W)
        
        ious=[]
        for c in range(one_hot_target.size(1)):
            ious.append(smp_f.iou(
                y_pr[:,c,:,:],
                one_hot_target[:,c,:,:],
                eps=self.eps,
                threshold=self.threshold,
                ignore_channels=self.ignore_channels,
            ))
        return torch.tensor(ious)
    
if __name__=="__main__":
    CONFIGS={
    "loss":'multiclass',
    "project":"Orange Segmentation",
    "activation":"softmax2d",
    "name":f"slurm_binary_dice_{datetime.datetime.now().date()}",
    "height" : 416+32 ,
    "train_size":0.8, #used for train/test split
    "epoch" : 150,
    "width" : 192+32,
    "encoder" : "resnet18",
    "batch_size" : 256,
     "learning_rate" : 0.00001,
    "shuffle_seed" : 100,
    "description":"Let's try this settings first.",
    'data_dir':'/home/hpc/rlvl/rlvl102h/Orange-Dataset-Segmentation/data/Orange_Image_Data/Orange_Image_Data',
    'labels_dir':'/home/hpc/rlvl/rlvl102h/Orange-Dataset-Segmentation/data/project-4-at-2024-03-02-15-13-497758b6.json',
    'train_dir':'/home/hpc/rlvl/rlvl102h/Orange-Dataset-Segmentation/train_res',
    'model_path': '/home/hpc/rlvl/rlvl102h/train_res/uresnet18_binary.pth',
    'start_epoch':0,
    'optimizer_dir':'/home/hpc/rlvl/rlvl102h/Orange-Dataset-Segmentation/train_res/Orange Segmentation/-07-30/optimizer.pth'#'/home/hpc/rlvl/rlvl102h/Orange-Dataset-Segmentation/train_res/Orange Segmentation/slurm_focal_2024-07-31/optimizer.pth'
    
    }


    train_dir = Path(CONFIGS['train_dir'])
    project_dir=train_dir/Path(CONFIGS['project'])
    run_dir = project_dir/Path(CONFIGS['name'])

    if not train_dir.exists():
        train_dir.mkdir(parents=True)

    if not project_dir.exists():
        project_dir.mkdir(parents=True)
        
    if not run_dir.exists():
        run_dir.mkdir(parents=True)
    with open(str(run_dir/"config.json"), 'w') as json_file:
        json.dump(CONFIGS, json_file, indent=4)
    
    # dataset and dataloader
    data_labels = DataLabels(root_dir=CONFIGS['data_dir'], 
                            labels_file=CONFIGS['labels_dir'], 
                            split_ratio=CONFIGS['train_size'],
                            max_samples=-100,
                            shuffle_seed=CONFIGS['shuffle_seed'])

    train_transform = get_train_transforms(height=CONFIGS['height'], width=CONFIGS['width'])
    test_transform = get_test_transforms(height=CONFIGS['height'], width=CONFIGS['width'])

    train_dataset = DataSet(kind=1, data_labels=data_labels, transform=train_transform)
    test_dataset = DataSet(kind=0, data_labels=data_labels, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=CONFIGS['batch_size'], shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=CONFIGS['batch_size'], shuffle=True, num_workers=4)

    model_path =CONFIGS['model_path']
    if Path(model_path).exists():
        model = torch.load(model_path)
        print(f"Loaded {model_path}")
    DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {DEVICE}")

    # criterion = smp.losses.FocalLoss(mode='multiclass')
    # criterion.__name__ = "focal_loss"
    criterion = smp.losses.DiceLoss(mode='multiclass')
    criterion.__name__ = "dice_loss"
    metrics = [
        IoU(threshold=0.5),
    ]
    optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=0.0001),
    ])
    optimizer_path = CONFIGS['optimizer_dir']
    if Path(optimizer_path).exists():
        optimizer=torch.load(str(optimizer_path))
        print(f"Loaded {optimizer_path}")

    
    train_epoch = smp.utils.train.TrainEpoch(
        model, 
        loss=criterion, 
        metrics=metrics, 
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model, 
        loss=criterion, 
        metrics=metrics, 
        device=DEVICE,
        verbose=True,
    )

    max_score = 0
    show_prediction_every=10
    all_logs = {}
    for i in range(0, CONFIGS['epoch']):
        i+=CONFIGS['start_epoch']
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(test_loader)
        valid_logs['bg_iou'], valid_logs['orange_iou']=val_iou_classwise(test_loader, DEVICE, model, metrics[0])
        wb_logs = {f'valid_{key}':value for key,value in valid_logs.items()}
        wb_logs.update({f'train_{key}':value for key,value in train_logs.items()})
        # wandb.log(wb_logs)
        all_logs[i]=wb_logs
        print(f"Valid IoUs [BG, Orange]: {(valid_logs['bg_iou'], valid_logs['orange_iou'])}")
        
        # if i%show_prediction_every==0:
        #     compare_results(test_dataset, model, i, DEVICE)
        
        # do something (save model, change lr, etc.)
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            log_best(run_dir,CONFIGS['start_epoch'], i, model)
            print('Model saved!')
        
        write_checkpoint_logs(run_dir, CONFIGS['start_epoch'], i, all_logs, model)
        torch.save(optimizer,str(run_dir/'optimizer.pth'))
        if i==CONFIGS['epoch']:
            break
        

    print("Training finished.")

