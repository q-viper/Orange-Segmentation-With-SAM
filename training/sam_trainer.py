
import datetime
import os
from pathlib import Path
import cv2
import json
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from PIL import Image
import json
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pickle as pkl
import torch.nn.functional as F
from collections import defaultdict
import torch
from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything import SamPredictor, sam_model_registry
from torch.nn.functional import threshold, normalize
from statistics import mean

from torch.utils.data import Dataset, DataLoader
import os
import json
from PIL import Image

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

class SAMDataSet(Dataset):
    def __init__(self, kind=1, data_labels=None,
                 color_space="RGB"):
        """
        kind: 1 for training and 0 for validation.
        data_labels: dictionary containing labels.
        transform: what transformation to apply.
        color_space: rgb/hsv or any other.
        multiclass: if True, target will be N,H,W else, N,C,H,W
        """
        self.data_labels = data_labels
        self.color_space=color_space
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
        # image = image.resize((self.width, self.height))
        img_arr = np.array(image)
        
        
        # Create masks for each class label
        background = np.zeros(img_arr.shape[:2], dtype=np.uint8) + 255
        healthy = np.zeros(img_arr.shape[:2], dtype=np.uint8)
        infected = np.zeros(img_arr.shape[:2], dtype=np.uint8)

        masks_dict = {"Background": background, "Healthy": healthy, "Infected": infected}
        bb_dict={k:[] for k in masks_dict.keys()}

        bboxes = []
        individual_masks = []
        for label_name, points_list in self.curr_labels[img_name].items():
            mask = masks_dict.get(label_name)
            # Draw each mask on the blank mask
            for points in points_list:
                mask = np.zeros_like(mask)
                points = np.round(points).astype(int)
                mask = cv2.fillPoly(mask, [points], 255)
                bounding_rect = cv2.boundingRect(points)
                background = cv2.fillPoly(background, [points], 0)
                bb_dict[label_name].append(bounding_rect)
                x, y, w, h= bounding_rect
                bboxes.append([x, y, x+w, y+h])
                individual_masks.append(mask)

            masks_dict[label_name] = mask
        background = background

        masks = np.zeros((img_arr.shape[0], img_arr.shape[1], 3), dtype=np.uint8)
        masks[:, :, 0] = masks_dict["Background"]
        masks[:, :, 1] = masks_dict["Healthy"]
        masks[:, :, 2] = masks_dict["Infected"]

        
        inputs = {'input_boxes':bboxes}
        inputs['ground_truth_mask'] = 255-masks_dict['Background']
        inputs['img_name']=img_name
        inputs['orig_img'] = img_arr
        inputs['ground_truth_masks'] = np.array(individual_masks)

        

        # print(idx)
        return inputs

def dice_loss(predicted_mask, ground_truth_mask):
    """
    Calculate Dice loss between two boolean masks of shape (H, W).
    
    Args:
    predicted_mask (torch.Tensor): Predicted mask tensor of shape (H, W).
    ground_truth_mask (torch.Tensor): Ground truth mask tensor of shape (H, W).
    
    Returns:
    float: Dice loss value.
    """
    # Convert boolean masks to float tensors
    predicted_mask = predicted_mask.float()
    ground_truth_mask = ground_truth_mask.float()
    
    # Calculate intersection and union
    intersection = (predicted_mask * ground_truth_mask).sum(axis=0)
    union = predicted_mask.sum(axis=0) + ground_truth_mask.sum(axis=0)
    
    # Calculate Dice coefficient
    dice_coefficient = (2 * intersection + 1e-6) / (union + 1e-6)
    
    # Calculate Dice loss
    dice_loss_value = 1 - dice_coefficient
    
    return dice_loss_value.mean()

def iou_score(predicted_mask, ground_truth_mask):
    intersection = torch.logical_and(predicted_mask, ground_truth_mask).sum(axis=0)
    union = torch.logical_or(predicted_mask, ground_truth_mask).sum(axis=0)
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()

def transform_inputs(inputs, model):
  transformed_data = defaultdict(dict)
  image = inputs['orig_img']
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  transform = ResizeLongestSide(model.image_encoder.img_size)
  input_image = transform.apply_image(image)
  input_image_torch = torch.as_tensor(input_image, device=device)
  transformed_image = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

  input_image = model.preprocess(transformed_image)
  original_image_size = image.shape[:2]
  input_size = tuple(transformed_image.shape[-2:])


  transformed_data['image'] = input_image
  transformed_data['input_size'] = input_size

  transformed_data['original_image_size'] = original_image_size
  return transformed_data, transform

def sam_step(inputs, sam_model, loss_fn):
  bbox_coords = inputs['input_boxes']
  transformed_data, transform = transform_inputs(inputs, sam_model)

  input_image = transformed_data['image'].to(device)
  input_size = transformed_data['input_size']
  original_image_size = transformed_data['original_image_size']
  ground_truth_masks=inputs['ground_truth_masks']

  # No grad here as we don't want to optimise the encoders
  with torch.no_grad():
    image_embedding = sam_model.image_encoder(input_image)

    prompt_box = np.array(bbox_coords)
    box = transform.apply_boxes(prompt_box, original_image_size)
    box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
    # box_torch = box_torch[None, :]

    sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
  low_res_masks, iou_predictions = sam_model.mask_decoder(
    image_embeddings=image_embedding,
    image_pe=sam_model.prompt_encoder.get_dense_pe(),
    sparse_prompt_embeddings=sparse_embeddings,
    dense_prompt_embeddings=dense_embeddings,
    multimask_output=False,
  )

  upscaled_masks = sam_model.postprocess_masks(low_res_masks, input_size, original_image_size).to(device)
  binary_mask = normalize(threshold(upscaled_masks, 0.0, 0))

  gt_mask_resized = torch.from_numpy(np.resize(ground_truth_masks, (len(binary_mask), 1, 
                                                                    ground_truth_masks.shape[1], 
                                                                    ground_truth_masks.shape[2]))).to(device)
  gt_binary_mask = torch.as_tensor(gt_mask_resized > 0, dtype=torch.float32)
  loss = loss_fn(binary_mask, gt_binary_mask)
  iou = iou_score(binary_mask, gt_binary_mask)
  
  return loss, binary_mask, gt_binary_mask, iou



if __name__=="__main__":
    CONFIGS={
    "loss":'multiclass',
    "project":"Orange Segmentation",
    "activation":"softmax2d",
    "name":f"slurm_sam_{datetime.datetime.now().date()}",
    "train_size":0.8, #used for train/test split
    "epoch" : 150,
    "batch_size" : 1,
    "max_data_samples" : -100,
    "learning_rate" : 0.0001,
    "shuffle_seed" : 100,
    "description":"Let's try this settings first.",
    'data_dir':'/home/hpc/rlvl/rlvl102h/Orange-Dataset-Segmentation/data/Orange_Image_Data/Orange_Image_Data',
    'labels_dir':'/home/hpc/rlvl/rlvl102h/Orange-Dataset-Segmentation/data/project-4-at-2024-03-02-15-13-497758b6.json',
    'train_dir':'/home/hpc/rlvl/rlvl102h/Orange-Dataset-Segmentation/train_res',
    'model_path': '/home/hpc/rlvl/rlvl102h/train_res/sdefault_model.pth',
    'start_epoch':0,
    'optimizer_dir':'/home/hpc/rlvl/rlvl102h/Orange-Dataset-Segmentation/train_res/Orange Segmentation/slurm_sam_2024-07-30/optimizer.pth'#'/home/hpc/rlvl/rlvl102h/Orange-Dataset-Segmentation/train_res/Orange Segmentation/slurm_focal_2024-07-31/optimizer.pth'
    
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
    
    model_type = 'vit_b'
    checkpoint = '/home/hpc/rlvl/rlvl102h/Orange-Dataset-Segmentation/data/sam_weights/sam_vit_b_01ec64.pth'
    model = sam_model_registry[model_type](checkpoint=checkpoint)
    
    optimizer = torch.optim.Adam(model.mask_decoder.parameters(), lr=CONFIGS['learning_rate'])

    
    if Path(CONFIGS['model_path']).exists():
        model.load_state_dict(torch.load(CONFIGS['model_path']))
        print("Model loaded.")
    if Path(CONFIGS['optimizer_dir']).exists():
        optimizer.load_state_dict(torch.load(CONFIGS['optimizer_dir']))
        print("Optimizer loaded.")
    for name, param in model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad_(False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    data_labels = DataLabels(root_dir=CONFIGS['data_dir'], 
                         labels_file=CONFIGS['labels_dir'], 
                         split_ratio=CONFIGS['train_size'],
                        max_samples=CONFIGS['max_data_samples'],
                        shuffle_seed=CONFIGS['shuffle_seed'])

    train_dataset = SAMDataSet(kind=1, data_labels=data_labels)
    test_dataset = SAMDataSet(kind=0, data_labels=data_labels)
    # seg_loss = dice_loss
    seg_loss = torch.nn.MSELoss()
    logs = {'epoch':[], 'train_loss':[], 'val_loss':[], 'train_iou':[], 'val_iou':[]}

    best_iou = 0
    num_epochs = CONFIGS['epoch']-CONFIGS['start_epoch']
    loss_fn = dice_loss
    model.train()
    for epoch in range(CONFIGS['start_epoch'], CONFIGS['epoch']):
        item_losses = 0
        item_ious = 0
        batch_ious = []
        batch_losses = []
        random_indices = np.random.permutation(len(train_dataset))
        pbar = tqdm(desc=f"Train Epoch {epoch+1}/{num_epochs}", 
                    total=len(train_dataset)//CONFIGS['batch_size'])
        nitems=0
        for i, item in enumerate(random_indices):
            nitems+=1
            inputs = train_dataset[item]
            loss, binary_mask, gt_binary_mask, iou1 = sam_step(inputs, model, loss_fn)
            item_losses+=loss
            item_ious+=iou1
            
            if i%CONFIGS['batch_size']==0:            
                batch_loss = item_losses/CONFIGS['batch_size']
                batch_iou = item_ious/CONFIGS['batch_size']
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                item_losses = 0
                item_ious = 0
                nitems=0
                batch_losses.append(batch_loss.item())
                batch_ious.append(batch_iou)
                #pbar.update(1)
                #pbar.set_postfix({'loss': batch_loss.item(), 'iou': batch_iou})
        if nitems>0:
            batch_loss = item_losses/nitems
            batch_iou = item_ious/nitems
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            batch_losses.append(batch_loss.item())
            batch_ious.append(batch_iou)
            #pbar.update(1)
            #pbar.set_postfix({'loss': batch_loss.item(), 'iou': batch_iou})

        #pbar.close()
        logs['train_loss'].append(mean(batch_losses))
        logs['train_iou'].append(mean(batch_ious))
        # validation
        item_losses = []
        item_ious = []
        batch_ious = []
        batch_losses = []
        random_indices = np.random.permutation(len(test_dataset))
        pbar = tqdm(desc=f"Validation Epoch {epoch+1}/{num_epochs}", total=len(test_dataset)//CONFIGS['batch_size'])

        for item in random_indices:
            inputs = test_dataset[item]
            with torch.no_grad():
                loss, binary_mask, gt_binary_mask, iou1 = sam_step(inputs, model, loss_fn)

            item_losses.append(loss.item())
            item_ious.append(iou1)

            if len(item_losses)==CONFIGS['batch_size']:
                batch_loss = sum(item_losses)/len(item_losses)
                batch_iou = sum(item_ious)/len(item_ious)
                item_losses=[]
                item_ious=[]
                batch_losses.append(batch_loss)
                batch_ious.append(batch_iou)
                #pbar.update(1)
                #pbar.set_postfix({'loss': batch_loss, 'iou': batch_iou})
        if len(item_losses)>0:
            batch_loss = mean(item_losses)
            batch_iou = mean(item_ious)
            batch_losses.append(batch_loss)
            batch_ious.append(batch_iou)
            #pbar.update(1)
            #pbar.set_postfix({'loss': batch_loss, 'iou': batch_iou})

        #pbar.close()
        logs['val_loss'].append(mean(batch_losses))
        logs['val_iou'].append(mean(batch_ious))
        print(f"Epoch {epoch+1}/{num_epochs} Train Loss: {logs['train_loss'][-1]} Val Loss: {logs['val_loss'][-1]} Train IoU: {logs['train_iou'][-1]} Val IoU: {logs['val_iou'][-1]}")

        if logs['val_iou'][-1]>best_iou:
            best_model_path = run_dir / 'best_model.pth'
            best_iou = logs['val_iou'][-1]
            torch.save(model.state_dict(), best_model_path)
            print(f"Best Model Saved at {best_model_path}")
        logs['epoch'].append(epoch)
        torch.save(model.state_dict(), run_dir / 'last_model.pth')
        torch.save(optimizer.state_dict(), run_dir / 'last_optimizer.pth')
        
        pkl.dump(logs, open(run_dir / 'logs.pkl', 'wb'))



    print("Training finished.")

