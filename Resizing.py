"""
This code fine-tunes the mask decoder of the Segment Anything model (SAM) for soil erosion segmenttaion task. 
It uses the resizing approch of our paper to do so. It means that the code resizes each individual agricultural field to 256 * 256 pixels,
then use them as input to fine-tune the pretrained SAM model.
"""
# importi necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import glob
import wandb
from sklearn.model_selection import train_test_split
import torch
import rasterio
from PIL import Image
from sklearn.metrics import f1_score, precision_score, recall_score, jaccard_score, accuracy_score, confusion_matrix

# Set the device to cuda if it is available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Configuring Environment Variables: This can help avoid compatibility issues
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Uses the Wandb library to visualize the results as well as training and validation losses. If you don't need them or you want to use tensorboard insted, you can comment all the lines that have Wandb.
wandb.init(project="SAM_FineTune")
import pandas as pd

# Load tiff  images and masks
img_path=glob.glob("D:/Erosion_signs/training/images/*.tif")
mask_path=glob.glob("D:/Erosion_signs/training/masks/*.tif")
img_path.sort()
mask_path.sort()
large_images=[]
large_masks=[]
image_files = []
mask_files = []
for i, j in zip(img_path, mask_path):
    with rasterio.open(i) as src:
        # Read all rows, all columns, bands 1, 2, 3 (assuming RGB)
        img = src.read([1, 2, 3])
        # If the shape is not (bands, rows, cols), transpose it to match the expected (rows, cols, bands)
        if img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        large_images.append(img)
        image_files.append(i)

        
    with rasterio.open(j) as src:
        mask = src.read(1)  # Read the first band (assuming single-band mask)
        large_masks.append(mask)
        mask_files.append(j)

# Convert lists to numpy arrays (keeping dtype='object' to allow for different shapes if needed)
large_images = np.array(large_images, dtype='object')
large_masks = np.array(large_masks, dtype='object')

print(f"large_images shape: {large_images.shape}")
print(f"large_images[0] shape: {large_images[0].shape}")


# Check shapes of images and masks before patchifying
for idx, (img, mask) in enumerate(zip(large_images, large_masks)):
    if img.shape[:2] != mask.shape:
        print(f"Shape mismatch at index {idx}: image shape {img.shape} vs mask shape {mask.shape}")
        print(f"Image file: {image_files[idx]}")
        print(f"Mask file: {mask_files[idx]}")

# Creating the Dataset Dictionary
dataset_dict = {
    "image": [Image.fromarray(img.astype(np.uint8)) for img in large_images],
    "label": [Image.fromarray(mask.astype(np.uint8)) for mask in large_masks],
}


# Create the dataset using the datasets.Dataset class
from datasets import Dataset
from PIL import Image
dataset = Dataset.from_dict(dataset_dict)
  
# Ensure reproducibility by setting seeds for random, numpy, and torch
seed = 2147483647

# Split the dataset into training and validation sets
train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=0.15, random_state=seed)
train_data = torch.utils.data.Subset(dataset, train_idx)
val_data = torch.utils.data.Subset(dataset, val_idx)


"""Let us make sure out images and masks (labels) are loading appropriately"""
img_num = random.randint(0, large_images.shape[0]-1)
# image_num = 0
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Assuming the first image is grayscale
axes[0].set_title("Image")

# Plot the second image on the right
axes[1].imshow(example_mask, cmap='gray')  # Assuming the second image is grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

plt.tight_layout()
plt.show()
plt.close()

"""Get bounding boxes from masks. We will use it as input prompt"""
def get_bounding_box(ground_truth_map):
  # get bounding box from mask
  y_indices, x_indices = np.where(ground_truth_map > 0)
  x_min, x_max = np.min(x_indices), np.max(x_indices)
  y_min, y_max = np.min(y_indices), np.max(y_indices)
  # add perturbation to bounding box coordinates
  H, W = ground_truth_map.shape
  x_min = max(0, x_min - np.random.randint(0, 20))
  x_max = min(W, x_max + np.random.randint(0, 20))
  y_min = max(0, y_min - np.random.randint(0, 20))
  y_max = min(H, y_max + np.random.randint(0, 20))
  bbox = [x_min, y_min, x_max, y_max]

  return bbox

"""Get points from masks. We will use it as input prompt""" 
def zero_out_circle(target_mask, center, radius, inverse=False):
    rows, cols = np.ogrid[:target_mask.shape[0], :target_mask.shape[1]]
    circular_mask = (rows - center[1])**2 + (cols - center[0])**2 <= radius**2
    target_mask[circular_mask] = 1 if inverse else 0
    return target_mask


def get_point_prompts(ground_truth_map, num_points=None, distance=None, positive=True, existing_points=[]):
    target_mask = np.copy(ground_truth_map)
    input_points = []

    if distance > 0:
        for point in existing_points:
            zero_out_circle(target_mask, point, distance, not positive)

    while len(input_points) < num_points:
        valid_indices = np.argwhere(target_mask != 0) if positive else np.argwhere(target_mask == 0)

        if len(valid_indices) == 0:
            print(f"Found only {len(input_points)} control points when {num_points} requested.")
            print("No valid points left.")
            break  # We have fewer than "number" points, but there are no valid points left.
        # END IF

        # TODO: Selection of point position could potentially be improved. If the point is too near
        # the edge (judged by the distance to the nearest invalid index) then try picking again.
        random_index = np.random.randint(0, len(valid_indices))
        row, col = valid_indices[random_index]
        new_point = [col, row]

        input_points.append(new_point)
        if distance > 0:
            zero_out_circle(target_mask, new_point, distance, not positive)
    # END WHILE

    return input_points


# Creating the Custom Dataset Class (SAMDataset)
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import SamProcessor

class SAMDataset(Dataset):
    def __init__(self, dataset, processor, resize=None, augment=None):
        self.dataset = dataset
        self.processor = processor
        self.resize = resize
        self.augment = augment
        
        # Using some data augmentation methods to increase the diversity of training data and avoid the overfitting
        self.augmentation_transforms = transforms.Compose([
            # transforms.RandomApply([transforms.RandomRotation(degrees=30)], p=0.5),
            #     transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.5),
            # transforms.RandomGrayscale(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5)
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        original_idx = idx 
        item = self.dataset[original_idx]
        image = item["image"]
        label = item["label"]
        
        if self.resize:
            image = self.resize(image)
            label = self.resize(label)

        if self.augment:
            # Apply the same augmentation to both image and label
            seed = np.random.randint(2147483647)  # Get a random seed
            random.seed(seed)
            torch.manual_seed(seed)
            image = self.augmentation_transforms(image)
            random.seed(seed)
            torch.manual_seed(seed)
            label = self.augmentation_transforms(label)

            # Plot the augmented image and mask 
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            ax[0].imshow(image)
            ax[0].set_title("Augmented Image")
            ax[1].imshow(label, cmap='gray')
            ax[1].set_title("Augmented Mask")
            # plt.show()
            plt.close()

        ground_truth_mask = np.array(label)

        # # get bounding box prompt
        # boxes = get_bounding_box(ground_truth_mask) # Uncomment if you want to use bounding boxes as prompts

        # # prepare image and prompt for the model
        # inputs = self.processor(image, input_boxes=[[boxes]], return_tensors="pt") # Uncomment if you want to use bounding boxes as prompts
        
        
        # Generate point prompts
        points = get_point_prompts(ground_truth_mask, num_points=85, distance=0) # Uncomment if you want to use regular grid points as prompts
        # print(points)
        # prepare image and points for the model
        inputs = self.processor(image, input_points=[points], return_tensors="pt") # Uncomment if you want to use points as prompts
        

        # remove batch dimension which the processor adds by default
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # add ground truth segmentation
        inputs["ground_truth_mask"] = ground_truth_mask

        return inputs

# Initialize the processor
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

# Creating Training and Validation Datasets
train_dataset = SAMDataset(dataset=train_data, processor=processor, 
                           resize=transforms.Resize([256, 256]), augment=True)  # Data augmentation is only applied to training data

val_dataset = SAMDataset(dataset=val_data, processor=processor,
                           resize=transforms.Resize([256, 256]), augment=False) # Data augmentation is not applied to validation data

print('training dataset', len(train_dataset))
print('validation dataset', len(val_dataset))


# # Create a DataLoader instance for the training dataset
from torch.utils.data import DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=False)

val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=True, drop_last=False)

batch = next(iter(train_dataloader))

batch["ground_truth_mask"].shape

# Load the model
from transformers import SamModel
model = SamModel.from_pretrained("facebook/sam-vit-base")

# make sure we only compute gradients for mask decoder
for name, param in model.named_parameters():
  if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
    param.requires_grad_(False)
    
from tqdm import tqdm
from datetime import datetime
from pytz import timezone
from torch.optim.lr_scheduler import ReduceLROnPlateau
from monai.losses import DiceLoss, FocalLoss

import monai

# Initialize the loss functions
seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

# Initialize the optimizer and learning rate scheduler
optimizer = torch.optim.Adam(model.mask_decoder.parameters(), lr=1e-3, weight_decay=0)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

# Calculating Validation Accuracy
def calc_validation_accuracy(large_masks_val, pred):
    y_true = large_masks_val.ravel()
    y_pred = pred.ravel()
        
    return {
        "f1_score_validation": f1_score(y_true, y_pred, zero_division=0),
        "precision_validation": precision_score(y_true, y_pred, zero_division=0),
        "recall_validation": recall_score(y_true, y_pred, zero_division=0),
        "iou_score_validation": jaccard_score(y_true, y_pred, zero_division=0),
    }

# Getting the Current Time in Central European Summer Time (CEST)
def get_current_time():
    # Set the timezone to Central European Summer Time (CEST)
    tz = timezone('Europe/Berlin')
    # Get the current time in the specified timezone
    current_datetime = datetime.now(tz)
    # Format the datetime object as desired
    return current_datetime.strftime("%H:%M %Y/%m/%d")

# Print the current time in Germany (CEST)
print(f"Training started at: {get_current_time()}")

# Training loop (number of epochs)
num_epochs = 50

# Set the device to cuda if it is available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Training and Validation Loop
best_loss = float('inf')
train_losses = []
val_losses = []
val_metrics = []

for epoch in range(num_epochs):
    model.train()
    epoch_losses = []
    
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
        outputs = model(pixel_values=batch["pixel_values"].to(device),
                        # input_boxes=batch["input_boxes"].to(device), #If you use bounding boxes as input prompt, uncomment this line
                        input_points=batch["input_points"].to(device), #If you use points as prompt, uncomment this line
                        multimask_output=False)

        predicted_masks = outputs.pred_masks.squeeze(1)
        ground_truth_masks = batch["ground_truth_mask"].float().to(device)
        
        loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_losses.append(loss.item())

    train_loss_mean = np.mean(epoch_losses)
    train_losses.append(train_loss_mean)

    model.eval()
    epoch_val_losses = []
 
    epoch_val_metrics = {
        "f1_score_validation": [],
        "precision_validation": [],
        "recall_validation": [],
        "iou_score_validation": [],
    }
    
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
            outputs = model(pixel_values=batch["pixel_values"].to(device),
                        # input_boxes=batch["input_boxes"].to(device), #If you use bounding boxes as prompt, uncomment this line
                        input_points=batch["input_points"].to(device), #If you use points as prompt, uncomment this line
                            multimask_output=False)

            predicted_masks = outputs.pred_masks.squeeze(1)
            ground_truth_masks = batch["ground_truth_mask"].float().to(device)
            
            val_loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))
            
            pred_binary_masks = (predicted_masks > 0.5).float()
            gt_binary_masks = ground_truth_masks
            
            batch_metrics = calc_validation_accuracy(gt_binary_masks.cpu(), pred_binary_masks.cpu())
            for key, value in batch_metrics.items():
                epoch_val_metrics[key].append(value)
            
            epoch_val_losses.append(val_loss.item())
    
    val_loss_mean = np.mean(epoch_val_losses)
    val_losses.append(val_loss_mean)
    
    mean_val_metrics = {key: np.mean(value) for key, value in epoch_val_metrics.items()}
    val_metrics.append(mean_val_metrics)

    print(f'EPOCH: {epoch+1}')
    print(f'loss: {train_loss_mean}, val_loss: {val_loss_mean}')
    print(f'Validation Metrics: {mean_val_metrics}')
    wandb.log({"train_loss": train_loss_mean, "val_loss": val_loss_mean, **mean_val_metrics})

    scheduler.step(val_loss_mean)
    
    # Saving the Best Model. To choose the best epoch, the epoch which has the minimum validation loss, is selected. It will avoid overfitting.
    if val_loss_mean < best_loss:
        best_loss = val_loss_mean
        # The checkpoint for best epoch is saved.
        torch.save(model.state_dict(), "D:/FineTuneSAM/checkpoints/resize400imbalanced100epochs_batch4_best_BBOX.pth")
    
    print(f"Best epoch: {np.argmin(val_losses) + 1}")
    print(f"Best val loss: {min(val_losses)}")

# Also the checkpoint for last epoch is saved.
torch.save(model.state_dict(), "D:/FineTuneSAM/checkpoints/resize400imbalanced100epochs_batch4_last_BBOX.pth")
print(f"Training ended at {get_current_time()}")

# Plotting the training and validation losses
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Losses')
plt.legend()
plt.ylim(0, 2)
plt.xlim(1, num_epochs)
plt.tight_layout()
plt.show()

# Plotting the validation metrics
metrics_names = ["f1_score_validation", "precision_validation", "recall_validation", "iou_score_validation"]

for metric in metrics_names:
    metric_values = [epoch_metrics[metric] for epoch_metrics in val_metrics]
    plt.plot(range(1, num_epochs + 1), metric_values, label=metric)

plt.xlabel('Epochs')
plt.ylabel('Metric Value')
plt.title('Validation Metrics')
plt.legend()
plt.ylim(0, 1)
plt.xlim(1, num_epochs)
plt.tight_layout()
plt.show()

# ***************************************************************************************************************************************
"""**Inference**"""
"""**For segmentation, we used point prompts in the training phase of our fine-tuned SAM. However, in the testing phase, we used two different approaches 
to comprehensively evaluate the performance of the model. First, we tested the model without prompts, as manual SE segmentation is challenging and in 
real-world scenarios often no pre-existing masks are available. Secondly, we considered a scenario where only minimal user input is possible in real applications. 
Here is when you use grid of points as prompts in testing phase (First approach) **"""

from transformers import SamModel, SamConfig, SamProcessor
import torch
import numpy as np
import rasterio
from PIL import Image
import matplotlib.pyplot as plt
import os
import glob
from rasterio.features import geometry_mask
from shapely.geometry import shape
import fiona

# Load the model configuration and processor
model_config = SamConfig.from_pretrained("facebook/sam-vit-base")
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

# Create an instance of the model architecture with the loaded configuration
my_model = SamModel(config=model_config)
# Update the model by loading the weights from saved file.
my_model.load_state_dict(torch.load("D:/FineTuneSAM/checkpoints/resize400imbalanced100epochs_batch4_best_BBOX.pth", map_location=torch.device('cpu')))

# Set the device to cuda if available, otherwise use cpu
device = "cuda" if torch.cuda.is_available() else "cpu"
my_model.to(device)


# Paths to test images and masks
img_path_test=glob.glob("D:/Erosion_signs/testing/images/*.tif")
mask_path_test=glob.glob("D:/Erosion_signs/testing/masks/*.tif")
shapefile_path = glob.glob("D:/Erosion_signs/testing/shapes/*.shp")

img_path_test.sort()
mask_path_test.sort()
shapefile_path.sort()

large_images_test = []
large_masks_test = []

for i, j, shp in zip(img_path_test, mask_path_test, shapefile_path):
    with rasterio.open(i) as src:
        img = src.read([1, 2, 3])
        if img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        
        with fiona.open(shp, "r") as shapefile:
            shapes = [shape(feature["geometry"]) for feature in shapefile]
            # mask out the pixels outside the shapefile
            mask = geometry_mask(shapes, transform=src.transform, invert=True, out_shape=(src.height, src.width))
        
        large_images_test.append(img)

    with rasterio.open(j) as src:
        mask = src.read(1)
        large_masks_test.append(mask)

large_images_test = np.array(large_images_test, dtype='object')
large_masks_test = np.array(large_masks_test, dtype='object')

print(f"large_images shape: {large_images_test.shape}")
print(f"large_masks shape: {large_masks_test.shape}")
print(f"large_images[0] shape: {large_images_test[0].shape}")

# Check for shape mismatches
for idx, (img_test, mask_test) in enumerate(zip(large_images_test, large_masks_test)):
    if img_test.shape[:2] != mask_test.shape:
        print(f"Shape mismatch at index {idx}: image shape {img_test.shape} vs mask shape {mask_test.shape}")

# Count the number of pixels inside and outside the shapefiles
outside_counts = []
inside_counts = []
for idx, img in enumerate(large_images_test):
    with rasterio.open(img_path_test[idx]) as src:
        with fiona.open(shapefile_path[idx], "r") as shapefile:
            shapes = [shape(feature["geometry"]) for feature in shapefile]
            mask = geometry_mask(shapes, transform=src.transform, invert=True, out_shape=(src.height, src.width))

        inside_count = np.sum(mask)
        outside_count = np.sum(~mask)
        outside_counts.append(outside_count)
        inside_counts.append(inside_count)

        print(f"Image {idx}:")
        print(f"Number of pixels inside shapefile: {inside_count}")
        print(f"Number of pixels outside shapefile: {outside_count}")

# Create grid points as prompts
array_size = 256
grid_size = 5
x = np.linspace(0, array_size-1, grid_size)
y = np.linspace(0, array_size-1, grid_size)
xv, yv = np.meshgrid(x, y)
xv_list = xv.tolist()
yv_list = yv.tolist()
input_points = [[[int(x), int(y)] for x, y in zip(x_row, y_row)] for x_row, y_row in zip(xv_list, yv_list)]
input_points = torch.tensor(input_points).view(1, 1, grid_size*grid_size, 2)
print('(batch_size, point_batch_size, num_points_per_image, 2):', np.array(input_points).shape)

# Test accuracy function
def calc_test_accuracy(large_masks_test, pred):
    y_true = large_masks_test.ravel()
    y_pred = pred.ravel()
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()    
    return {
        "f1_score_test": f1_score(y_true, y_pred, zero_division=0),
        "precision_test": precision_score(y_true, y_pred, zero_division=0),
        "recall_test": recall_score(y_true, y_pred, zero_division=0),
        "iou_score_test": jaccard_score(y_true, y_pred, zero_division=0),
        "accuracy_test": accuracy_score(y_true, y_pred),
        "true_positive": tp,
        "All_true_negative_pixels": tn,
        "false_positive": fp,
        "false_negative": fn,
    }

all_metrics = []
true_positives = []
true_negatives = []
false_positives = []
false_negatives = []
erosion_pixels =[]
for k in range(len(large_images_test)):
    large_test_image = large_images_test[k]
    large_mask_image = large_masks_test[k]

    original_shape = large_test_image.shape[:2]

    if large_test_image.dtype != np.uint8:
        large_test_image = large_test_image.astype(np.uint8)
    if large_mask_image.dtype != np.uint8:
        large_mask_image = large_mask_image.astype(np.uint8)

    large_test_image_resized = np.array(Image.fromarray(large_test_image).resize((256, 256), Image.BILINEAR))
    large_mask_image_resized = np.array(Image.fromarray(large_mask_image).resize((256, 256), Image.NEAREST))

    single_patch = Image.fromarray(large_test_image_resized)
    inputs = processor(single_patch, input_points=input_points, return_tensors="pt")


    inputs = {k: v.to(device) for k, v in inputs.items()}
    my_model.eval()

    with torch.no_grad():
        outputs = my_model(**inputs, multimask_output=False)

    single_patch_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
    single_patch_prob = single_patch_prob.cpu().numpy().squeeze()
    Threshold = 0.5
    pred_resized = (single_patch_prob > Threshold).astype(np.uint8)
    
    pred = np.array(Image.fromarray(pred_resized).resize(original_shape[::-1], Image.NEAREST))
    probability = np.array(Image.fromarray(single_patch_prob).resize(original_shape[::-1], Image.NEAREST))
    
    # Count the number of pixels with erosion
    erosion_pixel_num = np.sum(pred)
    print(f'Number of erosion pixels: {erosion_pixel_num}')


    """**For segmentation, we used point prompts in the training phase of our fine-tuned SAM. However, in the testing phase, we used two different approaches 
    to comprehensively evaluate the performance of the model. First, we tested the model without prompts, as manual SE segmentation is challenging and in 
    real-world scenarios often no pre-existing masks are available. Secondly, we considered a scenario where only minimal user input is possible in real applications. 
    Here is when you use second approach with points as prompts**"""

# from transformers import SamModel, SamConfig, SamProcessor
# import torch
# import numpy as np
# import rasterio
# from PIL import Image
# import matplotlib.pyplot as plt
# import os
# import glob
# from rasterio.features import geometry_mask
# from shapely.geometry import shape
# import fiona
# from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, jaccard_score, accuracy_score

# # Load the model configuration and processor
# model_config = SamConfig.from_pretrained("facebook/sam-vit-base")
# processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

# # Create an instance of the model architecture with the loaded configuration
# my_model = SamModel(config=model_config)
# # Update the model by loading the weights from saved file.
# my_model.load_state_dict(torch.load("D:/FineTuneSAM/checkpoints/resize400imbalanced100epochs_batch4_best_BBOX.pth", map_location=torch.device('cpu')))

# # Set the device to cuda if available, otherwise use cpu
# device = "cuda" if torch.cuda.is_available() else "cpu"
# my_model.to(device)

# # Paths to test images and masks
# img_path_test = glob.glob("D:/Erosion_signs/testing/images/*.tif")
# mask_path_test = glob.glob("D:/Erosion_signs/testing/masks/*.tif")
# shapefile_path = glob.glob("D:/Erosion_signs/testing/shapes/*.shp")

# img_path_test.sort()
# mask_path_test.sort()
# shapefile_path.sort()

# large_images_test = []
# large_masks_test = []

# for i, j, shp in zip(img_path_test, mask_path_test, shapefile_path):
#     with rasterio.open(i) as src:
#         img = src.read([1, 2, 3])
#         if img.shape[0] == 3:
#             img = np.transpose(img, (1, 2, 0))
        
#         with fiona.open(shp, "r") as shapefile:
#             shapes = [shape(feature["geometry"]) for feature in shapefile]
#             # mask out the pixels outside the shapefile
#             mask = geometry_mask(shapes, transform=src.transform, invert=True, out_shape=(src.height, src.width))
        
#         large_images_test.append(img)

#     with rasterio.open(j) as src:
#         mask = src.read(1)
#         large_masks_test.append(mask)

# large_images_test = np.array(large_images_test, dtype='object')
# large_masks_test = np.array(large_masks_test, dtype='object')

# print(f"large_images shape: {large_images_test.shape}")
# print(f"large_masks shape: {large_masks_test.shape}")
# print(f"large_images[0] shape: {large_images_test[0].shape}")

# # Check for shape mismatches
# for idx, (img_test, mask_test) in enumerate(zip(large_images_test, large_masks_test)):
#     if img_test.shape[:2] != mask_test.shape:
#         print(f"Shape mismatch at index {idx}: image shape {img_test.shape} vs mask shape {mask_test.shape}")

# # Count the number of pixels inside and outside the shapefiles
# outside_counts = []
# inside_counts = []
# for idx, img in enumerate(large_images_test):
#     with rasterio.open(img_path_test[idx]) as src:
#         with fiona.open(shapefile_path[idx], "r") as shapefile:
#             shapes = [shape(feature["geometry"]) for feature in shapefile]
#             mask = geometry_mask(shapes, transform=src.transform, invert=True, out_shape=(src.height, src.width))

#         inside_count = np.sum(mask)
#         outside_count = np.sum(~mask)
#         outside_counts.append(outside_count)
#         inside_counts.append(inside_count)

#         print(f"Image {idx}:")
#         print(f"Number of pixels inside shapefile: {inside_count}")
#         print(f"Number of pixels outside shapefile: {outside_count}")


# # Test accuracy function
# def calc_test_accuracy(large_masks_test, pred):
#     y_true = large_masks_test.ravel()
#     y_pred = pred.ravel()
    
#     tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()    
#     return {
#         "f1_score_test": f1_score(y_true, y_pred, zero_division=0),
#         "precision_test": precision_score(y_true, y_pred, zero_division=0),
#         "recall_test": recall_score(y_true, y_pred, zero_division=0),
#         "iou_score_test": jaccard_score(y_true, y_pred, zero_division=0),
#         "accuracy_test": accuracy_score(y_true, y_pred),
#         "true_positive": tp,
#         "All_true_negative_pixels": tn,
#         "false_positive": fp,
#         "false_negative": fn,
#     }

# all_metrics = []
# true_positives = []
# true_negatives = []
# false_positives = []
# false_negatives = []
# erosion_pixels =[]
# for k in range(len(large_images_test)):
#     large_test_image = large_images_test[k]
#     large_mask_image = large_masks_test[k]

#     original_shape = large_test_image.shape[:2]

#     if large_test_image.dtype != np.uint8:
#         large_test_image = large_test_image.astype(np.uint8)
#     if large_mask_image.dtype != np.uint8:
#         large_mask_image = large_mask_image.astype(np.uint8)

#     large_test_image_resized = np.array(Image.fromarray(large_test_image).resize((256, 256), Image.BILINEAR))
#     large_mask_image_resized = np.array(Image.fromarray(large_mask_image).resize((256, 256), Image.NEAREST))

#     single_patch = Image.fromarray(large_test_image_resized)
    
#     # Use get_point_prompts to generate input points
#     input_points = get_point_prompts(large_mask_image_resized, num_points=85, distance=0, positive=True)
#     input_points = torch.tensor(input_points).view(1, 1, len(input_points), 2)
    
#     inputs = processor(single_patch, input_points=input_points, return_tensors="pt")

#     inputs = {k: v.to(device) for k, v in inputs.items()}
#     my_model.eval()

#     with torch.no_grad():
#         outputs = my_model(**inputs, multimask_output=False)

#     single_patch_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
#     single_patch_prob = single_patch_prob.cpu().numpy().squeeze()
#     Threshold = 0.5
#     pred_resized = (single_patch_prob > Threshold).astype(np.uint8)
    
#     pred = np.array(Image.fromarray(pred_resized).resize(original_shape[::-1], Image.NEAREST))
#     probability = np.array(Image.fromarray(single_patch_prob).resize(original_shape[::-1], Image.NEAREST))
    
#     # Count the number of pixels with erosion
#     erosion_pixel_num = np.sum(pred)
#     print(f'Number of erosion pixels: {erosion_pixel_num}')
    
    
    """**For segmentation, we used point prompts in the training phase of our fine-tuned SAM. However, in the testing phase, we used two different approaches 
        to comprehensively evaluate the performance of the model. First, we tested the model without prompts, as manual SE segmentation is challenging and in 
        real-world scenarios often no pre-existing masks are available. Secondly, we considered a scenario where only minimal user input is possible in real applications. 
        Here is when you use second approach with bounding boxes as prompts**"""
# bounding box 
# from transformers import SamModel, SamConfig, SamProcessor
# import torch
# import numpy as np
# import rasterio
# from PIL import Image
# import matplotlib.pyplot as plt
# import os
# import glob
# from rasterio.features import geometry_mask
# from shapely.geometry import shape
# import fiona
# from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, jaccard_score, accuracy_score

# # Load the model configuration and processor
# model_config = SamConfig.from_pretrained("facebook/sam-vit-base")
# processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

# # Create an instance of the model architecture with the loaded configuration
# my_model = SamModel(config=model_config)
# # Update the model by loading the weights from saved file.
# my_model.load_state_dict(torch.load("D:/FineTuneSAM/checkpoints/resize400imbalanced100epochs_batch4_best_BBOX.pth", map_location=torch.device('cpu')))

# # Set the device to cuda if available, otherwise use cpu
# device = "cuda" if torch.cuda.is_available() else "cpu"
# my_model.to(device)

# # Paths to test images and masks
# img_path_test = glob.glob("D:/Erosion_signs/testing/images/*.tif")
# mask_path_test = glob.glob("D:/Erosion_signs/testing/masks/*.tif")
# shapefile_path = glob.glob("D:/Erosion_signs/testing/shapes/*.shp")

# img_path_test.sort()
# mask_path_test.sort()
# shapefile_path.sort()

# large_images_test = []
# large_masks_test = []

# for i, j, shp in zip(img_path_test, mask_path_test, shapefile_path):
#     with rasterio.open(i) as src:
#         img = src.read([1, 2, 3])
#         if img.shape[0] == 3:
#             img = np.transpose(img, (1, 2, 0))
        
#         with fiona.open(shp, "r") as shapefile:
#             shapes = [shape(feature["geometry"]) for feature in shapefile]
#             # mask out the pixels outside the shapefile
#             mask = geometry_mask(shapes, transform=src.transform, invert=True, out_shape=(src.height, src.width))
        
#         large_images_test.append(img)

#     with rasterio.open(j) as src:
#         mask = src.read(1)
#         large_masks_test.append(mask)

# large_images_test = np.array(large_images_test, dtype='object')
# large_masks_test = np.array(large_masks_test, dtype='object')

# print(f"large_images shape: {large_images_test.shape}")
# print(f"large_masks shape: {large_masks_test.shape}")
# print(f"large_images[0] shape: {large_images_test[0].shape}")

# # Check for shape mismatches
# for idx, (img_test, mask_test) in enumerate(zip(large_images_test, large_masks_test)):
#     if img_test.shape[:2] != mask_test.shape:
#         print(f"Shape mismatch at index {idx}: image shape {img_test.shape} vs mask shape {mask_test.shape}")

# # Count the number of pixels inside and outside the shapefiles
# outside_counts = []
# inside_counts = []
# for idx, img in enumerate(large_images_test):
#     with rasterio.open(img_path_test[idx]) as src:
#         with fiona.open(shapefile_path[idx], "r") as shapefile:
#             shapes = [shape(feature["geometry"]) for feature in shapefile]
#             mask = geometry_mask(shapes, transform=src.transform, invert=True, out_shape=(src.height, src.width))

#         inside_count = np.sum(mask)
#         outside_count = np.sum(~mask)
#         outside_counts.append(outside_count)
#         inside_counts.append(inside_count)

#         print(f"Image {idx}:")
#         print(f"Number of pixels inside shapefile: {inside_count}")
#         print(f"Number of pixels outside shapefile: {outside_count}")

# """Get bounding box from masks.""" # Myown..................................
# def get_bounding_box(ground_truth_map):
#     # get bounding box from mask
#     y_indices, x_indices = np.where(ground_truth_map > 0)
#     x_min, x_max = np.min(x_indices), np.max(x_indices)
#     y_min, y_max = np.min(y_indices), np.max(y_indices)
#     # add perturbation to bounding box coordinates
#     H, W = ground_truth_map.shape
#     x_min = max(0, x_min - np.random.randint(0, 20))
#     x_max = min(W, x_max + np.random.randint(0, 20))
#     y_min = max(0, y_min - np.random.randint(0, 20))
#     y_max = min(H, y_max + np.random.randint(0, 20))
#     bbox = [x_min, y_min, x_max, y_max]

#     return bbox

# # Test accuracy function
# def calc_test_accuracy(large_masks_test, pred):
#     y_true = large_masks_test.ravel()
#     y_pred = pred.ravel()
    
#     tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()    
#     return {
#         "f1_score_test": f1_score(y_true, y_pred, zero_division=0),
#         "precision_test": precision_score(y_true, y_pred, zero_division=0),
#         "recall_test": recall_score(y_true, y_pred, zero_division=0),
#         "iou_score_test": jaccard_score(y_true, y_pred, zero_division=0),
#         "accuracy_test": accuracy_score(y_true, y_pred),
#         "true_positive": tp,
#         "All_true_negative_pixels": tn,
#         "false_positive": fp,
#         "false_negative": fn,
#     }

# all_metrics = []
# true_positives = []
# true_negatives = []
# false_positives = []
# false_negatives = []
# erosion_pixels =[]
# for k in range(len(large_images_test)):
#     large_test_image = large_images_test[k]
#     large_mask_image = large_masks_test[k]

#     original_shape = large_test_image.shape[:2]

#     if large_test_image.dtype != np.uint8:
#         large_test_image = large_test_image.astype(np.uint8)
#     if large_mask_image.dtype != np.uint8:
#         large_mask_image = large_mask_image.astype(np.uint8)

#     large_test_image_resized = np.array(Image.fromarray(large_test_image).resize((256, 256), Image.BILINEAR))
#     large_mask_image_resized = np.array(Image.fromarray(large_mask_image).resize((256, 256), Image.NEAREST))

#     single_patch = Image.fromarray(large_test_image_resized)
    
#     # Use get_bounding_box to generate bounding box
#     bbox = get_bounding_box(large_mask_image_resized)
#     input_boxes = torch.tensor(bbox).view(1, 1, 4)
    
#     # inputs = processor(single_patch, input_boxes=input_boxes, return_tensors="pt")
#     inputs = processor(single_patch, return_tensors="pt")

#     inputs = {k: v.to(device) for k, v in inputs.items()}
#     my_model.eval()

#     with torch.no_grad():
#         outputs = my_model(**inputs, multimask_output=False)

#     single_patch_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
#     single_patch_prob = single_patch_prob.cpu().numpy().squeeze()
#     Threshold = 0.5
#     pred_resized = (single_patch_prob > Threshold).astype(np.uint8)
    
#     pred = np.array(Image.fromarray(pred_resized).resize(original_shape[::-1], Image.NEAREST))
#     probability = np.array(Image.fromarray(single_patch_prob).resize(original_shape[::-1], Image.NEAREST))
    
#     # Count the number of pixels with erosion
#     erosion_pixel_num = np.sum(pred)
#     print(f'Number of erosion pixels: {erosion_pixel_num}')
    
    
    """**The lines below are same for all senarios**"""
    # Plot the results
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    axes[0].imshow(large_test_image, cmap='gray')  
    axes[0].set_title("Image")
    cax = axes[1].imshow(probability, cmap='viridis', vmin=0, vmax=1)
    # fig.colorbar(cax, ax=axes[1], orientation='vertical')
    axes[1].set_title("Probability Map")
    axes[2].imshow(pred, cmap='gray') 
    axes[2].set_title("Prediction")
    axes[3].imshow(large_mask_image, cmap='gray') 
    axes[3].set_title("Ground-truth")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
    plt.tight_layout()
    plt.close()
    
    # Save the subplots with the input filename
    output_dir1 = "D:/FineTuneSAM/results/figs/"
    os.makedirs(output_dir1, exist_ok=True)
    input_filename1 = os.path.basename(img_path_test[k])
    fig_output_filename = os.path.join(output_dir1, input_filename1)
    fig.savefig(fig_output_filename, bbox_inches='tight')

    # Save the prediction
    output_dir = "D:/FineTuneSAM/save/"
    os.makedirs(output_dir, exist_ok=True)
    input_filename = os.path.basename(img_path_test[k])
    output_filename = os.path.join(output_dir, input_filename)

    with rasterio.open(img_path_test[k]) as src:
        transform = src.transform
        crs = src.crs

    with rasterio.open(
        output_filename, 'w',
        driver='GTiff',
        height=pred.shape[0],
        width=pred.shape[1],
        count=1,
        dtype=pred.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(pred, 1)

    print(f"Saved prediction to {output_filename}")
    
    metrics = calc_test_accuracy(large_mask_image_resized, pred_resized)
    print(f"Metrics for image {k}: {metrics}")  # Debug statement to check metrics
    all_metrics.append(metrics)
    
    true_positives.append(metrics["true_positive"])
    true_negatives.append(metrics["All_true_negative_pixels"])
    false_positives.append(metrics["false_positive"])
    false_negatives.append(metrics["false_negative"])
    erosion_pixels.append(erosion_pixel_num)


print(f"Outside counts: {outside_counts}")    
print(f"true_positives: {true_positives}")
print(f"All_true_negative_pixels: {true_negatives}")
print(f"false_positives: {false_positives}")
print(f"false_negatives: {false_negatives}")
true_negative_pixels_inside_shapefile = [a - b for a, b in zip(true_negatives, outside_counts)]
print(f"true_negative_pixels_inside_shapefile: {true_negative_pixels_inside_shapefile}")
print('number of erosion_pixels', erosion_pixels)


accuracies = []
precisions = []
recalls = []
dice_coefficients = []
ious = []
classes = []
ratios = []

# Classification of erosion based on the percentage of erosion damage in the fields
for idx, outside_count in enumerate(outside_counts):
    height, width = large_images_test[idx].shape[:2]
    print(f"Image {idx}: height={height}, width={width}")
    
    erosion_ratio = (erosion_pixels[idx]/ ((height * width) - outside_count))*100

    print('erosion_ratio', erosion_ratio)
    ratios.append(erosion_ratio)
    
    if erosion_ratio == 0:
        classification = 0
    elif 0 < erosion_ratio < 10:
        classification = 1
    elif 10 <= erosion_ratio < 30:
        classification = 2
    else:
        classification = 3
    
    classes.append(classification)
    
print('Percentage of erosion damage in the field', ratios)


# Calculate the metrics 
for idx, (tp, tn, fp, fn, tn_inside, outside_count) in enumerate(zip(true_positives, true_negatives, false_positives, false_negatives, true_negative_pixels_inside_shapefile, outside_counts)):
    accuracy = (tp + tn_inside) / (tp + tn_inside + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0

    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    dice_coefficients.append(dice)
    ious.append(iou)


# Plot each image with its class on top
for idx, (image, classification) in enumerate(zip(large_images_test, classes)):
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap='gray')  # Adjust cmap if your images are not grayscale
    plt.title(f"Class: {classification}")
    plt.axis('off')  # Hide the axes
    # plt.show()
    
# Print the calculated metrics and classes
print(f"Accuracies: {accuracies}")
print(f"Precisions: {precisions}")
print(f"Recalls: {recalls}")
print(f"Dice Coefficients: {dice_coefficients}")
print(f"Intersection-over-Union (IoU): {ious}")
print(f"Classes: {classes}")

# Calculate mean values of each metric
mean_accuracy = sum(accuracies) / len(accuracies)
mean_precision = sum(precisions) / len(precisions)
mean_recall = sum(recalls) / len(recalls)
mean_dice = sum(dice_coefficients) / len(dice_coefficients)
mean_iou = sum(ious) / len(ious)

# Print the mean values
print(f"Mean Accuracy: {mean_accuracy}")
print(f"Mean Precision: {mean_precision}")
print(f"Mean Recall: {mean_recall}")
print(f"Mean Dice Coefficient: {mean_dice}")
print(f"Mean Intersection-over-Union (IoU): {mean_iou}")

# Create a DataFrame with image names, ratios, and classes
field_names = [os.path.basename(img_path) for img_path in img_path_test]
df = pd.DataFrame({
    'Image Name': field_names,
    'Erosion Ratio': ratios,
    'Class': classes,
    'Recall': recalls,
    'Precision': precisions,
    'Dice': dice_coefficients,
    'IoU': ious
})

# Save the DataFrame to an Excel file
df.to_excel('image_classes.xlsx', index=False)

print("Classes saved to image_classes.xlsx")
# Plot the mean values
metrics = ['Accuracy', 'Precision', 'Recall', 'Dice Coefficient', 'IoU']
mean_values = [mean_accuracy, mean_precision, mean_recall, mean_dice, mean_iou]

plt.figure(figsize=(10, 6))
plt.bar(metrics, mean_values, color=['blue', 'green', 'red', 'purple', 'orange'])
plt.xlabel('Metrics')
plt.ylabel('Mean Value')
plt.title('Mean Evaluation Metrics')
plt.ylim(0, 1)  # Assuming the metrics range from 0 to 1
for i, v in enumerate(mean_values):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')

plt.show()

average_metrics = {key: sum(d[key] for d in all_metrics) / len(all_metrics) for key in all_metrics[0]}
print("Average Metrics:", average_metrics)

