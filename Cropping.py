"""
This code fine-tunes the mask decoder of the Segment Anything model (SAM) for soil erosion segmenttaion task. 
It uses the cropping approch of our paper to do so. It means that the code crops each individual agricultural field into different 256 * 256 image patches,
then use them as input to fine-tune the pretrained SAM model.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from patchify import patchify,unpatchify  #Only to handle large images
import random
import glob
import wandb
from sklearn.model_selection import train_test_split
import torch
import rasterio
torch.cuda.empty_cache()
from sklearn.metrics import f1_score, precision_score, recall_score, jaccard_score, accuracy_score, confusion_matrix

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
wandb.init(project="SAM_FineTune")


# Load tiff  images and masks
img_path=glob.glob("D:/Erosion_signs/training/images/*.tif")
mask_path=glob.glob("D:/Erosion_signs/training/masks/*.tif")


img_path.sort()
mask_path.sort()
large_images = []
large_masks = []

for i, j in zip(img_path, mask_path):
    with rasterio.open(i) as src:
        # Read all rows, all columns, bands 1, 2, 3 (assuming RGB)
        img = src.read([1, 2, 3])
        # If the shape is not (bands, rows, cols), transpose it to match the expected (rows, cols, bands)
        if img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        large_images.append(img)
        
    with rasterio.open(j) as src:
        mask = src.read(1)  # Read the first band (assuming single-band mask)
        large_masks.append(mask)

# Convert lists to numpy arrays (keeping dtype='object' to allow for different shapes if needed)
large_images = np.array(large_images, dtype='object')
large_masks = np.array(large_masks, dtype='object')

print(f"large_images shape: {large_images.shape}")
print(f"large_masks shape: {large_masks.shape}")
print(f"large_images[0] shape: {large_images[0].shape}")


# Print images with width or height less than 256 pixels
for img_file, img_array in zip(img_path, large_images):
    height, width, _ = img_array.shape
    if height < 256 or width < 256:
        print(f"Image {img_file} has dimensions {img_array.shape} which are less than 256 pixels (patch_size) in width or height.")
        
        
# Check shapes of images and masks before patchifying
for idx, (img, mask) in enumerate(zip(large_images, large_masks)):
    if img.shape[:2] != mask.shape:
        print(f"Shape mismatch at index {idx}: image shape {img.shape} vs mask shape {mask.shape}")
        
"""Now. let us divide these large images into smaller patches for training. We can use patchify or write custom code."""

#Desired patch size for smaller images and step size.
patch_size = 256
step = 256

all_img_patches = []
for img in range(large_images.shape[0]):
    large_image = large_images[img]
    patches_img = patchify(large_image, (patch_size, patch_size,3), step=step)  #Step=256 for 256 patches means no overlap

    for i in range(patches_img.shape[0]):
        for j in range(patches_img.shape[1]):

            single_patch_img = patches_img[i,j,:,:]
            all_img_patches.append(single_patch_img)

images = np.array(all_img_patches)
images=np.squeeze(images)

#Let us do the same for masks
all_mask_patches = []
for img in range(large_masks.shape[0]):
    large_mask = large_masks[img]
    patches_mask = patchify(large_mask, (patch_size, patch_size), step=step)  #Step=256 for 256 patches means no overlap

    for i in range(patches_mask.shape[0]):
        for j in range(patches_mask.shape[1]):

            single_patch_mask = patches_mask[i,j,:,:]
            single_patch_mask = single_patch_mask.astype(np.uint8)
            all_mask_patches.append(single_patch_mask)

masks = np.array(all_mask_patches)
masks=np.squeeze(masks)

print('before filtering image', images.shape)
print('before filtering mask', masks.shape)

"""Now, let us delete empty masks as they may cause issues later on during training. If a batch contains empty masks then the loss function will throw an error as it may not know how to handle empty tensors."""

# Create a list to store the indices of non-empty masks
valid_indices = [i for i, mask in enumerate(masks) if mask.max() != 0 and np.sum(mask==1)>=100]
# Filter the image and mask arrays to keep only the non-empty pairs
filtered_images = images[valid_indices]
filtered_masks = masks[valid_indices]
print("after filtering Image shape:", filtered_images.shape)  # e.g., (num_frames, height, width, num_channels)
print("after filtering Mask shape:", filtered_masks.shape)
"""Let us create a 'dataset' that serves us input images and masks for the rest of our journey."""

from datasets import Dataset
from PIL import Image

# Convert the NumPy arrays to Pillow images and store them in a dictionary
dataset_dict = {
    "image": [Image.fromarray(img) for img in filtered_images],
    "label": [Image.fromarray(mask) for mask in filtered_masks],
}

# Create the dataset using the datasets.Dataset class
dataset = Dataset.from_dict(dataset_dict)

seed = 42

train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=0.15, random_state=seed)
train_data = torch.utils.data.Subset(dataset, train_idx)
val_data = torch.utils.data.Subset(dataset, val_idx)


print('dataset:', dataset)


"""Let us make sure out images and masks (labels) are loading appropriately"""

img_num = random.randint(0, filtered_images.shape[0]-1)
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

# Display the images side by side
plt.show()
plt.close()

"""Get bounding boxes from masks."""
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

"""Get regular grid points from masks."""
# def get_point_prompts(ground_truth_map, grid_size=50):
#     H, W = ground_truth_map.shape
#     x = np.linspace(0, W-1, grid_size)
#     y = np.linspace(0, H-1, grid_size)
#     xv, yv = np.meshgrid(x, y)
#     xv_list = xv.flatten().tolist()
#     yv_list = yv.flatten().tolist()
#     points = list(zip(xv_list, yv_list))
#     return points


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


from torch.utils.data import Dataset

class SAMDataset(Dataset):
    def __init__(self, dataset, processor, augment=None):
        self.dataset = dataset
        self.processor = processor
        self.augment = augment

        self.augmentation_transforms = transforms.Compose([
            # transforms.RandomApply([transforms.RandomRotation(degrees=30)], p=0.5),
            # transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.5),
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
        prompt = get_bounding_box(ground_truth_mask) # Uncomment if you want to use bounding boxes as prompts

        # prepare image and prompt for the model
        inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt") # Uncomment if you want to use bounding boxes as prompts
        
        
        # Generate regular grid of point prompts
        # points = get_point_prompts(ground_truth_mask, num_points=85, distance=0) # Uncomment if you want to use regular grid points as prompts
    
        # # prepare image and points for the model
        # inputs = self.processor(image, input_points=[points], return_tensors="pt") # Uncomment if you want to use regular grid points as prompts
        
        
        # remove batch dimension which the processor adds by default
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # add ground truth segmentation
        inputs["ground_truth_mask"] = ground_truth_mask

        return inputs

# Initialize the processor
from transformers import SamProcessor
import torchvision.transforms as transforms
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")


train_dataset = SAMDataset(dataset=train_data, processor=processor, augment=True)  # Data augmentation is only applied to training data

val_dataset = SAMDataset(dataset=val_data, processor=processor, augment=False) # Data augmentation is not applied to validation data


print('training dataset', len(train_dataset))
print('validation dataset', len(val_dataset))



# Create a DataLoader instance for the training dataset
from torch.utils.data import DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=False)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=True, drop_last=False)

# To see if the batch is loading correctly based on above mentioned settings
batch = next(iter(train_dataloader))
for k,v in batch.items():
  print(k,v.shape)

batch["ground_truth_mask"].shape

# Load the model
from transformers import SamModel
model = SamModel.from_pretrained("facebook/sam-vit-base")

# make sure we only compute gradients for mask decoder
for name, param in model.named_parameters():
  if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
  # if name.startswith("prompt_encoder"):
    param.requires_grad_(False)


from tqdm import tqdm
from datetime import datetime
from pytz import timezone
from torch.optim.lr_scheduler import ReduceLROnPlateau
from monai.losses import DiceLoss, FocalLoss
import monai
# Initialize the loss functions
#  Try DiceCELoss or the combination of DiceLoss and FocalLoss
seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
dice_loss = DiceLoss(sigmoid=True)
focal_loss = FocalLoss(to_onehot_y=False, gamma=2.0)  # single-channel prediction

# Define the ratio
ratio = 19 / 20

# Combine the loss functions with the specified ratio
# def combined_loss(outputs, targets):
#     return ratio * focal_loss(outputs, targets) + (1 - ratio) * dice_loss(outputs, targets)
# # Combine the loss functions
def combined_loss(outputs, targets):
    return dice_loss(outputs, targets) + focal_loss(outputs, targets)

# Initialize the optimizer and learning rate scheduler
optimizer = torch.optim.Adam(model.mask_decoder.parameters(), lr=1e-3, weight_decay=0)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

def calc_validation_accuracy(large_masks_test, pred):
    y_true = large_masks_test.ravel()
    y_pred = pred.ravel()
    return {
        "f1_score_validation": f1_score(y_true, y_pred, zero_division=0),
        "precision_validation": precision_score(y_true, y_pred, zero_division=0),
        "recall_validation": recall_score(y_true, y_pred, zero_division=0),
        "iou_score_validation": jaccard_score(y_true, y_pred, zero_division=0),
        # "accuracy_validation": accuracy_score(y_true, y_pred),
    }

def get_current_time():
    # Set the timezone to Central European Summer Time (CEST)
    tz = timezone('Europe/Berlin')
    # Get the current time in the specified timezone
    current_datetime = datetime.now(tz)
    # Format the datetime object as desired
    return current_datetime.strftime("%H:%M %Y/%m/%d")

# Print the current time in Germany (CEST)
print(f"Training started at: {get_current_time()}")

# Training loop
num_epochs = 50

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

best_loss = float('inf')
train_losses = []
val_losses = []
val_metrics = []

for epoch in range(num_epochs):
    model.train()
    epoch_losses = []
    
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
        outputs = model(pixel_values=batch["pixel_values"].to(device),
                        # input_points=batch["input_points"].to(device),
                        input_boxes=batch["input_boxes"].to(device),
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
        # "accuracy_validation": []
    }
    
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
            outputs = model(pixel_values=batch["pixel_values"].to(device),
                            # input_points=batch["input_points"].to(device),
                            input_boxes=batch["input_boxes"].to(device),
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

    if val_loss_mean < best_loss:
        best_loss = val_loss_mean
        torch.save(model.state_dict(), "D:/FineTuneSAM/checkpoints/patchify10best.pth")
    
    print(f"Best epoch: {np.argmin(val_losses) + 1}")
    print(f"Best val loss: {min(val_losses)}")

torch.save(model.state_dict(), "D:/FineTuneSAM/checkpoints/patchify10best.pth")
print(f"Training ended at {get_current_time()}")

# Plotting the losses
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Losses')
plt.legend()
plt.ylim(0, 1)
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

from transformers import SamModel, SamConfig, SamProcessor
import torch
from rasterio.features import geometry_mask
from shapely.geometry import shape
import fiona

# Load the model configuration
model_config = SamConfig.from_pretrained("facebook/sam-vit-base")
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

# Create an instance of the model architecture with the loaded configuration
my_model = SamModel(config=model_config)
#Update the model by loading the weights from saved file.
my_model.load_state_dict(torch.load("D:/FineTuneSAM/checkpoints/patchify10best.pth", map_location=torch.device('cpu')))

# set the device to cuda if available, otherwise use cpu
device = "cuda" if torch.cuda.is_available() else "cpu"
my_model.to(device)


# Load tiff images and masks and shape files
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

# Check shapes of images and masks before patchifying
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
        
# Define the size of your array
array_size = 256

# Define the size of your grid
grid_size = 50

# Generate the grid points
x = np.linspace(0, array_size-1, grid_size)
y = np.linspace(0, array_size-1, grid_size)

# Generate a grid of coordinates
xv, yv = np.meshgrid(x, y)

# Convert the numpy arrays to lists
xv_list = xv.tolist()
yv_list = yv.tolist()

# Combine the x and y coordinates into a list of list of lists
input_points = [[[int(x), int(y)] for x, y in zip(x_row, y_row)] for x_row, y_row in zip(xv_list, yv_list)]
# (batch_size, point_batch_size, num_points_per_image, 2),

input_points = torch.tensor(input_points).view(1, 1, grid_size*grid_size, 2)
print('(batch_size, point_batch_size, num_points_per_image, 2):',np.array(input_points).shape)

import cv2 as cv
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import gc
from scipy.signal.windows import hamming

def calc_test_accuracy(large_masks_test, pred):
    y_true = large_masks_test.ravel()
    y_pred = pred.ravel()
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()    
    return {
        "f1_score_test": f1_score(y_true, y_pred, zero_division=0),
        "precision_test": precision_score(y_true, y_pred, zero_division=0),
        "recall_test": recall_score(y_true, y_pred, zero_division=0),
        "iou_score_test": jaccard_score(y_true, y_pred, zero_division=0),
        # "accuracy_test": accuracy_score(y_true, y_pred),
        "true_positive": tp,
        "All_true_negative_pixels": tn,
        "false_positive": fp,
        "false_negative": fn,
    }
    

def _spline_window(window_size, power=2):
    print("_spline_window")
    """
    Squared spline (power=2) window function:
    https://www.wolframalpha.com/input/?i=y%3Dx**2,+y%3D-(x-2)**2+%2B2,+y%3D(x-4)**2,+from+y+%3D+0+to+2
    """
    # Generate a Hamming window of the desired size
    hamming_window = hamming(window_size)
    
    # Calculate outer and inner parts of the spline window
    intersection = int(window_size / 4)
    wind_outer = (abs(2 * hamming_window) ** power) / 2
    wind_outer[intersection:-intersection] = 0

    wind_inner = 1 - (abs(2 * (hamming_window - 1)) ** power) / 2
    wind_inner[:intersection] = 0
    wind_inner[-intersection:] = 0

    wind = wind_inner + wind_outer
    wind = wind / np.average(wind)
    return wind

cached_2d_windows = dict()

import math

def calculate_total_patches(images, window_size, subdivisions):
    total_patches = 0
    step = window_size // subdivisions  # Step size between patches
    for img in images:
        height, width = img.shape[:2]  # Get dimensions of the image
        patches_along_height = math.ceil((height - window_size) / step) + 1
        patches_along_width = math.ceil((width - window_size) / step) + 1
        total_patches += patches_along_height * patches_along_width
    return total_patches

# Calculate the total number of patches
total_patches = calculate_total_patches(large_images_test, window_size=256, subdivisions=2)
print(f"Total number of patches: {total_patches}")


def _window_2D(window_size, power=2):
    print("_window_2D")
    """
    Make a 1D window function, then infer and return a 2D window function.
    Done with an augmentation, and self multiplication with its transpose.
    Could be generalized to more dimensions.
    """
    # Memoization
    global cached_2d_windows
    key = "{}_{}".format(window_size, power)
    if key in cached_2d_windows:
        wind = cached_2d_windows[key]
    else:
        wind = _spline_window(window_size, power)
        wind = np.expand_dims(np.expand_dims(wind, 1), 2)
        wind = wind * wind.transpose(1, 0, 2)

            # For demo purpose, let's look once at the window:
        plt.figure()
        plt.imshow(wind[:, :, 0], cmap="viridis")
        plt.title("2D Windowing Function for a Smooth Blending of "
                      "Overlapping Patches")
        # plt.show()
        cached_2d_windows[key] = wind
    return wind


def _pad_img(img, window_size, subdivisions):
    print("_pad_img")
    """
    Add borders to img for a "valid" border pattern according to "window_size" and
    "subdivisions".
    Image is an np array of shape (x, y, nb_channels).
    """
    aug = int(round(window_size * (1 - 1.0/subdivisions)))
    more_borders = ((aug, aug), (aug, aug), (0, 0))
    ret = np.pad(img, pad_width=more_borders, mode='reflect')
    gc.collect()

        # For demo purpose, let's look once at the window:
    plt.figure()
    plt.imshow(ret)
    plt.title("Padded Image for Using Tiled Prediction Patches\n"
                  "(notice the reflection effect on the padded borders)")
    # plt.show()
    return ret


def _unpad_img(padded_img, window_size, subdivisions):
    print("_unpad_img")
    """
    Undo what's done in the `_pad_img` function.
    Image is an np array of shape (x, y, nb_channels).
    """
    aug = int(round(window_size * (1 - 1.0/subdivisions)))
    ret = padded_img[
        aug:-aug,
        aug:-aug,
        :
    ]
    gc.collect()
    return ret


def _rotate_mirror_do(im):
    print("_rotate_mirror_do")
    """
    Duplicate an np array (image) of shape (x, y, nb_channels) 8 times, in order
    to have all the possible rotations and mirrors of that image that fits the
    possible 90 degrees rotations.

    It is the D_4 (D4) Dihedral group:
    https://en.wikipedia.org/wiki/Dihedral_group
    """
    mirrs = []
    mirrs.append(np.array(im))
    mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=1))
    mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=2))
    mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=3))
    im = np.array(im)[:, ::-1]
    mirrs.append(np.array(im))
    mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=1))
    mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=2))
    mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=3))
    return mirrs


def _rotate_mirror_undo(im_mirrs):
    print("rotate_mirror_undo")
    """
    merges a list of 8 np arrays (images) of shape (x, y, nb_channels) generated
    from the `_rotate_mirror_do` function. Each images might have changed and
    merging them implies to rotated them back in order and average things out.

    It is the D_4 (D4) Dihedral group:
    https://en.wikipedia.org/wiki/Dihedral_group
    """
    origs = []
    origs.append(np.array(im_mirrs[0]))
    origs.append(np.rot90(np.array(im_mirrs[1]), axes=(0, 1), k=3))
    origs.append(np.rot90(np.array(im_mirrs[2]), axes=(0, 1), k=2))
    origs.append(np.rot90(np.array(im_mirrs[3]), axes=(0, 1), k=1))
    origs.append(np.array(im_mirrs[4])[:, ::-1])
    origs.append(np.rot90(np.array(im_mirrs[5]), axes=(0, 1), k=3)[:, ::-1])
    origs.append(np.rot90(np.array(im_mirrs[6]), axes=(0, 1), k=2)[:, ::-1])
    origs.append(np.rot90(np.array(im_mirrs[7]), axes=(0, 1), k=1)[:, ::-1])
    return np.mean(origs, axis=0)


def _windowed_subdivs(padded_img, window_size, subdivisions, nb_classes, pred_func):
    print("_windowed_subdivs")
    """
    Create tiled overlapping patches.

    Returns:
        5D numpy array of shape = (
            nb_patches_along_X,
            nb_patches_along_Y,
            patches_resolution_along_X,
            patches_resolution_along_Y,
            nb_output_channels
        )

    Note:
        patches_resolution_along_X == patches_resolution_along_Y == window_size
    """
    WINDOW_SPLINE_2D = _window_2D(window_size=window_size, power=2)
    # Step of patches
    step = int(window_size/subdivisions)
    padx_len = padded_img.shape[0]
    pady_len = padded_img.shape[1]
    subdivs = []
    for i in range(0, padx_len-window_size+1, step):
        subdivs.append([])
        for j in range(0, pady_len-window_size+1, step):
            patch = padded_img[i:i+window_size, j:j+window_size, :]
            subdivs[-1].append(patch)
    # Here, `gc.collect()` clears RAM between operations.
    # It should run faster if they are removed, if enough memory is available.
    gc.collect()
    subdivs = np.array(subdivs)
    gc.collect()
    a, b, c, d, e = subdivs.shape
    subdivs = subdivs.reshape(a * b, c, d, e)
    gc.collect()
    subdivs = pred_func(subdivs)
    gc.collect()
    #subdivs = np.array([patch * WINDOW_SPLINE_2D for patch in subdivs])
    gc.collect()
    # Such 5D array:
    subdivs = subdivs.reshape(a, b, c, d, nb_classes)
    gc.collect()
    return subdivs


def _recreate_from_subdivs(subdivs, window_size, subdivisions, padded_out_shape):
    print("_recreate_from_subdivs")
    """
    Merge tiled overlapping patches smoothly.
    """
    step = int(window_size/subdivisions)
    padx_len = padded_out_shape[0]
    pady_len = padded_out_shape[1]

    y = np.zeros(padded_out_shape)

    a = 0
    for i in range(0, padx_len-window_size+1, step):
        b = 0
        for j in range(0, pady_len-window_size+1, step):
            windowed_patch = subdivs[a, b]
            y[i:i+window_size, j:j+window_size] = y[i:i+window_size, j:j+window_size] + windowed_patch
            b += 1
        a += 1
    return y / (subdivisions ** 2)


def predict_img_with_smooth_windowing(input_img, window_size, subdivisions, nb_classes, pred_func):
    print("predict_img_with_smooth_windowing2")
    """
    Apply the `pred_func` function to square patches of the image, and overlap
    the predictions to merge them smoothly.

    See 6th, 7th and 8th idea here:
    http://blog.kaggle.com/2017/05/09/dstl-satellite-imagery-competition-3rd-place-winners-interview-vladimir-sergey/
    """
    pad = _pad_img(input_img, window_size, subdivisions)
    pads = _rotate_mirror_do(pad)

    res = []
    for pad in tqdm(pads):
        # For every rotation:
        sd = _windowed_subdivs(pad, window_size, subdivisions, nb_classes, pred_func)
        one_padded_result = _recreate_from_subdivs(
            sd, window_size, subdivisions,
            padded_out_shape=list(pad.shape[:-1])+[nb_classes])
        res.append(one_padded_result)
    # Merge after rotations:
    padded_results = _rotate_mirror_undo(res)

    prd = _unpad_img(padded_results, window_size, subdivisions)

    prd = prd[:input_img.shape[0], :input_img.shape[1], :]


    plt.figure()
    plt.imshow(prd)
    plt.title("Smoothly Merged Patches that were Tiled Tighter")
    # plt.show()
    return prd


def round_predictions(prd, nb_channels_out, thresholds):
    print("round_predictions")
    """
    From a threshold list `thresholds` containing one threshold per output
    channel for comparison, the predictions are converted to a binary mask.
    """
    assert (nb_channels_out == len(thresholds))
    prd = np.array(prd)
    for i in range(nb_channels_out):
        # Per-pixel and per-channel comparison on a threshold to
        # binarize prediction masks:
        prd[:, :, i] = prd[:, :, i] > thresholds[i]
    return prd




def predict_for_patches(small_img_patches):
        y=[]
        print('small_img_patches', small_img_patches.shape)
        for img in list(small_img_patches):
            random_array = img


            single_patch = Image.fromarray(random_array)
            inputs = processor(single_patch, input_points=input_points, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            my_model.eval()

            with torch.no_grad():
                outputs = my_model(**inputs, multimask_output=False)

            single_patch_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
            single_patch_prob = single_patch_prob.cpu().numpy().squeeze()
            single_patch_prediction = (single_patch_prob > 0.5).astype(np.uint8)
            y.append(single_patch_prob)
            # print(len(y))
        y=np.array(y)
        gc.collect()
        return y


all_metrics = []
true_positives = []
true_negatives = []
false_positives = []
false_negatives = []
erosion_pixels =[]
for k in range(len(large_images_test)):
    large_test_image = large_images_test[k]
    large_mask_image = large_masks_test[k]

    window_size = 256
    img_resolution = 1000
    nb_channels_in = 3
    subdivisions = 2
    nb_channels_out = 1

    print("smoothly_predicted_img")
    smoothly_predicted_img = predict_img_with_smooth_windowing(
        large_test_image, window_size, subdivisions,
        nb_classes=nb_channels_out, pred_func=predict_for_patches
    )
    smoothly_predicted_img2 = (smoothly_predicted_img > 0.5).astype(np.uint8)
    smoothly_predicted_img2 = np.squeeze(smoothly_predicted_img2, axis=-1)
    
    # Count the number of pixels with erosion
    erosion_pixel_num = np.sum(smoothly_predicted_img2)
    print(f'Number of erosion pixels: {erosion_pixel_num}')
    erosion_pixels.append(erosion_pixel_num)

    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    axes[0].imshow(large_test_image, cmap='gray')
    axes[0].set_title("Image")

    cax = axes[1].imshow(smoothly_predicted_img.squeeze(2), cmap='viridis', vmin=0, vmax=1)
    # fig.colorbar(cax, ax=axes[1], orientation='vertical')
    axes[1].set_title("Probability Map")
    
    axes[2].imshow(smoothly_predicted_img2, cmap='gray')
    axes[2].set_title("Prediction")

    axes[3].imshow(large_mask_image, cmap='gray')
    axes[3].set_title("Ground-truth")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    # plt.show()
    
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
        height=smoothly_predicted_img2.shape[0],
        width=smoothly_predicted_img2.shape[1],
        count=1,
        dtype=smoothly_predicted_img2.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(smoothly_predicted_img2, 1)

    print(f"Saved prediction to {output_filename}")

    metrics = calc_test_accuracy(large_mask_image, smoothly_predicted_img2)
    print(metrics)
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

average_metrics = {key: sum(d[key] for d in all_metrics) / len(all_metrics) for key in all_metrics[0]}
print("Average Metrics:", average_metrics)


image_names = [f"image_{idx}.png" for idx in range(len(large_images_test))]


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
print('Classes:', classes)

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
    
# Create a DataFrame with image names, ratios, and classes
import pandas as pd
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

# Plot each image with its class on top
plt.figure(figsize=(10, 6))
plt.bar(average_metrics.keys(), average_metrics.values(), color=['blue', 'green', 'red', 'purple'])
plt.xlabel("Score")
plt.ylabel('Mean Value')
plt.title("Average Metrics")
plt.ylim(0, 1)
for i, v in enumerate(average_metrics):
    plt.text(i, average_metrics[v] + 0.02, f"{average_metrics[v]:.2f}", ha='center', fontweight='bold')
plt.show()
plt.close()
