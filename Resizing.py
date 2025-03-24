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
from sklearn.metrics import f1_score, precision_score, recall_score, jaccard_score, accuracy_score
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
wandb.init(project="SAM_FineTune")
from collections import Counter
from torch.utils.data import WeightedRandomSampler

# Load tiff  images and masks
img_path=glob.glob("D:/Erosion_signs/training/images/*.tif")
mask_path=glob.glob("D:/Erosion_signs/training/masks/*.tif")
# img_path = img_path[2:10]
# mask_path = mask_path[2:10]
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


plt.tight_layout()
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
def get_point_prompts(ground_truth_map, grid_size=50):
    H, W = ground_truth_map.shape
    x = np.linspace(0, W-1, grid_size)
    y = np.linspace(0, H-1, grid_size)
    xv, yv = np.meshgrid(x, y)
    xv_list = xv.flatten().tolist()
    yv_list = yv.flatten().tolist()
    points = list(zip(xv_list, yv_list))
    return points

"""Get random grid points from masks."""
def get_random_point_prompts(ground_truth_map, num_points=20):
    H, W = ground_truth_map.shape
    points = [(random.randint(0, W-1), random.randint(0, H-1)) for _ in range(num_points)]
    return points


from torch.utils.data import Dataset
from torchvision import transforms
from transformers import SamProcessor

class SAMDataset(Dataset):
    def __init__(self, dataset, processor, resize=None, augment=None):
        self.dataset = dataset
        self.processor = processor
        self.resize = resize
        self.augment = augment

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
        
        
        # # # Generate regular grid of point prompts
        points = get_point_prompts(ground_truth_mask) # Uncomment if you want to use regular grid points as prompts
    
        # prepare image and points for the model
        inputs = self.processor(image, input_points=[points], return_tensors="pt") # Uncomment if you want to use regular grid points as prompts
        
        # # Generate random point prompts
        # points = get_random_point_prompts(ground_truth_mask) # Uncomment if you want to use random points as prompts
    
        # # prepare image and points for the model
        # inputs = self.processor(image, input_points=[points], return_tensors="pt") # Uncomment if you want to use random points as prompts
        
        
        # remove batch dimension which the processor adds by default
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # add ground truth segmentation
        inputs["ground_truth_mask"] = ground_truth_mask

        return inputs

# Initialize the processor
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")


train_dataset = SAMDataset(dataset=train_data, processor=processor, 
                           resize=transforms.Resize([256, 256]), augment=True)  # Data augmentation is only applied to training data

val_dataset = SAMDataset(dataset=val_data, processor=processor,
                           resize=transforms.Resize([256, 256]), augment=False) # Data augmentation is not applied to validation data


print('training dataset', len(train_dataset))
print('validation dataset', len(val_dataset))


# # Create a DataLoader instance for the training dataset
from torch.utils.data import DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, drop_last=False)

val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True, drop_last=False)

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
#  Try DiceCELoss or the combination of DiceLoss and FocalLoss

seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

dice_loss = DiceLoss(sigmoid=True)
focal_loss = FocalLoss(to_onehot_y=False, gamma=2.0)  # single-channel prediction

# Define the ratio
ratio = 19 / 20

# Combine the loss functions with the specified ratio
def combined_loss(outputs, targets):
    return ratio * focal_loss(outputs, targets) + (1 - ratio) * dice_loss(outputs, targets)


# # # Combine the loss functions
# def combined_loss(outputs, targets):
#     return dice_loss(outputs, targets) + focal_loss(outputs, targets)

# Initialize the optimizer and learning rate scheduler
optimizer = torch.optim.Adam(model.mask_decoder.parameters(), lr=1e-3, weight_decay=0)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)


from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, jaccard_score, accuracy_score

def calc_validation_accuracy(large_masks_val, pred):
    y_true = large_masks_val.ravel()
    y_pred = pred.ravel()
    

    return {
        "f1_score_validation": f1_score(y_true, y_pred, zero_division=0),
        "precision_validation": precision_score(y_true, y_pred, zero_division=0),
        "recall_validation": recall_score(y_true, y_pred, zero_division=0),
        "iou_score_validation": jaccard_score(y_true, y_pred, zero_division=0),

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
num_epochs = 100

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
                        # input_boxes=batch["input_boxes"].to(device),
                        input_points=batch["input_points"].to(device),
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
                        # input_boxes=batch["input_boxes"].to(device),
                        input_points=batch["input_points"].to(device),
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
        torch.save(model.state_dict(), "D:/FineTuneSAM/checkpoints/resize400imbalanced100epochs_best.pth")
    
    print(f"Best epoch: {np.argmin(val_losses) + 1}")
    print(f"Best val loss: {min(val_losses)}")

torch.save(model.state_dict(), "D:/FineTuneSAM/checkpoints/resize400imbalanced100epochs_last.pth")
print(f"Training ended at {get_current_time()}")

# Plotting the losses
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
from transformers import SamModel, SamConfig, SamProcessor
import torch
import numpy as np
import rasterio
from PIL import Image
import matplotlib.pyplot as plt
import os
import glob

# Load the model configuration and processor
model_config = SamConfig.from_pretrained("facebook/sam-vit-base")
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

# Create an instance of the model architecture with the loaded configuration
my_mito_model = SamModel(config=model_config)
# Update the model by loading the weights from saved file.
my_mito_model.load_state_dict(torch.load("D:/FineTuneSAM/checkpoints/resize400imbalanced100epochs_best.pth", map_location=torch.device('cpu')))

# Set the device to cuda if available, otherwise use cpu
device = "cuda" if torch.cuda.is_available() else "cpu"
my_mito_model.to(device)


# Paths to test images and masks
img_path_test=glob.glob("D:/Erosion_signs/testing/images/*.tif")
mask_path_test=glob.glob("D:/Erosion_signs/testing/masks/*.tif")
img_path_test.sort()
mask_path_test.sort()

# Load images and masks without resizing
large_images_test = []
large_masks_test = []

for i, j in zip(img_path_test, mask_path_test):
    with rasterio.open(i) as src:
        img_test = src.read([1, 2, 3])
        if img_test.shape[0] == 3:
            img_test = np.transpose(img_test, (1, 2, 0))
        large_images_test.append(img_test)
        
    with rasterio.open(j) as src:
        mask_test = src.read(1)  
        large_masks_test.append(mask_test)

# Convert lists to numpy arrays (keeping dtype='object' to allow for different shapes if needed)
large_images_test = np.array(large_images_test, dtype='object')
large_masks_test = np.array(large_masks_test, dtype='object')

print(f"large_images_test shape: {large_images_test.shape}")
print(f"large_images_test[0] shape: {large_images_test[0].shape}")

# Check shapes of images and masks before patchifying
for idx, (img_test, mask_test) in enumerate(zip(large_images_test, large_masks_test)):
    if img_test.shape[:2] != mask_test.shape:
        print(f"Shape mismatch at index {idx}: image shape {img_test.shape} vs mask shape {mask_test.shape}")


# Generate the grid points for input points
array_size = 256
grid_size = 50
x = np.linspace(0, array_size-1, grid_size)
y = np.linspace(0, array_size-1, grid_size)
xv, yv = np.meshgrid(x, y)
xv_list = xv.tolist()
yv_list = yv.tolist()
input_points = [[[int(x), int(y)] for x, y in zip(x_row, y_row)] for x_row, y_row in zip(xv_list, yv_list)]
input_points = torch.tensor(input_points).view(1, 1, grid_size*grid_size, 2)
print('(batch_size, point_batch_size, num_points_per_image, 2):', np.array(input_points).shape)


from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, jaccard_score, accuracy_score

def calc_test_accuracy(large_masks_test, pred):
    y_true = large_masks_test.ravel()
    y_pred = pred.ravel()
    
    return {
        "f1_score_test": f1_score(y_true, y_pred, zero_division=0),
        "precision_test": precision_score(y_true, y_pred, zero_division=0),
        "recall_test": recall_score(y_true, y_pred, zero_division=0),
        "iou_score_test": jaccard_score(y_true, y_pred, zero_division=0),

    }
    
    
# Placeholder lists for results
all_metrics = []

for k in range(len(large_images_test)):
    large_test_image = large_images_test[k]
    large_mask_image = large_masks_test[k]

    original_shape = large_test_image.shape[:2]

    # Ensure the arrays have a numeric data type
    if large_test_image.dtype != np.uint8:
        large_test_image = large_test_image.astype(np.uint8)
    if large_mask_image.dtype != np.uint8:
        large_mask_image = large_mask_image.astype(np.uint8)

    # Resize images and masks to 256x256
    large_test_image_resized = np.array(Image.fromarray(large_test_image).resize((256, 256), Image.BILINEAR))
    large_mask_image_resized = np.array(Image.fromarray(large_mask_image).resize((256, 256), Image.NEAREST))

    single_patch = Image.fromarray(large_test_image_resized)
    inputs = processor(single_patch, input_points=input_points, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    my_mito_model.eval()

    with torch.no_grad():
        outputs = my_mito_model(**inputs, multimask_output=False)

    single_patch_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
    single_patch_prob = single_patch_prob.cpu().numpy().squeeze()
    pred_resized = (single_patch_prob > 0.5).astype(np.uint8)
    
    
    # Resize prediction and probability maps back to the original image size
    pred = np.array(Image.fromarray(pred_resized).resize(original_shape[::-1], Image.NEAREST))
    probability = np.array(Image.fromarray(single_patch_prob).resize(original_shape[::-1], Image.NEAREST))

    # Plot results
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
    # plt.show()
    plt.close()
    
    # Save the figure with the input filename
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
        count=1,  # Number of bands
        dtype=pred.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(pred, 1)

    print(f"Saved prediction to {output_filename}")
    
    metrics = calc_test_accuracy(large_mask_image, pred)
    print(metrics)
    all_metrics.append(metrics)

average_metrics = {key: sum(d[key] for d in all_metrics) / len(all_metrics) for key in all_metrics[0]}
print("Average Metrics:", average_metrics)


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
