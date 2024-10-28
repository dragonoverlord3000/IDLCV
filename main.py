import os
import numpy as np
import glob
import PIL.Image as Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random
import wandb

# Log in to WandB
wandb.login(key="4aaf96e30165bfe476963bc860d96770512c8060")

if torch.cuda.is_available():
    print("The code will run on GPU.")
else:
    print("The code will run on CPU.")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ########### #
# Datasets    #
# ########### #
root_path = "/dtu/datasets1/02516/"
dataset_folders = ["DRIVE", "PH2_Dataset_images"]

EPOCHS = 200
LEARNING_RATE = 5e-5
BATCH_SIZE = 4
NUM_WORKERS = 1
NUM_LAYERS = 6
SHUFFLE = True

flag_drive_val = True
flag_ph2_val = True
flag_drive_test = True
flag_ph2_test = True

# Define the CustomDataset class
class CustomDataset(Dataset):
    def __init__(self, split='train', image_transform=None, mask_transform=None, joint_transform=None, data_folder="DRIVE", split_percentages=(68,16,16), image_size=128):
        global flag_drive_val, flag_ph2_val, flag_drive_test, flag_ph2_test
        
        self.image_transform = image_transform if "train" else transforms.Compose([transforms.Normalize(mean=mean, std=std)])
        self.mask_transform = mask_transform
        if data_folder == "DRIVE":
            self.joint_transform = joint_transform if split == 'train' else transforms.Compose([transforms.RandomCrop((image_size,image_size))])
        else:
            self.joint_transform = joint_transform if split == 'train' else transforms.Compose([transforms.Resize((image_size,image_size))])
        self.split = split
        self.data_folder = data_folder
        self.image_paths = []
        self.mask_paths = []
    
        if data_folder == "DRIVE":
            root_dir = os.path.join(root_path, "DRIVE")
            data_dir = "training"
            images_dir = os.path.join(root_dir, data_dir, "images")
            masks_dir = os.path.join(root_dir, data_dir, "1st_manual")
    
            image_files = sorted(glob.glob(os.path.join(images_dir, "*.tif")))
            mask_files = sorted(glob.glob(os.path.join(masks_dir, "*.gif")))
    
            # Mapping from image IDs to file paths
            image_dict = {}
            for f in image_files:
                basename = os.path.basename(f)
                image_id = basename[:2]
                image_dict[image_id] = f
    
            mask_dict = {}
            for f in mask_files:
                basename = os.path.basename(f)
                image_id = basename[:2]
                mask_dict[image_id] = f
    
            # Split into train, val, and test sets
            random.seed(628)
            num_samples = len(image_dict)
            indices = list(range(num_samples))
            random.shuffle(indices)
    
            train_perc, val_perc, test_perc = split_percentages
            train_end = int(train_perc * num_samples / 100)
            val_end = train_end + int(val_perc * num_samples / 100)
    
            train_idxs = set(indices[:train_end])
            val_idxs = set(indices[train_end:val_end])
            test_idxs = set(indices[val_end:])
    
            for i, image_id in enumerate(image_dict.keys()):
                if self.split == 'train' and i not in train_idxs:
                    continue
                if self.split == 'val' and i not in val_idxs:
                    continue
                if self.split == 'test' and i not in test_idxs:
                    continue
                self.image_paths.append(image_dict[image_id])
                self.mask_paths.append(mask_dict[image_id])
    
            if flag_drive_val and split == "val":
                print("Drive val ", self.image_paths)
                flag_drive_val = False
            if flag_drive_test and split == "test":
                print("Drive test ", self.image_paths)
                flag_drive_test = False

        elif data_folder == "PH2_Dataset_images":
            root_dir = os.path.join(root_path, "PH2_Dataset_images")
            all_folders = sorted(glob.glob(os.path.join(root_dir, "IMD*")))
    
            # Split into train, val, and test sets
            random.seed(628)
            num_samples = len(all_folders)
            indices = list(range(num_samples))
            random.shuffle(indices)
    
            train_perc, val_perc, test_perc = split_percentages
            train_end = int(train_perc * num_samples / 100)
            val_end = train_end + int(val_perc * num_samples / 100)
    
            train_idxs = set(indices[:train_end])
            val_idxs = set(indices[train_end:val_end])
            test_idxs = set(indices[val_end:])
    
            for idx in indices:
                folder = all_folders[idx]
                if self.split == 'train' and idx not in train_idxs:
                    continue
                if self.split == 'val' and idx not in val_idxs:
                    continue
                if self.split == 'test' and idx not in test_idxs:
                    continue
                imd_id = os.path.basename(folder)
                image_file = os.path.join(folder, f"{imd_id}_Dermoscopic_Image", f"{imd_id}.bmp")
                mask_file = os.path.join(folder, f"{imd_id}_lesion", f"{imd_id}_lesion.bmp")
                self.image_paths.append(image_file)
                self.mask_paths.append(mask_file)

            if flag_ph2_val and split == "val":
                print("PH2 val ", self.image_paths)
                flag_ph2_val = False
            if flag_ph2_test and split == "test":
                print("PH2 test ", self.image_paths)
                flag_ph2_test = False

        else:
            raise ValueError(f"Dataset {data_folder} not found.")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")
    
        # Apply transformations
        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
    
        combined = torch.cat([image, mask], dim=0)
        if self.joint_transform:
            combined = self.joint_transform(combined)
        image = combined[:3, :, :]
        mask = combined[3:, :, :]
        return image, mask

# Define helper functions
def unnormalize(tensor, mean, std):
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def visualizer(dataset, num_samples=4, title="SBS-visualization", folder="./vis"):
    plt.figure(figsize=(10, 20))
    for i in range(num_samples):
        idx = i
        image, mask = dataset[idx]

        plt.subplot(num_samples, 2, i*2 + 1)
        plt.imshow(transforms.ToPILImage()(unnormalize(image, mean, std)))
        plt.title(f"Image {i+1}")
        plt.axis("off")

        plt.subplot(num_samples, 2, i*2 + 2)
        plt.imshow(transforms.ToPILImage()(mask), cmap="gray")
        plt.title(f"Mask {i+1}")
        plt.axis("off")

    plt.tight_layout()
    os.makedirs(folder, exist_ok=True)
    plt.savefig(f"{folder}/{title}.png")
    plt.close()

# Plot images, predictions and their masks - to show dataset
def visualizer_train(dataset, model, num_samples=4, title="Final_Validation_Visualization", run_id=""):
    """Visualize predictions on the validation set and save them with the given run_id."""
    model.eval()
    dataset_name = dataset.data_folder  # Use dataset name for directory structure
    num_samples = min(num_samples, len(dataset))  # Limit num_samples to available data
    plt.figure(figsize=(10, 20))
    os.makedirs(f"./vis/{dataset_name}/{run_id}", exist_ok=True)

    with torch.no_grad():
        for i in range(num_samples):
            image, mask = dataset[i]
            image = image.to(device).unsqueeze(0)
            pred = torch.sigmoid(model(image)).cpu().squeeze(0)

            # Plot the image
            plt.subplot(num_samples, 3, i*3 + 1)
            plt.imshow(transforms.ToPILImage()(unnormalize(image.squeeze(0), mean, std)))
            plt.title(f"Image {i+1}")
            plt.axis("off")

            # Plot the prediction
            plt.subplot(num_samples, 3, i*3 + 2)
            plt.imshow(pred.squeeze(0), cmap="gray")
            plt.title(f"Prediction {i+1}")
            plt.axis("off")

            # Plot the mask
            plt.subplot(num_samples, 3, i*3 + 3)
            plt.imshow(transforms.ToPILImage()(mask), cmap="gray")
            plt.title(f"Mask {i+1}")
            plt.axis("off")

    plt.tight_layout()
    plt.savefig(f"./vis/{dataset_name}/{run_id}/{title}.png")
    plt.close()

def get_stats(dataset):
    image, mask = dataset[0]
    print(f"Image size: {image.shape}, Mask size: {mask.shape}, Dataset length: {len(dataset)}")

# Initialize datasets
image_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.ColorJitter(),
    transforms.Normalize(mean=mean, std=std),
])

mask_transform = transforms.Compose([
    transforms.ToTensor()
])

# Define joint transforms with image size parameter
def get_joint_transforms(config):
    if config.dataset == 'DRIVE':
        joint_transform = transforms.Compose([
            transforms.RandomCrop((config.image_size, config.image_size)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomRotation(30)
        ])
    else:
        joint_transform = transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomRotation(30)
        ])
    return joint_transform

# Update build_datasets function
def build_datasets(config):
    joint_transform = get_joint_transforms(config)

    train_dataset = CustomDataset(split='train', image_transform=image_transform, mask_transform=mask_transform, joint_transform=joint_transform, data_folder=config.dataset, image_size=config.image_size)
    val_dataset = CustomDataset(split='val', image_transform=image_transform, mask_transform=mask_transform, data_folder=config.dataset, image_size=config.image_size)
    test_dataset = CustomDataset(split='test', image_transform=image_transform, mask_transform=mask_transform, data_folder=config.dataset, image_size=config.image_size)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=SHUFFLE, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=NUM_WORKERS)

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset

class UNet(nn.Module):
    def __init__(self, num_layers=4, base_channels=32, dropout=0.0):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers

        # Input convolution
        self.inp = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, padding=1, padding_mode="reflect"),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # Encoder (downsampling)
        self.encoder_blocks = nn.ModuleList()
        for i in range(num_layers):
            in_channels = base_channels * 2**i
            out_channels = base_channels * 2**(i+1)
            block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=2, padding_mode="reflect"),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, padding_mode="reflect"),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            )
            self.encoder_blocks.append(block)

        # Bottleneck
        bottleneck_channels = base_channels * 2**num_layers
        self.bottleneck_conv = nn.Sequential(
            nn.Conv2d(bottleneck_channels, bottleneck_channels, 3, padding=1, padding_mode="reflect"),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Decoder (upsampling)
        self.decoder_blocks = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                in_channels = bottleneck_channels + base_channels * 2**(num_layers - 1)
                out_channels = base_channels * 2**(num_layers - 1)
            elif i < num_layers - 1:
                in_channels = base_channels * 2**(num_layers - i) + base_channels * 2**(num_layers - i - 1)
                out_channels = base_channels * 2**(num_layers - i - 1)
            else:  # Last decoder block
                in_channels = base_channels * 2 + base_channels + 3  # x channels + encoder_outs[0] + x_input
                out_channels = base_channels
            block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1, padding_mode="reflect"),
                nn.Identity() if i == num_layers - 1 else nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, padding_mode="reflect"),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            )
            self.decoder_blocks.append(block)

        # Output convolution
        self.out = nn.Conv2d(base_channels, 1, 3, padding=1, padding_mode="reflect")

    def forward(self, x):
        x_input = x  # Store the input for skip connection

        # Input convolution
        x = self.inp(x)
        encoder_outs = [x]

        # Encoder
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
            encoder_outs.append(x)

        # Bottleneck
        x = self.bottleneck_conv(encoder_outs[-1])
        x = self.up(x)  # Upsample bottleneck output

        # Decoder
        for i, decoder_block in enumerate(self.decoder_blocks):
            if i < self.num_layers - 1:
                # Concatenate with corresponding encoder output
                x = decoder_block(torch.cat([x, encoder_outs[-(i+2)]], dim=1))
            else:
                # Last decoder block: concatenate with initial encoder output and input
                x = decoder_block(torch.cat([x, encoder_outs[0], x_input], dim=1))

        # Output convolution
        out = self.out(x)
        return out

# Define the build_optimizer function
def build_optimizer(model, optimizer_name, learning_rate):
    if optimizer_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    return optimizer

# Define loss functions
def bce_loss(y_pred, y_real):
    return F.binary_cross_entropy_with_logits(y_pred, y_real)

def dice_loss(y_pred, y_real, epsilon=1e-6):
    y_pred = torch.sigmoid(y_pred)
    y_real_flat = y_real.view(-1)
    y_pred_flat = y_pred.view(-1)

    intersection = 2 * (y_real_flat * y_pred_flat).sum() + epsilon
    denominator = y_real_flat.sum() + y_pred_flat.sum() + epsilon
    return 1 - (intersection / denominator)

def focal_loss(y_pred, y_real):
    gamma = 2
    y_hat = torch.sigmoid(y_pred)
    focal = -torch.sum((1 - y_hat) ** gamma * y_real * torch.log(y_hat + 1e-8) + (y_hat ** gamma) * (1 - y_real) * torch.log(1 - y_hat + 1e-8))
    return focal / y_real.numel()

def mixed_loss(y_pred, y_real):
    return bce_loss(y_pred, y_real) + dice_loss(y_pred, y_real)

def calculate_class_weights(dataset):
    total_pixels = 0
    foreground_pixels = 0
    background_pixels = 0

    for _, mask in tqdm(dataset, desc="Calculating class weights"):
        mask = mask > 0.5  # Assume mask is binary (foreground = 1, background = 0)
        foreground_pixels += mask.sum().item()
        background_pixels += mask.numel() - mask.sum().item()
        total_pixels += mask.numel()

    # Calculate weights: higher weight to the less frequent class
    weight_foreground = total_pixels / (2 * foreground_pixels)
    weight_background = total_pixels / (2 * background_pixels)
    return weight_foreground, weight_background

# Function to set criterion based on dataset type
def get_weighted_bce_loss(train_loader, dataset_name):
    weight_foreground, weight_background = calculate_class_weights(train_loader.dataset)
    
    def weighted_bce_loss(y_pred, y_real):
        # Assign weights based on the foreground and background classes
        weight_map = torch.where(y_real > 0.5, weight_foreground, weight_background)
        loss = F.binary_cross_entropy_with_logits(y_pred, y_real, weight=weight_map)
        return loss

    return weighted_bce_loss

# For WandB purposes
lossmap = {"bce": bce_loss, "dice": dice_loss, "mixed": mixed_loss, "focal": focal_loss}

# Define the compute_metrics function
def compute_metrics(preds, targets):
    thresholds = np.arange(0.1, 1.1, 0.1)
    
    best_dice = 0
    best_iou = 0
    best_acc = 0
    best_sensitivity = 0
    best_specificity = 0

    for threshold in thresholds:
        thresholded_preds = [(pred > threshold).float() for pred in preds]

        # Flatten predictions and targets to compute pixel-level metrics
        flat_preds = torch.cat([p.view(-1) for p in thresholded_preds])
        flat_targets = torch.cat([t.view(-1) for t in targets])

        # True positives, false positives, true negatives, false negatives
        TP = (flat_preds * flat_targets).sum().item()
        FP = (flat_preds * (1 - flat_targets)).sum().item()
        FN = ((1 - flat_preds) * flat_targets).sum().item()
        TN = ((1 - flat_preds) * (1 - flat_targets)).sum().item()

        # Compute each metric
        dice = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0
        iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
        acc = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) > 0 else 0
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

        # Keep track of the best values for each metric
        best_dice = max(best_dice, dice)
        best_iou = max(best_iou, iou)
        best_acc = max(best_acc, acc)
        best_sensitivity = max(best_sensitivity, sensitivity)
        best_specificity = max(best_specificity, specificity)

    return best_dice, best_iou, best_acc, best_sensitivity, best_specificity

# Define the save_checkpoint function
def save_checkpoint(model, optimizer, epoch, out_dict, path='model_checkpoint.pth'):
    checkpoint = {
        'epoch': epoch + 1,  # Save the next epoch number
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': out_dict
    }
    torch.save(checkpoint, path)
    print(f"Model checkpoint saved at epoch {epoch + 1}, name: {path}")

# Ensure the checkpoints directory exists
os.makedirs("checkpoints", exist_ok=True)

# Update the train function
def train(model, optimizer, train_loader, val_loader, test_loader, criterion, num_epochs=10, run_id=""):
    out_dict = {
        'train_dice': [],
        'val_dice': [],
        "test_dice": [],
        'train_iou': [],
        'val_iou': [],
        "test_iou": [],
        'train_acc': [],
        'val_acc': [],
        "test_acc": [],
        'train_sensitivity': [],
        'val_sensitivity': [],
        "test_sensitivity": [],
        'train_specificity': [],
        'val_specificity': [],
        "test_specificity": [],
        'train_loss': [],
        'val_loss': [],
        "test_loss": []
    }

    for epoch in range(num_epochs):
        model.train()
        train_loss = []
        train_preds = []
        train_targets = []

        # Training phase
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            train_preds.append(torch.sigmoid(output).cpu())
            train_targets.append(target.long().cpu())

        # Compute training metrics
        train_dice, train_iou, train_acc, train_sensitivity, train_specificity = compute_metrics(train_preds, train_targets)

        # Validation phase
        val_loss = []
        val_preds = []
        val_targets = []
        model.eval()

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)

                val_loss.append(loss.item())
                val_preds.append(torch.sigmoid(output).cpu())
                val_targets.append(target.long().cpu())

        # Compute validation metrics
        val_dice, val_iou, val_acc, val_sensitivity, val_specificity = compute_metrics(val_preds, val_targets)

        # Test phase
        test_preds = []
        test_targets = []
        model.eval()

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)

                test_preds.append(torch.sigmoid(output).cpu())
                test_targets.append(target.long().cpu())

        # Compute validation metrics
        test_dice, test_iou, test_acc, test_sensitivity, test_specificity = compute_metrics(test_preds, test_targets)

        # Record metrics
        out_dict['train_loss'].append(np.mean(train_loss))
        out_dict['val_loss'].append(np.mean(val_loss))

        out_dict['train_dice'].append(train_dice)
        out_dict['val_dice'].append(val_dice)
        out_dict['test_dice'].append(test_dice)

        out_dict['train_iou'].append(train_iou)
        out_dict['val_iou'].append(val_iou)
        out_dict['test_iou'].append(test_iou)

        out_dict['train_acc'].append(train_acc)
        out_dict['val_acc'].append(val_acc)
        out_dict['test_acc'].append(test_acc)

        out_dict['train_sensitivity'].append(train_sensitivity)
        out_dict['val_sensitivity'].append(val_sensitivity)
        out_dict['test_sensitivity'].append(test_sensitivity)

        out_dict['train_specificity'].append(train_specificity)
        out_dict['val_specificity'].append(val_specificity)
        out_dict['test_specificity'].append(test_specificity)


        # Log to WandB
        wandb.log({
            "val_dice": val_dice,
            "val_iou": val_iou,
            "val_acc": val_acc,
            "val_sensitivity": val_sensitivity,
            "val_specificity": val_specificity,
            "train_dice": train_dice,
            "train_iou": train_iou,
            "train_acc": train_acc,
            "train_sensitivity": train_sensitivity,
            "train_specificity": train_specificity,
            "test_dice": test_dice,
            "test_iou": test_iou,
            "test_acc": test_acc,
            "test_sensitivity": test_sensitivity,
            "test_specificity": test_specificity,
            "epoch": epoch,
            "run_id": run_id
        })

        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss: {out_dict['train_loss'][-1]:.3f}, "
              f"Val Loss: {out_dict['val_loss'][-1]:.3f}, "
              f"Train Acc: {train_acc*100:.2f}%, "
              f"Val Acc: {val_acc*100:.2f}%, "
              f"Test Acc: {test_acc*100:.2f}%, "
              f"Train Dice: {train_dice:.3f}, "
              f"Val Dice: {val_dice:.3f}, "
              f"Test Dice: {test_dice:.3f}, "
              f"Train IoU: {train_iou:.3f}, "
              f"Val IoU: {val_iou:.3f}, "
              f"Test IoU: {test_iou:.3f}")

    # Final validation visualization
    visualizer_train(val_loader.dataset, model, num_samples=2, title="Final_Validation_Visualization", run_id=run_id)
    save_checkpoint(model, optimizer, num_epochs, out_dict, f"./models/{run_id}.pth")
    return out_dict

# Update the sweep configuration for the DRIVE dataset
sweep_config_drive = {
    'method': 'bayes',
    'metric': {'name': 'val_loss', 'goal': 'minimize'},
    'parameters': {
        'learning_rate': {'values': [1e-3, 1e-4]},
        'batch_size': {'value': 4},
        'num_layers': {'values': [3, 4, 5, 6]},
        'base_channels': {'value': 16},
        "epochs": {"value": 200},
        "dropout": {"values": [0.0, 0.2]},
        'loss_function': {'values': ['bce', 'dice', 'mixed', 'focal', "weighted_drive"]},
        'dataset': {'value': 'DRIVE'}, 
        'image_size': {"values": [128, 256]}
    },
    'early_terminate': {
        'type': 'hyperband',
        's': 2,
        'eta': 3,
        'max_iter': 200
    }
}

# Update the sweep configuration for the PH2 dataset
sweep_config_ph2 = {
    'method': 'bayes',
    'metric': {'name': 'val_loss', 'goal': 'minimize'},
    'parameters': {
        'learning_rate': {'values': [1e-3, 1e-4]},
        'batch_size': {'value': 4},
        'num_layers': {'values': [3, 4, 5, 6]},
        'base_channels': {'value': 16},
        "epochs": {"value": 40},
        "dropout": {"values": [0.0, 0.2]},
        'loss_function': {'values': ['bce', 'dice', 'mixed', 'focal', "weighted_ph2"]},
        'dataset': {'value': 'PH2_Dataset_images'}, 
        'image_size': {'values': [128, 256]}
    },
    'early_terminate': {
        'type': 'hyperband',
        's': 2,
        'eta': 3,
        'max_iter': 40
    }
}

# Update the run_wandb function
def run_wandb(config=None):
    with wandb.init(config=config):
        config = wandb.config

        # Include the run id
        run_id = wandb.run.id
        config.run_id = run_id
        wandb.run.name = f"Run {run_id}"

        # Build datasets and dataloaders
        train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = build_datasets(config)

        # Build model
        model = UNet(num_layers=config.num_layers, base_channels=config.base_channels, dropout=config.dropout).to(device)
        print(model)

        # Build optimizer
        optimizer = build_optimizer(model, "adam", config.learning_rate)

        # Choose loss function
        criterion = None
        if config.loss_function in lossmap:
            criterion = lossmap[config.loss_function]
        elif config.loss_function == "weighted_drive":
            criterion = get_weighted_bce_loss(train_loader, "DRIVE")
        elif config.loss_function == "weighted_ph2":
            criterion = get_weighted_bce_loss(train_loader, "PH2")
            
        # Run training
        out_dict = train(model, optimizer, train_loader, val_loader, test_loader, criterion, num_epochs=config.epochs, run_id=run_id)

        # Save the model
        os.makedirs("./models", exist_ok=True)
        model_path = f"./models/model_{run_id}.pth"
        torch.save(model.state_dict(), model_path)
        wandb.save(model_path)


# # Initialize the sweep for the DRIVE dataset
sweep_id_drive = wandb.sweep(sweep_config_drive, project='SegmentationProject_DRIVE')
# Initialize the sweep for the PH2 dataset
# sweep_id_ph2 = wandb.sweep(sweep_config_ph2, project='SegmentationProject_PH2')

# Run the sweeps
wandb.agent(sweep_id_drive, run_wandb)
# wandb.agent(sweep_id_ph2, run_wandb)


