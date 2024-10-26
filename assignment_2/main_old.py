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

# Define the CustomDataset class
class CustomDataset(Dataset):
    def __init__(self, train=True, image_transform=None, mask_transform=None, joint_transform=None, data_folder="DRIVE", split_percentage=80):
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        if data_folder == "DRIVE":
            self.joint_transform = joint_transform if train else transforms.Compose([transforms.CenterCrop((128,128))])
        else:
            self.joint_transform = joint_transform if train else transforms.Compose([transforms.Resize((128,128))])
        self.train = train
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

            # Split into train and test sets
            random.seed(628)
            usefull_idxs = list(range(len(image_dict)))
            random.shuffle(usefull_idxs)
            split_idx = int(split_percentage * len(image_dict) / 100)
            selected_idxs = set(usefull_idxs[:split_idx])

            for i, image_id in enumerate(image_dict.keys()):
                if train and (not i in selected_idxs): continue
                if (not train) and i in selected_idxs: continue
                self.image_paths.append(image_dict[image_id])
                self.mask_paths.append(mask_dict[image_id])

        elif data_folder == "PH2_Dataset_images":
            root_dir = os.path.join(root_path, "PH2_Dataset_images")
            all_folders = sorted(glob.glob(os.path.join(root_dir, "IMD*")))

            # Split into train and test sets
            random.seed(628)
            random.shuffle(all_folders)
            split_idx = int(split_percentage * len(all_folders) / 100)
            selected_folders = all_folders[:split_idx] if train else all_folders[split_idx:]

            for folder in selected_folders:
                imd_id = os.path.basename(folder)
                image_file = os.path.join(folder, f"{imd_id}_Dermoscopic_Image", f"{imd_id}.bmp")
                mask_file = os.path.join(folder, f"{imd_id}_lesion", f"{imd_id}_lesion.bmp")
                self.image_paths.append(image_file)
                self.mask_paths.append(mask_file)
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

def get_stats(dataset):
    image, mask = dataset[0]
    print(f"Image size: {image.shape}, Mask size: {mask.shape}, Dataset length: {len(dataset)}")

# Initialize datasets
image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.ColorJitter(),
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

    train_dataset = CustomDataset(train=True, image_transform=image_transform, mask_transform=mask_transform, joint_transform=joint_transform, data_folder=config.dataset)
    test_dataset = CustomDataset(train=False, image_transform=image_transform, mask_transform=mask_transform, data_folder=config.dataset)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=SHUFFLE, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=NUM_WORKERS)

    return train_loader, test_loader, train_dataset, test_dataset

# Define the UNet model with dropout
class UNet(nn.Module):
    def __init__(self, num_layers=4, base_channels=32, dropout=0.0):
        super().__init__()
        self.dropout = dropout

        # Input convolution
        self.inp = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, padding=1, padding_mode="reflect"),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # Encoder (downsampling)
        self.encoder_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(base_channels * 2**i, base_channels * 2**(i+1), 3, padding=1, stride=2, padding_mode="reflect"),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            )
            for i in range(num_layers)
        ])

        # Bottleneck
        self.bottleneck_conv = nn.Sequential(
            nn.Conv2d(base_channels * 2**num_layers, base_channels * 2**num_layers, 3, padding=1, padding_mode="reflect"),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # Decoder (upsampling)
        self.decoder_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(base_channels * 2**(num_layers - i) * 2, base_channels * 2**(num_layers - i - 1), 3, padding=1, padding_mode="reflect"),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            )
            for i in range(num_layers)
        ])

        # Output convolution
        self.out = nn.Conv2d(base_channels, 1, 3, padding=1, padding_mode="reflect")

    def forward(self, x):
        # Input
        x = self.inp(x)

        # Encoder
        encoder_outs = []
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
            encoder_outs.append(x)

        # Bottleneck
        x = self.bottleneck_conv(encoder_outs[-1])

        # Decoder
        for i, decoder_block in enumerate(self.decoder_blocks):
            x = decoder_block(torch.cat([x, encoder_outs[-1 - i]], dim=1))

        # Output
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

# For WandB purposes
lossmap = {"bce": bce_loss, "dice": dice_loss, "mixed": mixed_loss, "focal": focal_loss}

# Define the compute_metrics function
def compute_metrics(preds, targets):
    preds = preds.view(-1)
    targets = targets.view(-1)
    TP = ((preds == 1) & (targets == 1)).sum().float()
    TN = ((preds == 0) & (targets == 0)).sum().float()
    FP = ((preds == 1) & (targets == 0)).sum().float()
    FN = ((preds == 0) & (targets == 1)).sum().float()

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    dice = 2 * TP / (2 * TP + FP + FN + 1e-8)
    iou = TP / (TP + FP + FN + 1e-8)  # Compute IoU here

    return precision.item(), recall.item(), f1.item(), dice.item(), iou.item()

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
def train(model, optimizer, train_loader, test_loader, trainset, testset, criterion, num_epochs=10, run_id=""):
    out_dict = {
        'train_acc': [],
        'test_acc': [],
        'train_loss': [],
        'test_loss': [],
        'train_precision': [],
        'test_precision': [],
        'train_recall': [],
        'test_recall': [],
        'train_f1': [],
        'test_f1': [],
        'train_dice': [],
        'test_dice': []
    }

    for epoch in range(num_epochs):
        model.train()
        train_correct = 0
        train_loss = []
        train_preds = []
        train_targets = []

        # Training phase
        for minibatch_no, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(data)

            # Compute loss
            loss = criterion(output, target)

            # Backward pass and update
            loss.backward()
            optimizer.step()

            # Track training loss
            train_loss.append(loss.item())

            # Predictions and accuracy
            predicted = (torch.sigmoid(output) >= 0.5).long()
            train_correct += (target.long() == predicted).sum().cpu().item()

            # Collect predictions and targets
            train_preds.append(predicted.cpu())
            train_targets.append(target.long().cpu())

        # Compute metrics on training data
        train_preds = torch.cat(train_preds)
        train_targets = torch.cat(train_targets)
        train_precision, train_recall, train_f1, train_dice, train_iou = compute_metrics(train_preds, train_targets)

        # Testing phase
        test_loss = []
        test_correct = 0
        test_preds = []
        test_targets = []
        model.eval()

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)

                # Track test loss
                test_loss.append(loss.item())

                # Predictions and accuracy
                predicted = (torch.sigmoid(output) >= 0.5).long()
                test_correct += (target.long() == predicted).sum().cpu().item()

                # Collect predictions and targets
                test_preds.append(predicted.cpu())
                test_targets.append(target.long().cpu())

        # Compute metrics on test data
        test_preds = torch.cat(test_preds)
        test_targets = torch.cat(test_targets)
        test_precision, test_recall, test_f1, test_dice, test_iou = compute_metrics(test_preds, test_targets)

        # Record statistics
        out_dict['train_acc'].append(train_correct / (len(trainset) * target.numel()))
        out_dict['test_acc'].append(test_correct / (len(testset) * target.numel()))
        out_dict['train_loss'].append(np.mean(train_loss))
        out_dict['test_loss'].append(np.mean(test_loss))
        out_dict['train_precision'].append(train_precision)
        out_dict['test_precision'].append(test_precision)
        out_dict['train_recall'].append(train_recall)
        out_dict['test_recall'].append(test_recall)
        out_dict['train_f1'].append(train_f1)
        out_dict['test_f1'].append(test_f1)
        out_dict['train_dice'].append(train_dice)
        out_dict['test_dice'].append(test_dice)
        out_dict['train_iou'].append(train_iou)
        out_dict['test_iou'].append(test_iou)

        # Log to WandB
        wandb.log({
            "train_acc": out_dict['train_acc'][-1],
            "test_acc": out_dict['test_acc'][-1],
            "train_loss": out_dict['train_loss'][-1],
            "test_loss": out_dict['test_loss'][-1],
            "train_precision": out_dict['train_precision'][-1],
            "test_precision": out_dict['test_precision'][-1],
            "train_recall": out_dict['train_recall'][-1],
            "test_recall": out_dict['test_recall'][-1],
            "train_f1": out_dict['train_f1'][-1],
            "test_f1": out_dict['test_f1'][-1],
            "train_dice": out_dict['train_dice'][-1],
            "test_dice": out_dict['test_dice'][-1],
            "train_iou": out_dict['train_iou'][-1],
            "test_iou": out_dict['test_iou'][-1],
            "epoch": epoch,
            "run_id": run_id
        })

        # # Saves the model
        # save_path = f"./checkpoints/epoch_{epoch}_{run_id}.pth"
        # save_checkpoint(model, optimizer, epoch, out_dict, save_path)

        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss: {out_dict['train_loss'][-1]:.3f}, "
              f"Test Loss: {out_dict['test_loss'][-1]:.3f}, "
              f"Train Acc: {out_dict['train_acc'][-1]*100:.2f}%, "
              f"Test Acc: {out_dict['test_acc'][-1]*100:.2f}%, "
              f"Train Dice: {out_dict['train_dice'][-1]:.3f}, "
              f"Test Dice: {out_dict['test_dice'][-1]:.3f}")

    return out_dict

# Update the sweep configuration
sweep_config = {
    'method': 'grid',
    'metric': {'name': 'test_loss', 'goal': 'minimize'},
    'parameters': {
        'learning_rate': {'values': [1e-4, 1e-3, 1e-2]},
        'batch_size': {'values': [1, 2, 4]},
        'num_layers': {'values': [3, 4, 5, 6]},
        'base_channels': {'values': [8, 16, 32]},
        'dropout': {'values': [0.0, 0.2, 0.5]},
        'loss_function': {'values': ['bce', 'dice', 'mixed', 'focal']},
        'dataset': {'values': ['DRIVE', 'PH2_Dataset_images']},
        'epochs': {'value': 1000},
        'image_size': {'value': 128}
    }
}

# Initialize the sweep
sweep_id = wandb.sweep(sweep_config, project='SegmentationProject')

# Update the run_wandb function
def run_wandb(config=None):
    with wandb.init(config=config):
        config = wandb.config

        # Include the run id
        run_id = wandb.run.id
        config.run_id = run_id
        wandb.run.name = f"Run {run_id}"

        # Build datasets and dataloaders
        train_loader, test_loader, train_dataset, test_dataset = build_datasets(config)

        # Build model
        model = UNet(num_layers=config.num_layers, base_channels=config.base_channels, dropout=config.dropout).to(device)

        # Build optimizer
        optimizer = build_optimizer(model, "adam", config.learning_rate)

        # Choose loss function
        criterion = lossmap[config.loss_function]

        # Run training
        out_dict = train(model, optimizer, train_loader, test_loader, train_dataset, test_dataset, criterion, num_epochs=config.epochs, run_id=run_id)

        # Save the model
        os.makedirs("./models", exist_ok=True)
        model_path = f"./models/model_{run_id}.pth"
        torch.save(model.state_dict(), model_path)
        wandb.save(model_path)

# Run the sweep
wandb.agent(sweep_id, run_wandb)
