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
# import wandb
# wandb.login(key="4aaf96e30165bfe476963bc860d96770512c8060")
from time import time
import os

if torch.cuda.is_available():
    print("The code will run on GPU.")
else:
    print("The code will run on CPU. Go to Edit->Notebook Settings and choose GPU as the hardware accelerator")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ########### #
# Datasets :) #
# ########### #
root_path = "/dtu/datasets1/02516/"
dataset_folders = ["DRIVE", "PH2_Dataset_images"]

EPOCHS = 20
BATCH_SIZE = 4
NUM_WORKERS = 1
SHUFFLE = True


class CustomDataset(Dataset):
    def __init__(self, train=True, image_transform=None, mask_transform=None, joint_transform=None, data_folder="DRIVE", split_percentage=80):
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        if data_folder == "DRIVE":
            self.joint_transform = joint_transform if train else transforms.Compose([transforms.RandomCrop((128,128))])
        else:
            self.joint_transform = joint_transform if train else transforms.Compose([transforms.Resize((128,128))])
        self.train = train
        self.data_folder = data_folder
        self.image_paths = []
        self.mask_paths = []

        if data_folder == "DRIVE":
            root_dir = os.path.join(root_path, "DRIVE")
            data_dir = "training" # if train else "test"
            images_dir = os.path.join(root_dir, data_dir, "images")
            masks_dir = os.path.join(root_dir, data_dir, "1st_manual") # 1st_manual is the ritnal one

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

            # Split into train and test sets (split_percentage% train, (100 - split_percentage)% test)
            random.seed(628)  # So that we can reproduce 
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

            # Split into train and test sets (split_percentage% train, (100 - split_percentage)% test)
            random.seed(628)  # So that we can reproduce 
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
        image = Image.open(self.image_paths[idx]).convert("RGB")  # Color
        mask = Image.open(self.mask_paths[idx]).convert("L")  # Grayscale

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

def unnormalize(tensor, mean, std):
    # Clone the tensor to avoid modifying the original one
    tensor = tensor.clone()
    
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    
    return tensor

# Example usage with your specific mean and std
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


# Plot images and their masks - to show dataset
def visualizer(dataset, num_samples=4, title="SBS-visualization", folder="./vis"):
    plt.figure(figsize=(10, 20))
    for i in range(num_samples):
        idx = i
        image, mask = dataset[idx]
        
        # Plot the image
        plt.subplot(num_samples, 2, i*2 + 1)
        plt.imshow(transforms.ToPILImage()(unnormalize(image, mean, std)))
        plt.title(f"Image {i+1}")
        plt.axis("off")

        # Plot the mask
        plt.subplot(num_samples, 2, i*2 + 2)
        plt.imshow(transforms.ToPILImage()(mask), cmap="gray")
        plt.title(f"Mask {i+1}")
        plt.axis("off")
        
    plt.tight_layout()
    plt.savefig(f"{folder}/{title}.png")
    plt.close()

def get_stats(dataset):
    image, mask = dataset[0]
    print(f"Image size: {image.shape}, Mask size: {mask.shape}, Dataset length: {len(dataset)}")

# Init datasets
image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

mask_transform = transforms.Compose([
    transforms.ToTensor()
])

########
# TODO #
########
# Hyperparameter - augmentation transforms
# Training test distinction


joint_transform_drive = transforms.Compose([
    transforms.RandomCrop((128,128)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.RandomRotation(30)
])

joint_transform_ph2 = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.RandomRotation(30)
])


# DRIVE
drive_dataset_before = CustomDataset(train=True, image_transform=image_transform, mask_transform=mask_transform, data_folder="DRIVE")
drive_dataset_train = CustomDataset(train=True, image_transform=image_transform, mask_transform=mask_transform, joint_transform=joint_transform_drive, data_folder="DRIVE")
drive_dataset_test = CustomDataset(train=False, image_transform=image_transform, mask_transform=mask_transform, data_folder="DRIVE")
visualizer(drive_dataset_before, 4, title="DRIVE_train_before")
visualizer(drive_dataset_train, 4, title="DRIVE_train_after")
visualizer(drive_dataset_test, 4, title="DRIVE_test_after")

print("#### DRIVE stats ####")
get_stats(drive_dataset_train)
get_stats(drive_dataset_test)

# PH2
ph2_dataset_before = CustomDataset(train=True, image_transform=image_transform, mask_transform=mask_transform, data_folder="PH2_Dataset_images")
ph2_dataset_train = CustomDataset(train=True, image_transform=image_transform, mask_transform=mask_transform, joint_transform=joint_transform_ph2, data_folder="PH2_Dataset_images")
ph2_dataset_test = CustomDataset(train=False, image_transform=image_transform, mask_transform=mask_transform, data_folder="PH2_Dataset_images")
visualizer(ph2_dataset_before, 4, title="PH2_train_before")
visualizer(ph2_dataset_train, 4, title="PH2_train_after")
visualizer(ph2_dataset_test, 4, title="PH2_test_after")

print("#### PH2 stats ####")
get_stats(ph2_dataset_train)
get_stats(ph2_dataset_test)


drive_train_dataloader = DataLoader(drive_dataset_train, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS)
ph2_train_dataloader = DataLoader(ph2_dataset_train, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS)
drive_test_dataloader = DataLoader(drive_dataset_test, batch_size=1, num_workers=NUM_WORKERS)
ph2_test_dataloader = DataLoader(ph2_dataset_test, batch_size=1, num_workers=NUM_WORKERS)


class ValidationMetrics:
    def __init__(self, t):
        self.t = t
    
    def __call__(self, image, mask):
        pass


class EncoderDecoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass



class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # encoder (downsampling)
        self.enc_conv0 = nn.Conv2d(3, 64, 3, padding=1)
        self.pool0 = nn.MaxPool2d(2, 2)  # 128 -> 64
        self.enc_conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)  # 64 -> 32
        self.enc_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # 32 -> 16
        self.enc_conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)  # 16 -> 8

        # bottleneck
        self.bottleneck_conv = nn.Conv2d(64, 64, 3, padding=1)

        # decoder (upsampling)
        self.upsample0 = nn.Upsample(16)  # 8 -> 16
        self.dec_conv0 = nn.Conv2d(64, 64, 3, padding=1)
        self.upsample1 = nn.Upsample(32)  # 16 -> 32
        self.dec_conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.upsample2 = nn.Upsample(64)  # 32 -> 64
        self.dec_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.upsample3 = nn.Upsample(128)  # 64 -> 128
        self.dec_conv3 = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, x):
        # encoder
        e0 = self.pool0(F.relu(self.enc_conv0(x)))
        e1 = self.pool1(F.relu(self.enc_conv1(e0)))
        e2 = self.pool2(F.relu(self.enc_conv2(e1)))
        e3 = self.pool3(F.relu(self.enc_conv3(e2)))

        # bottleneck
        b = F.relu(self.bottleneck_conv(e3))

        # decoder
        d0 = F.relu(self.dec_conv0(self.upsample0(b)))
        d1 = F.relu(self.dec_conv1(self.upsample1(d0)))
        d2 = F.relu(self.dec_conv2(self.upsample2(d1)))
        d3 = self.dec_conv3(self.upsample3(d2))
        return d3

# Small sanity check
tester_model = UNet()
print("Call time")
for images, mask in drive_train_dataloader:
    print(images.shape, mask.shape)
    out = tester_model(images)
    print(out.shape)
    break

num_pixels_drive = 0
num_positive_drive = 0
for image_batch, mask_batch in drive_train_dataloader:
    for mask in mask_batch:
        mask = mask[0]
        num_pixels_drive += len(mask) * len(mask[0])
        num_positive_drive += sum(sum(r) for r in mask)

num_pixels_ph2 = 0
num_positive_ph2 = 0
for image_batch, mask_batch in ph2_train_dataloader:
    for mask in mask_batch:
        mask = mask[0]
        num_pixels_ph2 += len(mask) * len(mask[0])
        num_positive_ph2 += sum(sum(r) for r in mask)

weights_drive = [1 - (num_pixels_drive - num_positive_drive)/num_pixels_drive, 1 - num_positive_drive/num_pixels_drive]
weights_ph2 = [1 - (num_pixels_ph2 - num_positive_ph2)/num_pixels_ph2, 1 - num_positive_ph2/num_pixels_ph2]

# Binary crossentropy
def bce_loss(y_pred, y_real):
    return F.binary_cross_entropy_with_logits(y_pred, y_real)

def dice_loss(y_pred, y_real, epsilon=1e-6):
    y_pred = torch.sigmoid(y_pred)  
    y_real_flat = y_real.view(-1)
    y_pred_flat = y_pred.view(-1)

    intersection = 2 * y_real_flat * y_pred_flat              
    numerator = torch.mean(intersection + 1)          
    denominator = torch.mean(y_real_flat + y_pred_flat) + 1    
    return 1 - (numerator / denominator)

from torchmetrics.classification import Dice
dice_score = Dice()

def focal_loss(y_pred, y_real):
    gamma = 2
    y_hat = torch.sigmoid(y_pred)
    focal = -torch.sum((1-y_hat) ** gamma * y_real * torch.log(y_hat) + (1 - y_real) * torch.log((1-y_hat)))
    return focal

weights_drive = torch.tensor(weights_drive, dtype=torch.float32)
weights_ph2 = torch.tensor(weights_ph2, dtype=torch.float32)

weighted_crossentropy_drive = nn.CrossEntropyLoss(weight=weights_drive, reduction='none')
weighted_crossentropy_ph2 = nn.CrossEntropyLoss(weight=weights_ph2, reduction='none')


# Train loop
def train(model, opt, loss_fn, epochs, train_loader, test_loader, title="dataset"):
    for epoch in range(epochs):
        tic = time()
        print("* Epoch %d/%d" % (epoch+1, epochs))
        avg_loss = 0
        model.train()  # train mode
        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            # set parameter gradients to zero
            opt.zero_grad() 
            # forward
            Y_pred = model(X_batch)
            loss = loss_fn(Y_pred, Y_batch)  # forward-pass
            loss.backward()  # backward-pass
            opt.step()  # update weights
            # calculate metrics to show the user
            avg_loss += loss / len(train_loader)
        toc = time()
        # show intermediate results
        model.eval()  # testing mode
        Y_hat = []
        test_to_plot = []
        avg_dice = 0
        for i, (image, mask) in enumerate(test_loader):
            pred = torch.sigmoid(model(image.to(device)).detach().cpu().squeeze(dim=0))
            avg_dice += dice_score(pred, mask.squeeze(dim=0).to(torch.int32))
            if i < 4:
                Y_hat.append(model(image.to(device)).detach().cpu())
                test_to_plot.append([image.squeeze(dim=0), pred])
        avg_dice /= len(test_loader)
        print(f" - loss: {avg_loss}")
        print(f" - Dice score: {avg_dice}")

        visualizer(test_to_plot, num_samples=4, title=f"Dataset-{title}-Epoch-{epoch}", folder="./train_plots")

drive_model = UNet().to(device)
PH2_model = UNet().to(device)


os.makedirs("./train_plots", exist_ok=True)
print(f"Folder './train_plots' created: {os.path.isdir('./train_plots')}")
train(drive_model, optim.Adam(drive_model.parameters()), weighted_crossentropy_drive, EPOCHS, drive_train_dataloader, drive_test_dataloader, title="DRIVE")
train(PH2_model, optim.Adam(PH2_model.parameters()), weighted_crossentropy_ph2, EPOCHS, ph2_train_dataloader, ph2_test_dataloader, title="PH2")

os.makedirs("./models", exist_ok=True)
print(f"Folder './models' created: {os.path.isdir('./models')}")
torch.save(drive_model, "./models/drive_model.pth")
torch.save(PH2_model, "./models/ph2_model.pth")
