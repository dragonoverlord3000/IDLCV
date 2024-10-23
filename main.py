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

BATCH_SIZE = 4
NUM_WORKERS = 1
SHUFFLE = True


class CustomDataset(Dataset):
    def __init__(self, train=True, image_transform=None, mask_transform=None, joint_transform=None, data_folder="DRIVE", split_percentage=80):
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.joint_transform = joint_transform
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

# Plot images and their masks - to show dataset
def visualizer(dataset, num_samples=4, title="SBS-visualization"):
    plt.figure(figsize=(10, 20))
    for i in range(num_samples):
        idx = i
        image, mask = dataset[idx]
        
        # Plot the image
        plt.subplot(num_samples, 2, i*2 + 1)
        plt.imshow(transforms.ToPILImage()(image))
        plt.title(f"Image {i+1}")
        plt.axis("off")

        # Plot the mask
        plt.subplot(num_samples, 2, i*2 + 2)
        plt.imshow(transforms.ToPILImage()(mask), cmap="gray")
        plt.title(f"Mask {i+1}")
        plt.axis("off")
        
    plt.tight_layout()
    plt.savefig(f"./vis/{title}.png")

def get_stats(dataset):
    image, mask = dataset[0]
    print(f"Image size: {image.shape}, Mask size: {mask.shape}, Dataset length: {len(dataset)}")

# Init datasets
image_transform = transforms.Compose([
    transforms.ToTensor(),
])

mask_transform = transforms.Compose([
    transforms.ToTensor(),
])

########
# TODO #
########
# Hyperparameter - augmentation transforms
# Training test distinction


joint_transform = transforms.Compose([
    transforms.RandomCrop((128,128)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.RandomRotation(30)
])

# DRIVE
drive_dataset_before = CustomDataset(train=True, image_transform=image_transform, mask_transform=mask_transform, data_folder="DRIVE")
drive_dataset_train = CustomDataset(train=True, image_transform=image_transform, mask_transform=mask_transform, joint_transform=joint_transform, data_folder="DRIVE")
drive_dataset_test = CustomDataset(train=False, image_transform=image_transform, mask_transform=mask_transform, data_folder="DRIVE")
visualizer(drive_dataset_before, 4, title="DRIVE_train_before")
visualizer(drive_dataset_train, 4, title="DRIVE_train_after")
visualizer(drive_dataset_test, 4, title="DRIVE_test_after")

print("#### DRIVE stats ####")
get_stats(drive_dataset_train)
get_stats(drive_dataset_test)

# PH2
ph2_dataset_before = CustomDataset(train=True, image_transform=image_transform, mask_transform=mask_transform, data_folder="PH2_Dataset_images")
ph2_dataset_train = CustomDataset(train=True, image_transform=image_transform, mask_transform=mask_transform, joint_transform=joint_transform, data_folder="PH2_Dataset_images")
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

# Binary crossentropy
def bce_loss(y_pred, y_real):
    return F.binary_cross_entropy_with_logits(y_pred, y_real)

# Train loop
def train(model, opt, loss_fn, epochs, train_loader, test_loader, title="dataset"):
    X_test, Y_test = next(iter(test_loader))
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
            loss = loss_fn(Y_batch, Y_pred)  # forward-pass
            loss.backward()  # backward-pass
            opt.step()  # update weights
            # calculate metrics to show the user
            avg_loss += loss / len(train_loader)
        toc = time()
        print(" - loss: %f" % avg_loss)
        # show intermediate results
        model.eval()  # testing mode
        Y_hat = F.sigmoid(model(X_test.to(device))).detach().cpu()
        for k in range(1):
            plt.subplot(2, 6, k+1)
            plt.imshow(np.rollaxis(X_test[k].numpy(), 0, 3), cmap="gray")
            plt.title("Real")
            plt.axis("off")
            plt.subplot(2, 6, k+7)
            plt.imshow(Y_hat[k, 0], cmap="gray")
            plt.title("Output")
            plt.axis("off")
        plt.suptitle("%d / %d - loss: %f" % (epoch+1, epochs, avg_loss))
        plt.savefig(f"./train_plots/epoch-{epoch}-time-{title}.png")

drive_model = UNet()
PH2_model = UNet()

os.mkdirs("./train_plots", exists_ok=True)
train(drive_model, optim.Adam(drive_model.parameters()), bce_loss, 20, drive_train_dataloader, drive_test_dataloader, title="DRIVE")
train(PH2_model, optim.Adam(PH2_model.parameters()), bce_loss, 20, ph2_train_dataloader, ph2_train_dataloader, title="PH2")

torch.save(drive_model)
torch.save(PH2_model)
