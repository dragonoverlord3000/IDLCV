import os
import numpy as np
import glob
import PIL.Image as Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import wandb
wandb.login(key='4aaf96e30165bfe476963bc860d96770512c8060')
import uuid
import os

if torch.cuda.is_available():
    print("The code will run on GPU.")
else:
    print("The code will run on CPU. Go to Edit->Notebook Settings and choose GPU as the hardware accelerator")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ###### #
# Hotdog #
# ###### #
class Hotdog_NotHotdog(torch.utils.data.Dataset):
    def __init__(self, train, transform, data_path='/dtu/datasets1/02516/hotdog_nothotdog'):
        'Initialization'
        self.transform = transform
        data_path = os.path.join(data_path, 'train' if train else 'test')
        image_classes = [os.path.split(d)[1] for d in glob.glob(data_path +'/*') if os.path.isdir(d)]
        image_classes.sort()
        self.name_to_label = {c: id for id, c in enumerate(image_classes)}
        self.image_paths = glob.glob(data_path + '/*/*.jpg')

    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.image_paths[idx]

        image = Image.open(image_path)
        c = os.path.split(os.path.split(image_path)[0])[1]
        y = self.name_to_label[c]
        X = self.transform(image)
        
        # print(f"Class: {c}, Label: {y}") ---> apparently 0 is hotdog and 1 is nothotdog :|

        return X, y
    

def build_dataset(batch_size, image_size):
    path = "./hotdog_nothotdog"
    size = image_size
    train_transform = transforms.Compose([transforms.Resize((size, size)),
                                          transforms.ToTensor(), 
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomVerticalFlip(), 
                                          transforms.RandomRotation(30), 
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    test_transform = transforms.Compose([transforms.Resize((size, size)),
                                        transforms.ToTensor(), 
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    batch_size = batch_size
    trainset = Hotdog_NotHotdog(train=True, transform=train_transform,data_path=path)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testset = Hotdog_NotHotdog(train=False, transform=test_transform,data_path=path)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader,trainset, testset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model_ft = models.resnet18(weights='IMAGENET1K_V1')
num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to ``nn.Linear(num_ftrs, len(class_names))``.
model_ft.fc = nn.Linear(num_ftrs, 2)
model_ft = model_ft.to(device)
criterion = nn.CrossEntropyLoss()
# Observe that all parameters are being optimized
optimizer_ft = torch.optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# Define the training function
def train(model, optimizer, train_loader, test_loader, trainset, testset, num_epochs=10, run_id=""):
    out_dict = {
        'train_acc': [],
        'test_acc': [],
        'train_loss': [],
        'test_loss': []
    }

    for epoch in tqdm(range(num_epochs), unit='epoch'):
        model.train()
        train_correct = 0
        train_loss = []
        
        # Training phase
        for minibatch_no, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
            data, target = data.to(device), target.to(device).float().unsqueeze(1)
            
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
            predicted = (output >= 0.5).long().squeeze(1)  # Binary classification
            train_correct += (target.squeeze(1).long() == predicted).sum().cpu().item()

        # Testing phase
        test_loss = []
        test_correct = 0
        model.eval()  # Set the model to evaluation mode
        
        with torch.no_grad():  # No need to track gradients during testing
            for data, target in test_loader:
                data, target = data.to(device), target.to(device).float().unsqueeze(1)
                output = model(data)
                loss = criterion(output, target)
                
                # Track test loss
                test_loss.append(loss.item())
                
                # Predictions and accuracy
                predicted = (output >= 0.5).long().squeeze(1)
                test_correct += (target.squeeze(1).long() == predicted).sum().cpu().item()

        # Record statistics
        out_dict['train_acc'].append(train_correct / len(trainset))
        out_dict['test_acc'].append(test_correct / len(testset))
        out_dict['train_loss'].append(np.mean(train_loss))
        out_dict['test_loss'].append(np.mean(test_loss))

        # Log to WandB
        wandb.log({
            "train_acc": out_dict['train_acc'][-1],
            "test_acc": out_dict['test_acc'][-1],
            "train_loss": out_dict['train_loss'][-1],
            "test_loss": out_dict['test_loss'][-1],
            "epoch": epoch,
            "run_id": run_id
        })

        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss: {out_dict['train_loss'][-1]:.3f}, "
              f"Test Loss: {out_dict['test_loss'][-1]:.3f}, "
              f"Train Acc: {out_dict['train_acc'][-1]*100:.1f}%, "
              f"Test Acc: {out_dict['test_acc'][-1]*100:.1f}%", flush=True)

    return out_dict

BATCH_SIZE = 64
IMAGE_SIZE = 128
train_loader, test_loader,trainset, testset = build_dataset(BATCH_SIZE, IMAGE_SIZE)

run = wandb.init(entity="wandbee", \
        project="model_registry_example")
train(model_ft, optimizer_ft, train_loader, test_loader, trainset, testset)
