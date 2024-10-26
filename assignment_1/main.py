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
import matplotlib.pyplot as plt
import wandb
wandb.login(key='4aaf96e30165bfe476963bc860d96770512c8060')
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

path = "./hotdog_nothotdog"

# ########## #
# Parameters #
# ########## #
import math
sweep_config = {
    'method' : "grid", # 'random', #bayes
}

metric = {
    'name': 'loss',
    'goal': 'minimize'
  }

parameters_dict = {
    'optimizer' : {
        'values': ['adam', 'sgd']
    },
    'dropout' : {
        'values' : [0, 0.2, 0.5]
    },
    'epochs' : {
        'value': 20
    },
    'learning_rate' : {
        "values": [0.0001, 0.001, 0.01, 0.1, 1]
    },
    'batch_size': {
        "values": [8, 16, 32, 64]
    },
    'image_size':{
        'value': 128
    },
    "num_layers": {
        "values": list(range(1, 7))
    }
}
sweep_config['metric'] = metric
sweep_config['parameters'] = parameters_dict

sweep_id = wandb.sweep(sweep_config, project='Hotdogs2')

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

class Base_Network(nn.Module):
    def __init__(self, dropout, num_layers, feature_size=128, base_channel_sz = 8):
        super(Base_Network, self).__init__()

        self.beginning = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=base_channel_sz, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(base_channel_sz),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        feature_size //= 2 # Convolution -> Padding -> Pooling

        layers = []
        for i in range(num_layers):
          layers.append(nn.Conv2d(in_channels=base_channel_sz*(i+1), out_channels=base_channel_sz*(i+2), kernel_size=3, padding=1))
          layers.append(nn.BatchNorm2d(base_channel_sz*(i+2)))
          layers.append(nn.ReLU())
          
          layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
          feature_size //= 2 # Convolution -> Padding -> Pooling

        self.convolutional = nn.Sequential(*layers)
        self.fully_connected = nn.Sequential(
            nn.Linear(in_features= base_channel_sz * (num_layers + 1) * feature_size * feature_size, out_features=512),
            nn.ReLU(),
            nn.Dropout(dropout), # added dropout to reduce overfitting

            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.beginning(x)
        x = self.convolutional(x)
        #reshape x so it becomes flat, except for the first dimension (which is the minibatch)
        x = x.view(x.size(0), -1)

        x = self.fully_connected(x)
        return x

def build_optimizer(model, optimizer, learning_rate):
  if optimizer == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
  elif optimizer == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
  return optimizer

def save_checkpoint(model, optimizer, epoch, path='model_checkpoint.pth'):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, path)
    print(f"Model checkpoint saved at epoch {epoch}, name: {path}")

# Define the loss function
criterion = nn.BCELoss()

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
        
        # Saves the model
        save_path = f"./checkpoints/epoch_{epoch}_{run_id}.pth"
        save_checkpoint(model, optimizer, epoch, save_path)

        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss: {out_dict['train_loss'][-1]:.3f}, "
              f"Test Loss: {out_dict['test_loss'][-1]:.3f}, "
              f"Train Acc: {out_dict['train_acc'][-1]*100:.1f}%, "
              f"Test Acc: {out_dict['test_acc'][-1]*100:.1f}%")

    return out_dict

os.makedirs("checkpoints", exist_ok=True)

def run_wandb(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config):
        config = wandb.config

        # Include the run id so that we can identify the saved models
        config = wandb.config
        run_id = wandb.run.id 
        config.run_id = run_id
        wandb.run.name = f"Run {run_id}"

        train_loader, test_loader,trainset,testset = build_dataset(config.batch_size,config.image_size)
        model = Base_Network(config.dropout, config.num_layers) #build_network(config.fc_layer_size, config.dropout)
        model.to(device)
        print(model)


        optimizer = build_optimizer(model, config.optimizer, config.learning_rate)

        # Generate a random id for this run and this model
        train(model, optimizer, train_loader, test_loader, trainset, testset, config.epochs, run_id)

wandb.agent(sweep_id, run_wandb)