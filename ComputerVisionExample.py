### IMPORTS FUNCTIONS AND GLOBAL_VARIABLES BLOCK ###

from turtle import forward
import torch
from torch import nn, relu
from torch.utils.data import DataLoader, dataloader
import torchvision
import numpy as np
from torchvision import transforms
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch.onnx
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import pandas as pd
from helper_functions import plot_decision_boundary
from timeit import default_timer as timer
from tqdm.auto import tqdm
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def accuracy_fn(y_pred_labels, y_true_labels):
    correct = torch.eq(y_true_labels, y_pred_labels).sum().item()
    acc = correct / len(y_pred_labels) * 100
    return acc

def print_train_time(start : float, end : float):
    total_time = end - start
    print(f"Train time: {total_time:.3f}")
    return total_time

def eval_model(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module, accuracy_fn):
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            y_logits = model(X)

            loss += loss_fn(y_logits, y)
            acc += accuracy_fn(y_logits.argmax(dim=1), y)
        
        loss /= len(data_loader)
        acc /= len(data_loader)

    return {"model_name": model.__class__.__name__, 
            "model_loss": loss.item(),
            "model_acc": acc}

### /IMPORTS FUNCTIONS AND GLOBAL_VARIABLES BLOCK ###


### DATA PREPARATION AND EXPERIMENTS BLOCK ###

trainData = torchvision.datasets.MNIST("dataMINST", True, download=True, transform=ToTensor(), target_transform=None)
testData = torchvision.datasets.MNIST("dataMINST", False, download=True, transform=ToTensor(), target_transform=None)

trainDataLoader = DataLoader(trainData, 32, True)
testDataLoader = DataLoader(testData, 32, False)

### /DATA PREPARATION AND EXPERIMENTS BLOCK ###


### TRAINING AND TESTING LOOPS BLOCK ###

def train_step(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer, accuracy_fn, device: torch.device = device):
    train_loss, train_acc = 0, 0
    model.train()

    for i, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)
        logits = model(X)

        loss = loss_fn(logits, y)

        train_loss += loss
        train_acc += accuracy_fn(logits.argmax(dim=1), y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train acc: {train_acc:.2f}%")

def test_step(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module, accuracy_fn, device: torch.device = device):
    test_loss, test_acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X_test, y_test in data_loader:
            X_test, y_test = X_test.to(device), y_test.to(device)
            test_logits = model(X_test)
            test_loss += loss_fn(test_logits, y_test)
            test_acc += accuracy_fn(test_logits.argmax(dim=1), y_test)
        
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test loss: {test_loss:.5f} | Test acc: {test_acc:.2f}%\n\n\n")

### /TRAINING AND TESTING LOOPS BLOCK ###


### MODEL CREATION BLOCK ###
class MNISTConvNNModel(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*7*7, out_features=output_shape)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x

### /MODEL CREATION BLOCK ###


### Training and testing block ###

for epoch in range(epochs):
    print(f"Epoch: {epoch}\n-----")
    train_step(model_0, dataLoader_train, loss_fn, optimizer, accuracy_fn, device)
    test_step(model_0, dataLoader_test, loss_fn, accuracy_fn, device)
































