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

train_data = datasets.FashionMNIST("data", True, download=True, transform=ToTensor(), target_transform=None)
test_data = datasets.FashionMNIST("data", False, download=True, transform=ToTensor(), target_transform=None)
image, label = train_data[0]
class_names = train_data.classes

Shape = [32, 28, 28, 1]   # Batch size, height, width, color channels
dataLoader_train = DataLoader(dataset=train_data, batch_size=Shape[0], shuffle=True)
dataLoader_test = DataLoader(dataset=test_data, batch_size=Shape[0], shuffle=False)

torch.manual_seed(42)
train_features_batch, train_labels_batch = next(iter(dataLoader_train))
print(train_features_batch.shape)

flatten_model = nn.Flatten()
x = train_features_batch[0]
print(x.shape)
output = flatten_model(x)
print(output)
print(output.shape)

### /DATA PREPARATION AND EXPERIMENTS BLOCK ###


### MODEL CREATION BLOCK ###

class FashionMNISTModelV0(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(  # Sequential robi to, że dane będą po kolei przechodziły przez każdą warstwę
            nn.Flatten(),
            nn.Linear(input_shape, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units + 10),
            nn.ReLU(),
            nn.Linear(hidden_units + 10, output_shape)
        )

    def forward(self, x):
        return self.layer_stack(x)

### /MODEL CREATION BLOCK ###


### TRAINING AND TESTING SETUP BLOCK ###

torch.manual_seed(42)
model_0 = FashionMNISTModelV0(input_shape=784, hidden_units=30, output_shape=len(class_names)).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_0.parameters(), lr=0.1)

torch.manual_seed(42)
train_time = timer()
epochs = 0

### /TRAINING AND TESTING SETUP BLOCK ###


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


for epoch in range(epochs):
    print(f"Epoch: {epoch}\n-----")
    train_step(model_0, dataLoader_train, loss_fn, optimizer, accuracy_fn, device)
    test_step(model_0, dataLoader_test, loss_fn, accuracy_fn, device)

### /TRAINING AND TESTING LOOPS BLOCK ###


### FINAL EVALUATION BLOCK ###

train_time_end = timer()
total_train_time_model_0 = print_train_time(train_time, train_time_end)
model_0_results = eval_model(model_0, dataLoader_test, loss_fn, accuracy_fn)
print(model_0_results)

### /FINAL EVALUATION BLOCK ###


### CONVOLUTIONAL NEURAL NETWORK BLOCK ### 

class FashionConvNNModel(nn.Module):
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

torch.manual_seed(42)
model_2 = FashionConvNNModel(input_shape=1, hidden_units=30, output_shape=len(class_names)).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_2.parameters(), lr=0.1)

torch.manual_seed(42)
time_model2_start = timer()
epochs = 3

for epoch in tqdm(range(epochs)):
    print(f"EEpoch: {epoch}\n-----")
    train_step(model_2, dataLoader_train, loss_fn, optimizer=optimizer, accuracy_fn=accuracy_fn, device=device)
    test_step(model_2, dataLoader_test, loss_fn,  accuracy_fn=accuracy_fn, device=device)

time_model2_stop = timer()

def make_predictions(model: torch.nn.Module, data: list, device: torch.device = device):
    pred_prebs = []
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for sample in data:
            sample = torch.unsqueeze(sample, dim=0).to(device)
            pred_logits = model(sample)
            pred_propabilities = torch.softmax(pred_logits.squeeze(), dim=0)
            pred_prebs.append(pred_propabilities.cpu())

    return torch.stack(pred_prebs)        



random.seed(42)
test_samples = []
test_labels = []
for sample, label in random.sample(list(test_data), k=9):
    test_samples.append(sample)
    test_labels.append(label)

pred_probs = make_predictions(model = model_2, data=test_samples)
pred_classes = pred_probs.argmax(dim=1)
print(pred_classes)

plt.figure(figsize=(9, 9))
nrows = 3
ncols = 3
for i, sample in enumerate(test_samples):
    plt.subplot(nrows, ncols, i+1)
    plt.imshow(sample.squeeze(), cmap="gray")
    if class_names[pred_classes[i]] == class_names[test_labels[i]]:
        plt.title("Pred: " + class_names[pred_classes[i]] + "| Truth: " + class_names[test_labels[i]], fontsize=10, c="g")
    else:
        plt.title("Pred: " + class_names[pred_classes[i]] + "| Truth: " + class_names[test_labels[i]], fontsize=10, c="r")

plt.show()





















