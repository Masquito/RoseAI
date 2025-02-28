import random
import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch.onnx
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import pandas as pd
from helper_functions import plot_decision_boundary
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.classification import BinaryAccuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def accuracy_fn(y_pred_labels, y_true_labels):
    correct = torch.eq(y_true_labels, y_pred_labels).sum().item()
    acc = correct / len(y_pred_labels) * 100
    return acc

# Data preparation block
NUM_CLASSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 42

X_blob, y_blob = make_blobs(n_samples=1000, n_features=NUM_FEATURES, centers=NUM_CLASSES, cluster_std=1.5, random_state=RANDOM_SEED)
X_blob = torch.from_numpy(X_blob).type(torch.float).to(device)
y_blob = torch.from_numpy(y_blob).type(torch.float).to(device)
trainPercentage = int(0.80 * len(X_blob))

X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(X_blob, y_blob, train_size=trainPercentage, random_state=RANDOM_SEED)

class MultiClassificationModelV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(NUM_FEATURES, 50)
        self.hiddenLayer1 = nn.Linear(50, 50)
        self.hiddenLayer2 = nn.Linear(50, 12)
        self.output = nn.Linear(12, NUM_CLASSES)
        self.silu = nn.SiLU()

    def forward(self, x):
        return (self.output(self.silu(self.hiddenLayer2(self.silu(self.hiddenLayer1(self.silu(self.input(x))))))))

model_0 = MultiClassificationModelV1().to(device)

X_blob_train, X_blob_test, y_blob_train, y_blob_test = X_blob_train.to(device), X_blob_test.to(device), y_blob_train.to(device), y_blob_test.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_0.parameters(), lr=0.05)

model_0.eval()
with torch.inference_mode():
    y_logits = model_0(X_blob_test.to(device))

y_pred_probs = torch.softmax(y_logits, 1)
y_pred_labels = torch.argmax(y_pred_probs, 1).type(torch.float)

epochs = 700
epoch_count = []
loss_count = []
test_loss_count = []
y_blob_test = y_blob_test.type(torch.long)
y_blob_train = y_blob_train.type(torch.long)
for epoch in range(epochs):
    model_0.train()
    y_logits = model_0(X_blob_train)
    y_local_pred_labels = torch.argmax(torch.softmax(y_logits, 1), 1)
    loss = loss_fn(y_logits, y_blob_train)
    acc = accuracy_fn(y_local_pred_labels, y_blob_train)
    optimizer.zero_grad()

    loss.backward()
    
    optimizer.step()

    model_0.eval()
    with torch.inference_mode():
        test_logits = model_0(X_blob_test)
        testPredWithLabels = torch.argmax(torch.softmax(test_logits, 1), 1)
        test_loss = loss_fn(test_logits, y_blob_test)
        test_acc = accuracy_fn(testPredWithLabels, y_blob_test)
        epoch_count.append(epoch)
        loss_count.append(loss)
        test_loss_count.append(test_loss)

    #if epoch % 10 == 0:
        #print(f"Epoch {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")

# metric = MulticlassAccuracy(4)
# print(metric(y_local_pred_labels, y_blob_train))


### EXCERCISES ###

# Data prep block
Features, classes = make_moons(1000, random_state=RANDOM_SEED)
X_moon, y_moon = torch.from_numpy(Features).type(torch.float).to(device), torch.from_numpy(classes).type(torch.float).to(device)
trainSplit = int(0.80 * len(X_moon))


X_moon_train, X_moon_test, y_moon_train, y_moon_test = train_test_split(X_moon, y_moon, train_size=trainSplit, random_state=RANDOM_SEED)

y_moon_train = y_moon_train.unsqueeze(1)
y_moon_test = y_moon_test.unsqueeze(1)
class MoonModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 30)
        self.layer2 = nn.Linear(30, 30)
        self.layer3 = nn.Linear(30, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.layer3(self.relu(self.layer2(self.relu(self.layer1(x)))))


model_1 = MoonModelV0().to(device)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model_1.parameters(), lr=0.05)
epochs = 1000

X_moon_train, X_moon_test, y_moon_train, y_moon_test = X_moon_train.to(device), X_moon_test.to(device), y_moon_train.to(device), y_moon_test.to(device)

for epoch in range(epochs):
    model_1.train()

    logits = model_1(X_moon_train)
    labels = torch.round(torch.tanh(logits))

    loss = loss_fn(logits, y_moon_train)
    acc = accuracy_fn(labels, y_moon_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model_1.eval()
    with torch.inference_mode():
        test_logits = model_1(X_moon_test)
        test_labels = torch.round(torch.tanh(test_logits))
        test_loss = loss_fn(test_logits, y_moon_test)
        test_acc = accuracy_fn(test_labels, y_moon_test)

    

    
















