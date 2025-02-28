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
NUM_CLASSES = 3
NUM_FEATURES = 2
RANDOM_SEED = 42

N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
X = np.zeros((N*K,D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels
for j in range(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y[ix] = j

points = torch.from_numpy(X).type(torch.float).to(device)
labels = torch.from_numpy(y).type(torch.float).to(device)
trainSplit = int(0.80 * len(points))

points_train, points_test, labels_train, labels_test = train_test_split(points, labels, train_size=trainSplit, random_state=42)
labels_train = labels_train.type(torch.long)
labels_test = labels_test.type(torch.long)
class SpiralsClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(NUM_FEATURES, 20)
        self.layer2 = nn.Linear(20, 10)
        self.layer3 = nn.Linear(10, NUM_CLASSES)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.layer3(self.relu(self.layer2(self.relu(self.layer1(x)))))

model_2 = SpiralsClassificationModel().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_2.parameters(), lr=0.05)

epochs = 500

for epoch in range(epochs):
    model_2.train()

    logits = model_2(points_train)
    pred_labels = torch.argmax(torch.softmax(logits, 1), 1)

    loss = loss_fn(logits, labels_train)
    acc = accuracy_fn(pred_labels, labels_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model_2.eval()
    with torch.inference_mode():
        test_logits = model_2(points_test)
        test_labels = torch.argmax(torch.softmax(test_logits, 1), 1)
        test_loss = loss_fn(test_logits, labels_test)
        test_acc = accuracy_fn(test_labels, labels_test)

    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("train")
plot_decision_boundary(model_2, points_train, labels_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_2, points_test, labels_test)
plt.show()



