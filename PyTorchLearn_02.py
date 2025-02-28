import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch.onnx
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import pandas as pd

from helper_functions import plot_decision_boundary

### FUNCTIONS ###


# Calculate accuracy - out of 100 examples, what percentage does our model get right?
def accuracy_fn(y_pred_labels, y_true_labels):
    correct = torch.eq(y_true_labels, y_pred_labels).sum().item()
    acc = correct / len(y_pred_labels) * 100
    return acc

def plot_data(data):
    fig, ax = plt.subplots(figsize=(10, 9))
    for label, subset in data.groupby('label'):
        ax.scatter(subset["X1"], subset["X2"], label=label, alpha=0.7)
    ax.legend()
    ax.set_title("Data from circles")
    ax.set_ylabel('X2')
    ax.set_xlabel('X1')
    plt.show()


### FUNCTIONS ###



# 1. Make data to work with
n_samples = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X, y = make_circles(n_samples, noise=0.03, random_state=42)
circles = pd.DataFrame({"X1": X[:, 0],
                        "X2": X[:, 1],
                        "label": y})
print(circles.head(10))

# Split data into test and train sets ratio 80% - 20%
train_split = int(0.80 * len(circles))
X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
y = y.unsqueeze(dim=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_split, random_state=42)

class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 50)
        self.silu = nn.SiLU()
        self.layer2 = nn.Linear(50, 45)
        self.layer3 = nn.Linear(45, 1)
        

    def forward(self, x):
        x = self.silu(self.layer1(x))
        return self.layer3(self.silu(self.layer2(x))) # x -> layer1 -> layer2 -> output

model_0 = CircleModelV0().to(device)
with torch.inference_mode():
    untrained_predictions = model_0(X_test.to(device))


# Loss function and optimizer -> this is problem specific
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model_0.parameters(), lr=0.1)

torch.manual_seed(42)
torch.cuda.manual_seed(42)

epochs = 1000
epoch_count = []
loss_count = []
test_loss_count = []

X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)
print(X_train)
for epoch in range(epochs):
    # Training
    model_0.train()

    # 1. Forward pass
    local_pred_logits = model_0(X_train)
    y_pred = torch.round(torch.sigmoid(local_pred_logits)) # turn logits into prediction propabilities and then into prediction labels
    
    # 2. Calculate loss/accuracy
    loss = loss_fn(local_pred_logits, y_train) #BCEWithLogitsLoss expects raw logits as input
    acc = accuracy_fn(y_pred, y_train)

    # 3. Zero gradients
    optimizer.zero_grad()

    # 4. Loss backward
    loss.backward()

    # 5. Optimizer step
    optimizer.step()

    ### Testing
    model_0.eval()
    with torch.inference_mode():
        test_logits = model_0(X_test)
        test_pred = torch.round(torch.sigmoid(test_logits))
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(test_pred, y_test)
        epoch_count.append(epoch)
        loss_count.append(loss)
        test_loss_count.append(test_loss)

    #if epoch % 10 == 0:
       # print(f"Epoch {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("train")
plot_decision_boundary(model_0, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_0, X_test, y_test)
plt.show()




















































