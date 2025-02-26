import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch.onnx

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#1. Create known parameteres

weight = 0.7
bias = 0.3

# Create data
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

print(X[:10])
print(y[:10])

# Splitting data into training and test sets
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]  #<--- od początku do train split
X_test, y_test = X[train_split:], y[train_split:]  #<--- od train split do konca

def plot_predictions(trainData, trainLabels, testData, testLabels, predictions=None):
    plt.figure(figsize=(10, 7))
    plt.scatter(trainData, trainLabels, c='blue', s=10, label='Training data')
    plt.scatter(testData, testLabels, c='green', s=10, label='Test data')
    if predictions is not None:
        plt.scatter(testData, predictions, c='red', s=10, label='Predictions')
    plt.xlabel('Input data')
    plt.ylabel('Label')
    plt.title('Training and Test Data')
    plt.legend()
    plt.grid(True)
    plt.show()

#PyTorch model buidling essentials
# torch.nn - contains buidling blocks for neural networks (computational graphs = neural netowrk)
# torch.nn.parameter - parameter of the netowrk (weight or bias - we can change parameters)
# torch.nn.Module - base class for all neural network modules, overwrite forward()
# torch.optim - contains optimizers for updating parameters (algorithms that make the parameters better - change values from random to desired for learning aim)
# def forward() - Define what happenes in the forward computation

# Build a model for linear regression
class LinearRegressionModule(nn.Module):    #Inheritance from nn.Module (almost everything inside PyTorch)
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias


torch.manual_seed(42)
model_0 = LinearRegressionModule()
print(model_0.state_dict())

# Hyperparameter - parameter that is set by the programmer
# Setup a loss function
loss_fn = nn.L1Loss()

# Optimizer - Takes into account the loss of a model and adjusts the model's parameters
# Stochastic gradient decent
optimizer = torch.optim.SGD(model_0.parameters(), lr=0.01)  #lr - learning rate 

#Building a training loop and a testing loop
# 0. Loop through data
# 1. Forward pass - Forward proppagation
# 2. Calculate loss
# 3. Optimizer zero grad
# 4. Loss backward (**Backpropagation**) - move backwards through the network to calculate the gradients of each of the parameters of our model with respect to the loss
# 5. Optimizer step (**Gradient decent**) - use optimizer to adjust model parameteres to improve the loss


#An epoch is one loop through the data
epochs = 500
epoch_count = []
loss_values = []
test_loss_values = []
### Training
# 0. Loop through data

for epoch in range(epochs):
    #Set the model to training mode
    model_0.train() # Train mode in PyTorch sets all parameters requires_grad=True and so it means that they will train (adjust parameteres to minimize loss)

    # 1. Forward pass - Forward propagation
    y_pred = model_0(X_train)

    # 2. Calculate loss
    loss = loss_fn(y_pred, y_train)

    # 3. Optimizer zero grad - Zero the gradients of all parameters. It is needed because we need the optimizer to start fresh every epoch with the current values not accumulate them
    optimizer.zero_grad()

    # 4. Loss backward (**Backpropagation**) - move backwards through the network to calculate the gradients of each of the parameters of our model with respect to the loss
    loss.backward()

    # 5. Optimizer step (**Gradient decent**) - use optimizer to adjust model parameteres to improve the loss
    optimizer.step()

    ### Testing - evaluate the patterns that the model has learned
    model_0.eval()

    with torch.inference_mode():   #turns off gradient tracking
        test_pred = model_0(X_test)
        test_loss = loss_fn(test_pred, y_test)

    epoch_count.append(epoch)
    loss_values.append(loss)
    test_loss_values.append(test_loss)

plt.plot(epoch_count, np.array(torch.tensor(loss_values).numpy()), label='Training loss')
plt.plot(epoch_count, test_loss_values, label='Test loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
print(model_0.state_dict())

# Save the model
# torch,save()
# torch.load()
# torch.nn.Module.load_state_dict()   <-- loads only the parameters of the model (weights and biases)




















