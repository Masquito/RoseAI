import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


#Creating scalar
scalar = torch.tensor(7)
print(scalar)
print(scalar.ndim)
print(scalar.item())

#Creating vector
vector = torch.tensor([7, 7])
print(vector)
print(vector.ndim)

#Creating MATRIX
MATRIX = torch.tensor([[7, 8],[9, 10]])
print(MATRIX)
print(MATRIX.ndim)

#Creating TENSOR
TENSOR = torch.tensor([[[7, 8],[9, 10],[11, 12]],[[7, 8],[9, 10],[11, 13]]])
print(TENSOR)
print(TENSOR.ndim)
print(TENSOR.shape)
print(TENSOR[0])
print(TENSOR[1])

#Random Tensors
rand_tensor = torch.rand(5, 10)
print(rand_tensor)

#Zeros and Ones
zeros = torch.zeros(3, 7)
print(zeros)

ones = torch.ones(5, 10)
print(ones)

#Tensor ranges
t = torch.arange(0, 100, 5)
print(t)

#Tensor like
t_like = torch.zeros_like(t)
print(t_like)

#Getting Tensor info
#tensor.dtype - typ danych
#tensor.device - urządzenie
#tensor.shape - forma

some_tensor = torch.rand(5, 7, dtype=torch.float16, device=None, requires_grad=False)
someTensorInt = torch.randint(0, 30, (5, 7), dtype=torch.int16, device=None, requires_grad=False)
print(some_tensor)
print(someTensorInt)
print(f"Shape: {some_tensor.shape}")
print(f"Datatype: {some_tensor.dtype}")
print(f"Device: {some_tensor.device}")
print(some_tensor * someTensorInt)

#Tensor manipulation
tensor = torch.tensor([1, 2, 3], dtype=torch.float32)
tensor.add_(5)
tensor.mul_(2)
tensor.div_(10)
print(tensor)

#Element wise multiplication
tensor = torch.tensor([1, 2, 3], dtype=torch.float32)
tensor2 = torch.tensor([4, 5, 6], dtype=torch.float32)
tensor.mul_(tensor2)
print(tensor)

#Matrix manipulation == dot product
print(torch.matmul(tensor2, tensor2))

#Tensor transpose (switch axis and shapes np. 2x3 --> 3x2)
tensor3 = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
print(tensor3)
print(tensor3.t())

#Tensor agregation
tensor4 = torch.arange(0, 100, 10, dtype=torch.float32)
print(torch.min(tensor4))
print(torch.max(tensor4))
print(torch.mean(tensor4)) #<---- mean to średnia
print(torch.sum(tensor4)) #<---- sum to suma

print(torch.argmin(tensor4))   #<-- index of the smallest element
print(torch.argmax(tensor4))


#Reshaping, stacking, squeezing and unsqueezing
tensor5 = torch.arange(1., 10.)
print(tensor5)

#Add extra dimension
print(tensor5.reshape(3, 3))
tensor5 = tensor5.reshape(3, 3)
print(tensor5.reshape(3, 3, 1))

#Change the view
z = tensor5.view(3, 3, 1)
print(z)
print(tensor5)

#Stacking tensors   <----- Stackowanie dodaje nowy dimension czasem, dim=0 stackuje wieraszami, dim=1 stackuje kolumnami
tensor6 = torch.tensor([1, 2, 3, 4, 5, 6])
tensor7 = torch.tensor([[1, 20, 3],[4, 10, 6]])
print(torch.stack([tensor6, tensor6, tensor6, tensor6], dim=1))

#Tensor squeeze
print(tensor6.unsqueeze(0))   #<--- dodaje jeden wymiar w tym samym wierszu  tensor([[1, 2, 3, 4, 5, 6]])
tensor6 = tensor6.unsqueeze(1)  #<--- zamienia każdy element na nowy wiersz 
#    tensor([[1],
#            [2],
#            [3],
#            [4],
#            [5],
#            [6]])

print(tensor6)
tensor6 = tensor6.squeeze()  #<-- usuwa wymiary, które mają '1' np. z size(1,1,4) zostanie size(4), a z size(9,1,4) zostanie size(9,4)
print(tensor6)

#Tensor permute  <----- Permute zmienia forme tensora, np. z size(3,4,5) zostanie size(5,3,4)
x = torch.tensor([[[1, 20, 3],[4, 10, 6]],
    [[7, 8, 9],[10, 11, 12]]])
print(x[:, 0])
print(x.permute(2, 0, 1).size())   #<--- tutaj parametry permute definiują indeksy wymiarów ->> torch.Size([5, 3, 4])

# ': bierze wszystko z danego wymiaru'

#Tensors and NumPy
array = np.arange(1.0, 8.0)
tensor = torch.from_numpy(array)
print(tensor)
print(array)

# PyTorch reproducibility (trying to take random out of random)
# Z każdym razem tworząc tensor z randomowymi wartościami są one losowe,
# ale można im nadać **random seed**, żeby dodać jakieś ziarno dla generatora losowych liczb
# Trochę jak w Minecraft seed trochę określał tą losowość, ten generator świata
random_tensor_A = torch.rand(3, 4)
random_tensor_B = torch.rand(3, 4)
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)

random_tensor_C = torch.rand(3, 4)
torch.manual_seed(RANDOM_SEED)
random_tensor_D = torch.rand(3, 4)
print(random_tensor_C)
print(random_tensor_D)
print(random_tensor_C == random_tensor_D)

#Running PyTorch objects and tensors on GPU for faster computtation
#Device agnostic code!!!!
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

#Create a tensor
tensoret = torch.tensor([1,2,3])
tensoret.to(device)


##EXCERCISES
rand_tens = torch.rand(7, 7)
rand_tens2 = torch.rand(1, 7)
efect_tens = torch.matmul(rand_tens, rand_tens2.t())
torch.manual_seed(42)
rand_tens3 = torch.rand(7, 7)
rand_tens4 = torch.rand(1, 7)
print(torch.matmul(rand_tens3, rand_tens4.t()))
torch.cuda.manual_seed(42)
print(efect_tens.argmax())
print(efect_tens.argmin())
torch.manual_seed(7)
big_tens = torch.rand(1, 1, 1, 10)
big_tens_reduced = big_tens.squeeze()
print(big_tens)
print(big_tens.size())
print(big_tens_reduced)
print(big_tens_reduced.size())


training_data = datasets.FashionMNIST(
    root="data",
    download=True,
    train=True,
    transform=ToTensor(),
)

test_data = datasets.FashionMNIST(
    root="data",
    download=True,
    train=False,
    transform=ToTensor(),
)









