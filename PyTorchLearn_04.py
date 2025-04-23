import torch
from torch import nn
import random
from PIL import Image
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
import glob
from pathlib import Path
import os
import typing


device = "cuda" if torch.cuda.is_available() else "cpu"
data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"

train_dir = image_path / "train"
test_dir = image_path / "test"

random.seed(42)
image_path_list = list(image_path.glob("*/*/*.jpg"))
random_image_path = random.choice(image_path_list)
img = Image.open(random_image_path)
imgAsArray = np.asarray(img)

dataTransform = transforms.Compose([
  transforms.Resize(size=(64, 64)),
  transforms.RandomHorizontalFlip(p=0.5),
  transforms.ToTensor()
])

trainData = datasets.ImageFolder(root=train_dir, 
                                  transform=dataTransform,   #transform data
                                  target_transform=None)     #transform labels

testData = datasets.ImageFolder(root=test_dir,
                                  transform=dataTransform)

BATCH_SIZE = 32
trainDataloader = DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=True)
testDataloader = DataLoader(dataset=testData, batch_size=BATCH_SIZE, shuffle=False)

targetDirectory = train_dir

def FindClasses(directory: str) -> tuple[list[str], dict[str, int]]:
  classNamesFound = sorted([entry.name for entry in list(os.scandir(directory))])
  if not classNamesFound:
    raise FileNotFoundError(f"Couldn't find any classes in {directory}")
  classToIdx = {className: i for i, className in enumerate(classNamesFound)}
  return classNamesFound, classToIdx

print(FindClasses(train_dir))






