import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from utils import imshow, show_random, train_cnn, test_cnn, check_cnn, show_statistics
from Dataset import DogsVsCatsDataset
from cnn import CNN
from time import time
from torchvision import datasets, transforms
import torch
from torch import nn, optim

######## Constants #########
datset_file = './src/train/'

test_proportion = .2
data_size = 0

cnn = CNN()

# Enable this lines if you want use the resnet50

# cnn = torchvision.models.resnet50(pretrained=True)
# num_ftrs = cnn.fc.in_features
# cnn.fc = nn.Linear(num_ftrs, 2)

batch_size = 4

epochs = 10
batch_rate = 1000
learning_rate = 0.001

momentum = 0.9

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(cnn.parameters(), lr = learning_rate, momentum = momentum)

PATH_TO_SAVE = './dogs_vs_cats_net.pth'
PATH_TO_LOAD = './dogs_vs_cats_net_83_61_72.pth'

dataset = DogsVsCatsDataset(data_dir = datset_file, data_size = data_size)

train_size = int((1 - test_proportion) * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

## TRAIN and SAVE the CNN
# train_cnn(epochs, cnn, criterion, train_loader, optimizer, batch_rate)
# cnn.save(PATH_TO_SAVE)

## LOAD previous cnn
cnn.load(PATH_TO_LOAD)

## TEST the CNN
test_cnn(cnn, test_loader)

## CHECK the CNN
check_cnn(cnn, test_loader)

## Show the respective graphs
show_statistics(cnn, test_loader)