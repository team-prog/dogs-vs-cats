import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from utils import imshow, show_random, train_cnn, test_cnn, check_cnn, show_image
from Dataset import DogsVsCatsDataset
from cnn import CNN
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
from random import randrange

######## Constants #########
datset_file = './train'

test_proportion = .2
data_size = 1000

cnn = CNN()

batch_size = 4

epochs = 10

learning_rate = 0.001

momentum = 0.9

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(cnn.parameters(), lr = learning_rate, momentum = momentum)

PATH_TO_SAVE = './dogs_vs_cats_net.pth'

labels = {'label': ['cat', 'dog'] }

labels_encoder = LabelEncoder()
number_labels = labels_encoder.fit_transform(labels['label'])

dataset = DogsVsCatsDataset(data_dir = datset_file, data_size = data_size, label_source = number_labels)

train_size = int((1 - test_proportion) * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

## TRAIN and SAVE the CNN
# train_cnn(epochs, cnn, criterion, train_loader, optimizer)
# cnn.save(PATH_TO_SAVE)

## LOAD previous cnn
# cnn.load(PATH_TO_SAVE)

## TEST the CNN
# test_cnn(cnn, test_loader)

# ## CHECK the CNN
# check_cnn(cnn, test_loader)

image_number = randrange(0, len(dataset) - 1)
# image_path, label = fetch_image(dataset, image_number)
# print("fetched")
# scaled_image = scale_image(image_path)
# print("scaled")
show_image(dataset, image_number)
