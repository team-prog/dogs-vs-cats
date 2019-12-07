import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        #Capa convolución 1: Toma imágenes con 3 canales y un kernel de
        #5x5 para generar imágenes de 224 canales
        #Entrada: 32x32; Salida: 28x28 (Se pierden 2 pixeles de cada borde)
        #Entrada: 256x256; Salida: 254x254 (Se pierden 2 pixeles de cada borde)
        self.conv1 = nn.Conv2d(3, 256, 5) # Kerkel (filter 1) of 5x5
        #Capa MaxPool. Se deja un elemento de cada kernel de 2x2
        #Entrada: 254x254; Salida: 127x127
        self.pool = nn.MaxPool2d(2, 2) # Stride of 2
        #Capa convolución 2: Toma imágenes de 224 canales y un kernel de
        #5x5 para generar imágenes de 16 canales.
        #Entrada: 14x14; Salida: 10x10
        #Entrada: 127x127; Salida: 125x125
        self.conv2 = nn.Conv2d(256, 64, 5) # Kerkel (filter 2) of 5x5 with input of 6 and output of 16
        self.fc1 = nn.Linear(64 * 61 * 61, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2) # Change this one by max soft ?

    def forward(self, x):
        # First Layer:  x => filter 1 + Relu => max pool
        # Second Layer: x => filter 2 + Relu => max pool
        # x => flattening of x
        # Third Layer: x => Relu x
        # Fourth Layer: x => Relu x
        # print('x = ', ' shape: ', x.shape)
        x = self.pool(F.relu(self.conv1(x)))
        # print('1 x = ', ' shape: ', x.shape)
        x = self.pool(F.relu(self.conv2(x)))
        # print('2 x = ', ' shape: ', x.shape)
        x = x.view(-1, 64 * 61 * 61)
        # print('3 x = ', ' shape: ', x.shape)
        x = F.relu(self.fc1(x))
        # print('4 x = ', ' shape: ', x.shape)
        x = F.relu(self.fc2(x))
        # print('5 x = ', ' shape: ', x.shape)
        # x = F.softmax(self.fc3(x), 1)
        x = self.fc3(x)
        # print('6 x = ', ' shape: ', x.shape)
        return x

    def save(self, path):
      torch.save(self.state_dict(), path)

    def load(self, path):
      self.load_state_dict(torch.load(path))
