import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5) # Kerkel (filter 1) of 5x5
        self.pool = nn.MaxPool2d(2, 2) # Stride of 2
        self.conv2 = nn.Conv2d(32, 64, 5) # Kerkel (filter 2) of 5x5 with input of 6 and output of 16
        self.fc1 = nn.Linear(64 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10) # Change this one by max soft ?

    def forward(self, x):
        # First Layer:  x => filter 1 + Relu => max pool
        # Second Layer: x => filter 2 + Relu => max pool
        # x => flattening of x
        # Third Layer: x => Relu x
        # Fourth Layer: x => Relu x
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def save(self, path):
      torch.save(self.state_dict(), path)

    def load(self, path):
      self.load_state_dict(torch.load(path))
