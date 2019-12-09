import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import PIL.Image as Image
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import torchvision.transforms as transforms
from time import sleep


classes = ('cat', 'dog')

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def show_random(loader):
  images, labels = iter(loader).next()
  # show images
  imshow(torchvision.utils.make_grid(images))
  # print labels
  print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

def train_cnn(epochs, cnn, criterion, trainloader, optimizer):
  for epoch in range(epochs):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = cnn(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
  print('Finished Training')

def test_cnn(cnn, testloader):
  correct = 0
  total = 0
  with torch.no_grad():
    for data in testloader:
      images, labels = data
      outputs = cnn(images)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

  print('Accuracy of the network on the 10000 test images: %d %%' % (
      100 * correct / total))


def check_cnn(cnn, testloader):
  class_correct = list(0. for i in range(2))
  class_total = list(0. for i in range(2))
  with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = cnn(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

  for i in range(2):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

def show_statistics(cnn, test_loader):
  show_confusion_matrix(cnn, test_loader)


def labels(test_loader):
  res = []
  with torch.no_grad():
    for data in test_loader:
      _, labels = data
      res.append(labels[0])
  return res

def show_confusion_matrix(cnn, test_loader):
  cnn.eval()
  labels_encoder = LabelEncoder()
  labels_encoder.fit_transform(labels(test_loader))
  inputs = list()
  outputs = list()
  for _, tensor in enumerate(test_loader):   
    value, output = tensor
    inputs.append(value)
    outputs.append(output)
  inputs = torch.cat(inputs)
  outputs = torch.cat(outputs)
  _, predicted = torch.max(cnn(inputs), 1)
  cm = confusion_matrix(outputs.numpy(), predicted.numpy())
  plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
  plt.colorbar()
  tick_marks = np.arange(2)
  plt.xticks(tick_marks, labels_encoder.inverse_transform(range(2)), rotation=45)
  plt.yticks(tick_marks, labels_encoder.inverse_transform(range(2)))
  plt.xlabel("The model prediction")
  plt.ylabel("Real value")
  plt.show()
