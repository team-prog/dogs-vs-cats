from Dataset import DogsVsCatsDataset
import torch
import csv
from cnn import CNN

TEST_PATH = './dogs-vs-cats-redux-kernels-edition/test'
CNN_NAME = 'dogs_vs_cats_net_83_53_68'
CNN_PATH = './' + CNN_NAME + '.pth'
CSV_NAME = CNN_NAME + '_test_outputs.csv'
CSV_PATH = './csvs/' + CSV_NAME

def generate(cnn):
  batch_size = 4
  dataset = DogsVsCatsDataset(data_dir = TEST_PATH, data_size = 0)
  loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
  with open(CSV_NAME, 'w', newline='') as csvfile:
    fieldnames = ['id', 'label']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for i, data in enumerate(loader, 0):
      images, _ = data
      _, predicted = torch.max(cnn(images), 1)
      for j in range(batch_size):
        writer.writerow({'id': i + j, 'label': predicted[j].item()})

# cnn = CNN()
# cnn.load(CNN_PATH)
# generate(cnn)
