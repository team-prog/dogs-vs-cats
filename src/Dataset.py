import os
import random
import pandas as pd
import PIL.Image as Image
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data.dataset
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

#Algunos parámetros:
#dir_imagenes: Directorio donde están las imágenes extraídas del 7zip de Kaggle
#(Las que valen son las de TRAIN, que son aquellas que tienen las etiquetas en 
# el archivo trainLabels.csv)
# dir_imagenes = 'train/'
#cant_archivos: Cuantos archivos del repositorio usar.
#El valor 0 significa usar todos.
#Se puede poner un número arbitrario para pruebas
# cant_archivos = 0
#ruta_trainlabels: Ruta del archivo trainLabels.csv (relativa a donde está este .py)
# ruta_trainlabels = 'trainLabels.csv'

#Constructor para el Dataset basado en las imágenes
# class Cifar10Dataset(torch.utils.data.Dataset):

labels = {'label': ['cat', 'dog'] }
labels_encoder = LabelEncoder()
number_labels = labels_encoder.fit_transform(labels['label'])

class DogsVsCatsDataset(torch.utils.data.Dataset):
    #data_dir: El directorio del que se leerán las imágenes
    #label_source: De dónde se obtendrán las etiquetas
    #data_size: Cuantos archivos usar (0 = todos)
    def __init__(self, data_dir, label_source = number_labels, data_size = 0):
        files = os.listdir(data_dir)
        files = [os.path.join(data_dir,x) for x in files]
        if data_size < 0 or data_size > len(files):
            assert("Data size should be between 0 to number of files in the dataset")
        if data_size == 0:
            data_size = len(files)
        self.data_size = data_size
        self.files = random.sample(files, self.data_size)
        self.label_source = label_source
        
    def __len__(self):
        return self.data_size
    
    def __getitem__(self, idx):
        image_address = self.files[idx]
        pil_image = Image.open(image_address)
        # transformation = transforms.Resize((32, 32))
        transformation = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(64),
            # transforms.CenterCrop(64)
        ])
        resized_image = transformation(pil_image)
        numpy_array_image = np.array(resized_image)
        numpy_array_image = numpy_array_image / 255
        numpy_array_image = numpy_array_image.transpose(2, 0, 1)
        tensor_image = torch.Tensor(numpy_array_image)        
        if "dog" in image_address:
            label_idx = 1
        else:
            label_idx = 0
        label = self.label_source[label_idx]
        label = torch.tensor(label).long()
        return tensor_image, label
