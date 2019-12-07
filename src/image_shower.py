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
class DogsVsCatsDataset(torch.utils.data.Dataset):
    #data_dir: El directorio del que se leerán las imágenes
    #label_source: De dónde se obtendrán las etiquetas
    #data_size: Cuantos archivos usar (0 = todos)
    def __init__(self, data_dir, label_source, data_size = 0):
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
        # resize = transforms.Resize(1024)
        
        resized_image = transforms.RandomCrop(32)(pil_image)
        
        
        
        #Se deja los valores de la imágen en el rango 0-1
        image = np.array(resized_image)
        image = image / 255
        #Se traspone la imagen para que el canal sea la primer coordenada
        # (la red espera NxMx3)
        
        
        image = image.transpose(2,0,1)
        image = torch.Tensor(image)
        #Se puede agregar: Aplicar normalización (Hacer que los valores vayan
        #entre -1 y 1 pero con el 0 en el valor promedio.
        #Los parámetros estos están precalculados para el set CIFAR-10 
        #image = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))(image)
        # label_idx = int(image_address[:-4].split("/")[1])-1
        
        if "dog" in image_address:
            label_idx = 1
        else:
            label_idx = 0
        label = self.label_source[label_idx]
        label = torch.tensor(label).long()
        return image, label

#Levantamos los labels del archivo csv
# labels = pd.read_csv(ruta_trainlabels)
# labels = {'label': ['cat', 'dog'] }

# #Lo transformamos a números con un labelEncoder
# #labels_encoder es importante: Es el que me va a permitir revertir la 
# #transformación para conocer el nombre de una etiqueta numérica
# labels_encoder = LabelEncoder()
# labels_numeros=labels_encoder.fit_transform(labels['label'])

# #Generamos el DataSet con nuestros datos de entrenamiento
# # cifar_dataset = Cifar10Dataset(data_dir = dir_imagenes, data_size = cant_archivos, label_source = labels_numeros)
# cifar_dataset = DogsVsCatsDataset(data_dir = dir_imagenes, data_size = cant_archivos, label_source=labels_numeros)

##Antes de pasar a la separación en datos de training y test, podemos verificar
##que estamos levantando las imágenes de manera correcta. Defino una función que
##dado un número toma la imágen en esa posición del dataset (Ojo, recordar que
##está mezclado), y grafica la imágen junto con su etiqueta.
# def mostrarImagen(dataset, nroImagen, encoder):
#     imagen, etiqueta = dataset[nroImagen]
#     #Se regresa la imágen a formato numpy
#     #Es necesario trasponer la imágen para que funcione con imshow
#     #(imshow espera 3xNxM)
#     imagen = imagen.numpy()
#     imagen = imagen.transpose(1,2,0)
#     plt.imshow(imagen)
#     #Recupero la etiqueta de la imágen usando el encoder
#     plt.title(labels_encoder.inverse_transform([etiqueta])[0])

# mostrarImagen(cifar_dataset, 10, labels_encoder)
# plt.show()
