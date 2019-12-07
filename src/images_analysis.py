import os
import random
import pandas as pd
import PIL.Image as Image
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import torch.utils.data.dataset
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

dir_imagenes = 'train/'
files = os.listdir(dir_imagenes)
files = [os.path.join(dir_imagenes, x) for x in files]
data_size = len(files)

widths = []
heights = []
for file in files[0:15]:
    pil_image = Image.open(file)
    print("name")
    print(file)
    print("image: width, height")
    print(pil_image.size)
    width, height = pil_image.size
    widths.append(width)
    heights.append(height)

plt.plot(widths, heights)
plt.show()

