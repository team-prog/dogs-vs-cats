import os
import PIL.Image as Image
import matplotlib.pyplot as plt
from Dataset import DogsVsCatsDataset
from random import randrange

######## Constants #########
dataset_path = './train'
data_size = 1000
dataset = DogsVsCatsDataset(data_dir = dataset_path, data_size = data_size)

def show_folder_images(dir_path, quantity = 0):
  files = os.listdir(dir_path)
  files = [os.path.join(dir_path, x) for x in files]
  widths = []
  heights = []
  for file in files[0:quantity]:
      pil_image = Image.open(file)
      print("name: ", file)
      width, height = pil_image.size
      print("image: width, height", pil_image.size)
      widths.append(width)
      heights.append(height)
  plt.plot(widths, heights)
  plt.show()

def show_image(dataset, image_number):
    image, label = dataset[image_number]
    image = image.numpy()
    image = image.transpose(1,2,0)
    plt.imshow(image)
    plt.show(block=False)
    print("showing: ", image_number, " label: ", label.item())
    print("waiting...")
    plt.pause(3)
    print("closing...")
    plt.close()
    show_image(dataset, image_number+1)

image_number = randrange(0, len(dataset) - 1)
show_image(dataset, image_number)

