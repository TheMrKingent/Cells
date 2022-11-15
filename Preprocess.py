'''
Preprocess e csv writing

Data preprocess:
* convert to grayscale
* rescale to equal size (32 x 32)
'''

from PIL import Image
import os
import numpy as np
import pandas as pd


# Set wd to current file's location
file_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(file_path)

gt_size = (32,32)
row_list = []
colnames = ['label'] + list(range(gt_size[0]*gt_size[1]))

# Uninfected (label = 0)
names0 = os.listdir(os.path.join('data', 'Uninfected'))
for name in names0:
    img = Image.open(os.path.join('data', 'Uninfected', name))
    img = img.resize(gt_size)
    img = img.convert(mode='L')  # convert to gray scale
    array = np.array(img).reshape(gt_size[0]*gt_size[1])
    row = np.append(0, array)  # append label 0
    row_list.append(row)

# Parasitized (label = 1)
names1 = os.listdir(os.path.join('data', 'Parasitized'))
for name in names1:
    img = Image.open(os.path.join('data', 'Parasitized', name))
    img = img.resize(gt_size)
    img = img.convert(mode='L')  # convert to gray scale
    array = np.array(img).reshape(gt_size[0]*gt_size[1])
    row = np.append(1, array)  # append label 1
    row_list.append(row)

df = pd.DataFrame(row_list, columns=colnames)

# Write csv
df.to_csv('dat.csv', index=False)