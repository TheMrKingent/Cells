'''
Preprocess e csv writing

Data preprocess:
* rescale to equal size (32 x 32)
* convert to grayscale
* mask black pixels
* join mask and grayscale
'''

from PIL import Image, ImageEnhance
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
    
    # B/W
    bw = img.convert(mode='L')
    enhancer = ImageEnhance.Contrast(bw)
    bw = enhancer.enhance(10)
    
    # Mask
    mask = img
    pixels_mask = mask.load()
    for i in range(mask.size[0]): # for every pixel:
        for j in range(mask.size[1]):
            if pixels_mask[i,j] == (0,0,0): # if not black:
                pixels_mask[i,j] = (255, 255, 255) # change to white
    
    # Join Mask ove B/W
    pixels_bw = bw.load()
    for i in range(mask.size[0]):
        for j in range(mask.size[1]):
            if pixels_mask[i,j] == (255, 255, 255):
                pixels_bw[i,j] = 255
   
    array = np.array(bw).reshape(gt_size[0]*gt_size[1])
    row = np.append(0, array)  # append label 0
    row_list.append(row)

# Parasitized (label = 1)
names1 = os.listdir(os.path.join('data', 'Parasitized'))
for name in names1:
    img = Image.open(os.path.join('data', 'Parasitized', name))
    img = img.resize(gt_size)
    
    # B/W
    bw = img.convert(mode='L')
    enhancer = ImageEnhance.Contrast(bw)
    bw = enhancer.enhance(10)
    
    # Mask
    mask = img
    pixels_mask = mask.load()
    for i in range(mask.size[0]): # for every pixel:
        for j in range(mask.size[1]):
            if pixels_mask[i,j] == (0,0,0): # if not black:
                pixels_mask[i,j] = (255, 255, 255) # change to white
    
    # Join Mask ove B/W
    pixels_bw = bw.load()
    for i in range(mask.size[0]):
        for j in range(mask.size[1]):
            if pixels_mask[i,j] == (255, 255, 255):
                pixels_bw[i,j] = 255
   
    array = np.array(bw).reshape(gt_size[0]*gt_size[1])
    row = np.append(1, array)  # append label 0
    row_list.append(row)

# Write Df
df = pd.DataFrame(row_list, columns=colnames)

# Write csv
df.to_csv('dataset/dat_bwstain.csv', index=False)