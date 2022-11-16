'''
Preprocess e csv writing

Data preprocess:
* rescale to equal size (32 x 32)
* convert to grayscale
* mask black pixels
* join mask and grayscale
'''

from PIL import Image, ImageEnhance, ImageOps
import os
import numpy as np
import pandas as pd


# Set wd to current file's location
file_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(file_path)

gt_size = (32,32)
row_list = []
folders = ['Uninfected', 'Parasitized']


for cat in [0,1]:

    names = os.listdir(os.path.join('data', folders[cat]))
    for name in names:
        
        img = Image.open(os.path.join('data', folders[cat], name))
        img = img.resize(gt_size)
        
        # Create B/W contrast enhanced
        opt = img.convert(mode='L')
        enhancer = ImageEnhance.Contrast(opt)
        opt = enhancer.enhance(10)
        
        # Create Mask
        mask = img
        pixels_mask = mask.load()
        for i in range(mask.size[0]):
            for j in range(mask.size[1]):
                if pixels_mask[i,j] == (0,0,0):
                    pixels_mask[i,j] = (255, 255, 255)
        
        # Join Mask over B/W
        pixels_opt = opt.load()
        for i in range(mask.size[0]):
            for j in range(mask.size[1]):
                if pixels_mask[i,j] == (255, 255, 255):
                    pixels_opt[i,j] = 255
        
        # Invert
        opt = ImageOps.invert(opt)
        
        # Finalize
        array = np.array(opt).reshape(gt_size[0]*gt_size[1])
        row = np.append(i, array)  # append label
        row_list.append(row)


# Write Df
colnames = ['label'] + list(range(gt_size[0]*gt_size[1]))
df = pd.DataFrame(row_list, columns=colnames)

# Write csv
df.to_csv('dataset/dat_inverseStain.csv', index=False)