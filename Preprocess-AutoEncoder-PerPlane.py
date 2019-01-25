# Import Libraries
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from os.path import isfile, join, isdir
from os import listdir, mkdir, sep
from PIL import Image

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import imageio
import json


# Parameters
train_src = join('datasets', join('BRAIN_TUMOR', 'imagesTr'))
test_src = join('datasets', join('BRAIN_TUMOR', 'imagesTs'))
#folder_map = {0:'FLAIR', 1:'T1W', 2:'T1GD', 3:'T2W'}
folder_map = {3: 'T2W'}
cm = ScalarMappable(norm=None, cmap='gray')
sample_format = 'BRATS_{:03d}.nii.gz'
data_folder = 'ae_data'


# Create folders if not present
if not isdir(data_folder):
    mkdir(data_folder)
    for x in folder_map.keys():
        mkdir(join(data_folder, folder_map[x]))
else:
    for x in folder_map.keys():
        if not isdir(join(data_folder, folder_map[x])):
            mkdir(join(data_folder, folder_map[x]))

# img = nib.load(join(test_src, sample_format.format(490))).get_fdata()
# print(np.repeat(np.expand_dims(img[:,:,25,3], axis=-1), repeats=3, axis=-1).shape)
cnt = 0
# Normalize Images
for i in range(1, 485):
    sample = join(train_src, sample_format.format(i))
    img = nib.load(sample).get_fdata()
    for z in range(25, 125):
        for m in range(4):
            fmg = ((img[:, :, z, m] - img[:, :, z, m].min()) / (img[:, :, z, m].max() - img[:, :, z, m].min())) * 255
            fmg = np.clip(fmg, 0, 255)
            fmg = fmg.astype('uint8')
            fmg = Image.fromarray(cm.to_rgba(fmg, bytes=True), 'RGBA')
            background = Image.new("RGB", fmg.size)
            background.paste(fmg, mask=fmg.split()[3])
            fmg = background.resize(size=(256, 256), resample=Image.BILINEAR)
            fmg.save(join(data_folder, join(folder_map[m], 'IMG{:06d}.png'.format(cnt))))
        cnt += 1
    print('.', end='')

train_cnt = cnt
print(train_cnt)

cnt = train_cnt
for i in range(485, 751):
    sample = join(test_src, sample_format.format(i))
    img = nib.load(sample).get_fdata()
    for z in range(25, 125):
        for m in range(4):
            fmg = ((img[:, :, z, m] - img[:, :, z, m].min()) / (img[:, :, z, m].max() - img[:, :, z, m].min())) * 255
            fmg = np.clip(fmg, 0, 255)
            fmg = fmg.astype('uint8')
            fmg = Image.fromarray(cm.to_rgba(fmg, bytes=True), 'RGBA')
            background = Image.new("RGB", fmg.size)
            background.paste(fmg, mask=fmg.split()[3])
            fmg = background.resize(size=(256, 256), resample=Image.BILINEAR)
            fmg.save(join(data_folder, join(folder_map[m], 'IMG{:06d}.png'.format(cnt))))
        cnt += 1
    print('.', end='')

print(cnt)
