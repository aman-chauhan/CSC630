# Normalized Preprocessing

# import libraries
from os.path import isfile, join, isdir
from os import listdir, mkdir, sep
from PIL import Image

import nibabel as nib
import numpy as np
import imageio
import json
import sys


def main(split, mode):
    # parameters
    train_src = join('datasets', join('BRAIN_TUMOR', 'imagesTr'))
    test_src = join('datasets', join('BRAIN_TUMOR', 'imagesTs'))
    folder_map = {'FLAIR': 0, 'T1W': 1, 'T1GD': 2, 'T2W': 3}
    sample_format = 'BRATS_{:03d}.nii.gz'
    data_folder = 'train_ae_data'

    # create folder if not created
    if not isdir(data_folder):
        mkdir(data_folder)
        if not isdir(join(data_folder, mode)):
            mkdir(join(data_folder, mode))
    else:
        if not isdir(join(data_folder, mode)):
            mkdir(join(data_folder, mode))
    print('Folders created.')

    # Find min and max from training set if it doesn't exist
    # else fetch from json
    print('Starting to calculate Min and Max.')
    minval = float('inf')
    maxval = float('-inf')
    if not isfile('min_max.json'):
        if split < 485:
            for i in range(1, split):
                sample = join(train_src, sample_format.format(i))
                img = nib.load(sample).get_fdata()
                min_img = img[:, :, 25:125, folder_map[mode]].min()
                max_img = img[:, :, 25:125, folder_map[mode]].max()
                if min_img < minval:
                    minval = min_img
                if max_img > maxval:
                    maxval = max_img
                print('.', end='')
        else:
            for i in range(1, 485):
                sample = join(train_src, sample_format.format(i))
                img = nib.load(sample).get_fdata()
                min_img = img[:, :, 25:125, folder_map[mode]].min()
                max_img = img[:, :, 25:125, folder_map[mode]].max()
                if min_img < minval:
                    minval = min_img
                if max_img > maxval:
                    maxval = max_img
                print('.', end='')
            for i in range(485, split):
                sample = join(test_src, sample_format.format(i))
                img = nib.load(sample).get_fdata()
                min_img = img[:, :, 25:125, folder_map[mode]].min()
                max_img = img[:, :, 25:125, folder_map[mode]].max()
                if min_img < minval:
                    minval = min_img
                if max_img > maxval:
                    maxval = max_img
                print('.', end='')
        d = {'min': minval, 'max': maxval}
        json.dump(d, open('min_max.json', 'w'))
        print('Min and Max saved.')
    else:
        d = json.load(open('min_max.json', 'r'))
        minval = d['min']
        maxval = d['max']
    print('Min and Max calculated. Min=' + str(minval) + ', Max=' + str(maxval))

    cnt = 0
    # normalize images
    for i in range(1, 485):
        sample = join(train_src, sample_format.format(i))
        img = nib.load(sample).get_fdata()
        for z in range(25, 125):
            fmg = ((img[:, :, z, folder_map[mode]] - minval) / (maxval - minval)) * 255
            fmg = np.clip(fmg, 0, 255)
            fmg = fmg.astype('uint8')
            fmg = Image.fromarray(np.repeat(np.expand_dims(fmg, axis=-1), repeats=3, axis=-1), 'RGB')
            fmg = fmg.resize(size=(256, 256), resample=Image.BILINEAR)
            fmg.save(join(data_folder, join(mode, 'IMG{:06d}.png'.format(cnt))))
            cnt += 1
        print('.', end='')
    for i in range(485, 751):
        sample = join(test_src, sample_format.format(i))
        img = nib.load(sample).get_fdata()
        for z in range(25, 125):
            fmg = ((img[:, :, z, folder_map[mode]] - minval) / (maxval - minval)) * 255
            fmg = np.clip(fmg, 0, 255)
            fmg = fmg.astype('uint8')
            fmg = Image.fromarray(np.repeat(np.expand_dims(fmg, axis=-1), repeats=3, axis=-1), 'RGB')
            fmg = fmg.resize(size=(256, 256), resample=Image.BILINEAR)
            fmg.save(join(data_folder, join(mode, 'IMG{:06d}.png'.format(cnt))))
            cnt += 1
        print('.', end='')
    print('Images generated.')


if __name__ == '__main__':
    main(int(sys.argv[1]), sys.argv[2])
