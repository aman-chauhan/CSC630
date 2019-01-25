# coding: utf-8

# # Classification Preprocessing

# ## Import Libraries


from sklearn.model_selection import train_test_split
from PIL import Image

import numpy as np
import pydicom
import json
import os


# ## Fetch list of all files and the classification of each file


mapping = {'DIPG': 0, 'MB': 1, 'EP': 2}
x = []
y = []
for root, dirs, files in os.walk('datasets/Stanford', topdown=True):
    if len(root.split('/')) >= 3:
        for name in sorted(files):
            if not name.startswith('.'):
                path = '-'.join(os.path.join(root, name).split('-')[:2])
                if path not in x:
                    x.append(path)
                    y.append(mapping[root.split('/')[2]])


# ## Stratified Split into Train/Test


X_train, X_test = train_test_split(x, test_size=0.1,
                                   random_state=10,
                                   shuffle=True,
                                   stratify=y)


# ## Fetch Train filenames


X_train_files = []
Y_train = []
for root, dirs, files in os.walk('datasets/Stanford', topdown=True):
    for name in sorted(files):
        path = os.path.join(root, name)
        if any([path.startswith(x) for x in X_train]):
            X_train_files.append(path)
            Y_train.append(mapping[root.split('/')[2]])


# ## Fetch Test filenames


X_test_files = []
Y_test = []
for root, dirs, files in os.walk('datasets/Stanford', topdown=True):
    for name in sorted(files):
        path = os.path.join(root, name)
        if any([path.startswith(x) for x in X_test]):
            X_test_files.append(path)
            Y_test.append(mapping[root.split('/')[2]])


# ## Find Training Maxima and Minima


minval = float('inf')
maxval = float('-inf')
for i, filename in enumerate(X_train_files):
    img = pydicom.dcmread(filename).pixel_array
    if img.min() < minval:
        minval = img.min()
    if img.max() > maxval:
        maxval = img.max()


# ## Save the maxima and minima


minval = int(minval)
maxval = int(maxval)
d = {'min': minval, 'max': maxval}
json.dump(d, open('classification_min_max.json', 'w'))
print('Min and Max saved.')


# ## Normalize Training Data


if not os.path.exists('normal_classifier_data'):
    os.mkdir('normal_classifier_data')
    os.mkdir(os.path.join('normal_classifier_data', 'train'))
    os.mkdir(os.path.join('normal_classifier_data', 'test'))

for i, filename in enumerate(X_train_files):
    img = pydicom.dcmread(filename).pixel_array
    img = (img - minval) * 255.0 / (maxval - minval)
    img = np.clip(img, 0, 255)
    img = img.astype('uint8')
    img = Image.fromarray(np.repeat(np.expand_dims(img, axis=-1), repeats=3, axis=-1), 'RGB')
    ratio = float(256) / max(img.size)
    new_size = tuple([int(x * ratio) for x in img.size])
    img = img.resize(size=new_size, resample=Image.LANCZOS)
    fmg = Image.new("RGB", (256, 256))
    fmg.paste(img, ((256 - new_size[0]) // 2,
                    (256 - new_size[1]) // 2))
    fmg = fmg.rotate(270, resample=Image.BILINEAR)
    fmg.save(os.path.join(os.path.join('normal_classifier_data', 'train'),
                          filename.split(os.sep)[-1].split('.')[0] + '.png'))
    print(i)


# ##  Normalize Testing Data


for i, filename in enumerate(X_test_files):
    img = pydicom.dcmread(filename).pixel_array
    img = (img - minval) * 255.0 / (maxval - minval)
    img = np.clip(img, 0, 255)
    img = img.astype('uint8')
    img = Image.fromarray(np.repeat(np.expand_dims(img, axis=-1), repeats=3, axis=-1), 'RGB')
    ratio = float(256) / max(img.size)
    new_size = tuple([int(x * ratio) for x in img.size])
    img = img.resize(size=new_size, resample=Image.LANCZOS)
    fmg = Image.new("RGB", (256, 256))
    fmg.paste(img, ((256 - new_size[0]) // 2,
                    (256 - new_size[1]) // 2))
    fmg = fmg.rotate(270, resample=Image.BILINEAR)
    fmg.save(os.path.join(os.path.join('normal_classifier_data', 'test'),
                          filename.split(os.sep)[-1].split('.')[0] + '.png'))
    print(i)


# ## Generate training and testing filenames


with open('train.txt', 'w') as fp:
    for i, filename in enumerate(X_train_files):
        fp.write(os.path.join(os.path.join('classifier_data', 'train'),
                              filename.split(os.sep)[-1].split('.')[0] + '.png') + '\n')


with open('test.txt', 'w') as fp:
    for i, filename in enumerate(X_test_files):
        fp.write(os.path.join(os.path.join('classifier_data', 'test'),
                              filename.split(os.sep)[-1].split('.')[0] + '.png') + '\n')
