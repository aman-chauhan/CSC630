# Import Libraries
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
from keras.callbacks import Callback
from keras.utils import Sequence
from keras.models import Model

from keras.layers import BatchNormalization
from keras.layers import MaxPooling2D
from keras.layers import UpSampling2D
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import Input

from keras.models import load_model
from keras import backend as K

from sklearn.model_selection import train_test_split
from imageio import imread

import numpy as np
import json
import math
import sys
import os


# Clear existing sessions and graphs
K.clear_session()


# load the ImageNet model
resnet50 = ResNet50(weights='imagenet', include_top=False, input_shape=(None, None, 3))


# function to extract 'resnet_layers' number of layers from ResNet50
def resnet50_model(resnet_layers):
    i = 0
    layer_map = {1: 'add_1', 2: 'add_2', 3: 'add_3'}
    while (True):
        if resnet50.layers[i].name == layer_map[resnet_layers]:
            break
        i += 1
    model = Model(inputs=resnet50.layers[0].input,
                  outputs=resnet50.layers[i].output, name='resnet_layers')
    for layer in model.layers:
        layer.trainable = False
    model.compile('adadelta', 'mse')
    return model


# function to return basic convolution block as a Model
def conv_block(prev_filters, cb_filters, cb_kernel_size, cb_strides, cb_name):
    input_layer = Input(batch_shape=(None, None, None, prev_filters),
                        name='{}_input'.format(cb_name))
    convolution_layer = Conv2D(filters=cb_filters,
                               kernel_size=cb_kernel_size,
                               strides=cb_strides,
                               padding='same',
                               kernel_initializer='he_uniform',
                               bias_initializer='he_uniform',
                               name='{}_conv2d'.format(cb_name))(input_layer)
    batchnorm_layer = BatchNormalization(scale=False,
                                         name='{}_batchnorm'.format(cb_name))(convolution_layer)
    output_layer = Activation(activation='relu',
                              name='{}_relu'.format(cb_name))(batchnorm_layer)
    return Model(inputs=input_layer, outputs=output_layer, name='{}'.format(cb_name))


# function to return Encoder as a Model
def encoder(resnet_layers, top_filters, reduction_layers, name):
    input_layer = Input(batch_shape=(None, None, None, 3),
                        name='{}_input'.format(name))
    layer = resnet50_model(resnet_layers)(input_layer)
    layer = conv_block(K.int_shape(layer)[-1],
                       top_filters,
                       3,
                       1,
                       '{}_top_{}x{}'.format(name, 3, 3))(layer)
    layer = Activation(activation='relu',
                       name='{}_top_ac'.format(name))(layer)
    layer_cnt = reduction_layers
    while (layer_cnt != 0):
        prev_filters = K.int_shape(layer)[-1]
        layer = conv_block(prev_filters,
                           prev_filters + 32,
                           3,
                           1,
                           '{}_conv_{}_{}x{}'.format(name, reduction_layers - layer_cnt, 3, 3))(layer)
        layer_cnt -= 1
        if (layer_cnt == 0):
            break
        layer = conv_block(prev_filters + 32,
                           prev_filters + 32,
                           3,
                           1,
                           '{}_conv_{}_{}x{}'.format(name, reduction_layers - layer_cnt, 3, 3))(layer)
        layer = MaxPooling2D(pool_size=(3, 3),
                             strides=(2, 2),
                             padding='same',
                             name='{}_max2d_{}{}'.format(name, reduction_layers - layer_cnt, 'pool'))(layer)
        layer_cnt -= 1
    return Model(inputs=input_layer, outputs=layer, name='{}'.format(name))


# function to return the bottle-neck as a Model
def bottleneck(prev_filters, neck_layers, name):
    input_layer = Input(batch_shape=(None, None, None, prev_filters),
                        name='{}_input'.format(name))
    layer_cnt = neck_layers
    layer = conv_block(prev_filters,
                       prev_filters,
                       3,
                       1,
                       '{}_conv_{}_{}x{}'.format(name, neck_layers - layer_cnt, 3, 3))(input_layer)
    layer_cnt -= 1
    while (layer_cnt != 0):
        layer = conv_block(prev_filters,
                           prev_filters,
                           3,
                           1,
                           '{}_conv_{}_{}x{}'.format(name, neck_layers - layer_cnt, 3, 3))(layer)
        layer_cnt -= 1
    return Model(inputs=input_layer, outputs=layer, name='{}'.format(name))


# function to return the Decoder as a Model
def decoder(prev_filters, neck_size, target_size, name):
    input_layer = Input(batch_shape=(None, None, None, prev_filters),
                        name='{}_input'.format(name))
    total_cnt = int(math.log(target_size / neck_size, 2)) - 2
    layer_cnt = total_cnt
    layer = UpSampling2D(size=(2, 2),
                         name='{}_up2d_base'.format(name))(input_layer)
    while (layer_cnt != 0):
        prev_filters = K.int_shape(layer)[-1]
        layer = conv_block(prev_filters,
                           prev_filters,
                           3,
                           1,
                           '{}_conv_{}_{}x{}'.format(name, 2 * (total_cnt - layer_cnt), 3, 3))(layer)
        layer = conv_block(prev_filters,
                           prev_filters // 2,
                           3,
                           1,
                           '{}_conv_{}_{}x{}'.format(name, 2 * (total_cnt - layer_cnt) + 1, 3, 3))(layer)
        layer = UpSampling2D(size=(2, 2),
                             name='{}_up2d_{}{}'.format(name, 2 * (total_cnt - layer_cnt) + 1, 'up'))(layer)
        layer_cnt -= 1
    output_layer = conv_block(K.int_shape(layer)[-1],
                              3,
                              3,
                              1,
                              '{}_conv_{}_{}x{}'.format(name, 2 * (total_cnt - layer_cnt) + 1, 3, 3))(layer)
    return Model(inputs=input_layer, outputs=output_layer, name='{}'.format(name))


# function to return the entire AutoEncoder as a Model
def autoencoder(size, resnet_layers, top_filters, reduction_layers, neck_layers, name):
    bottleneck_size = int(math.pow(2, 5 - math.floor(reduction_layers / 2)))
    input_layer = Input(batch_shape=(None, None, None, 3),
                        name='{}_input'.format(name))
    encoder_layers = encoder(resnet_layers=resnet_layers,
                             top_filters=top_filters,
                             reduction_layers=reduction_layers,
                             name='enc')(input_layer)
    bottleneck_layers = bottleneck(prev_filters=K.int_shape(encoder_layers)[-1],
                                   neck_layers=neck_layers,
                                   name='neck')(encoder_layers)
    decoder_layers = decoder(prev_filters=K.int_shape(bottleneck_layers)[-1],
                             neck_size=bottleneck_size,
                             target_size=size,
                             name='dec')(bottleneck_layers)
    return Model(inputs=input_layer, outputs=decoder_layers, name='{}'.format(name))


# inspired from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
# class to feed data to the model during fit
class AEDataGenerator(Sequence):
    def __init__(self, list_IDs, data_path, name_format, batch_size=32, n_channel=3, shuffle=True, augment=True):
        self.list_IDs = list_IDs
        self.batch_size = batch_size
        self.n_channel = n_channel
        self.shuffle = shuffle
        self.augment = augment
        self.prng = np.random.RandomState(10)
        self.datagen = ImageDataGenerator(rotation_range=90,
                                          width_shift_range=0.15,
                                          height_shift_range=0.15,
                                          shear_range=0.01,
                                          fill_mode='constant',
                                          cval=0,
                                          zoom_range=0.10)
        self.data_path = data_path
        self.name_format = name_format
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            self.prng.shuffle(self.indexes)

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        return self.__data_generation(list_IDs_temp)

    def __data_generation(self, list_IDs_temp):
        X = np.empty((self.batch_size, 256, 256, self.n_channel))
        Y = np.empty((self.batch_size, 256, 256, self.n_channel))

        for i, ID in enumerate(list_IDs_temp):
            if not self.augment:
                X[i] = np.expand_dims(imread(os.path.join(self.data_path,
                                                          self.name_format.format(ID))),
                                      axis=0)
            else:
                seed = self.prng.randint(0, 1000000)
                X[i] = np.expand_dims(self.datagen.random_transform(imread(os.path.join(self.data_path,
                                                                                        self.name_format.format(ID))),
                                                                    seed=seed),
                                      axis=0)
        np.copyto(Y, X)
        return (preprocess_input(X), Y)


# class for saving epoch counts as a Callback
class EpochCallback(Callback):
    def __init__(self, key):
        self.key = key

    def on_epoch_end(self, epoch, logs=None):
        d = {}
        if os.path.exists('r50-epochs.json'):
            d = json.load(open('r50-epochs.json'))
        d[self.key] = {}
        d[self.key]['trained'] = 'n'
        d[self.key]['epochs'] = epoch
        json.dump(d, open('r50-epochs.json', 'w'))


# function to train the parameterized model
def ae_cv_train(ae_config, epochs, train, test, mode, batch_size):
    cnt = 0
    key = 'train_resnet_50'

    # generators for normalized dataset
    training_generator = AEDataGenerator(train, os.path.join('train_ae_data', mode), 'IMG{:06d}.png',
                                         batch_size, 3, True, True)
    validation_generator = AEDataGenerator(test, os.path.join('train_ae_data', mode), 'IMG{:06d}.png',
                                           3000, 3, True, True)

    # generators for normalized dataset
    # training_generator = AEDataGenerator(train, os.path.join('ae_data', mode), 'IMG{:06d}.png',
    #                                      batch_size, 3, True, True)
    # validation_generator = AEDataGenerator(test, os.path.join('ae_data', mode), 'IMG{:06d}.png',
    #                                        3000, 3, True, True)

    model = None
    if os.path.exists(os.path.join('models', '{}.h5'.format(key))):
        model = load_model(os.path.join('models', '{}.h5'.format(key)))
        print('Loading model from history...')
    else:
        model = autoencoder(size=ae_config['size'],
                            resnet_layers=ae_config['resnet_layers'],
                            top_filters=ae_config['top_filters'],
                            reduction_layers=ae_config['reduction_layers'],
                            neck_layers=ae_config['neck_layers'],
                            name=ae_config['name'])
        model.compile('adadelta', 'mse', ['mse', 'mae'])
        print('Created new model...')

    initial_epoch = None
    e = None
    if os.path.exists('r50-epochs.json'):
        e = json.load(open('r50-epochs.json'))
        if key in e:
            initial_epoch = e[key]['epochs']
        else:
            e[key] = {'trained': 'n'}
            initial_epoch = -1
    else:
        e = {key: {'trained': 'n'}}
        initial_epoch = -1

    if not os.path.exists(os.path.join('logs', '{}_log'.format(key))):
        os.mkdir(os.path.join('logs', '{}_log'.format(key)))
    if e[key]['trained'] == 'n':
        epoch_cbk = EpochCallback(key)
        checkpoint = ModelCheckpoint(filepath=os.path.join('models', '{}.h5'.format(key)),
                                     monitor='val_loss', verbose=1,
                                     save_best_only=True, mode='min')
        earlystop = EarlyStopping(monitor='val_loss',
                                  patience=5,
                                  mode='min')
        tensorboard = TensorBoard(log_dir=os.path.join('logs', '{}_log'.format(key)), histogram_freq=1,
                                  batch_size=batch_size, write_graph=False,
                                  write_grads=False, write_images=True)
        history = model.fit_generator(generator=training_generator, epochs=epochs, verbose=1,
                                      callbacks=[epoch_cbk, earlystop, checkpoint, tensorboard],
                                      validation_data=validation_generator.__getitem__(0),
                                      use_multiprocessing=True, workers=4,
                                      initial_epoch=initial_epoch + 1)
        e = json.load(open('r50-epochs.json'))
        e[key]['trained'] = 'y'
        json.dump(e, open('r50-epochs.json', 'w'))
    eval_results = model.evaluate_generator(generator=validation_generator,
                                            use_multiprocessing=True, workers=4)

    d = {}
    if os.path.exists('scores.json'):
        d = json.load(open('scores.json'))
    d[key] = eval_results[1]
    json.dump(d, open('scores.json', 'w'))


# main driver function
def main(i, j, k, l):
    X = [x for x in range(75000)]
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=10, shuffle=False, stratify=None)
    if not os.path.exists('models'):
        os.mkdir('models')
    if not os.path.exists('logs'):
        os.mkdir('logs')
    d = {}
    if os.path.exists('scores.json'):
        d = json.load(open('scores.json'))
    ae_config = {'resnet_layers': i, 'top_filters': j,
                 'reduction_layers': k, 'neck_layers': l,
                 'size': 256, 'name': 'autoencoder'}
    # replace X_train with X to train on full dataset
    ae_cv_train(ae_config, 100, X_train, X_test, 'T2W', 30)


if __name__ == '__main__':
    main(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
