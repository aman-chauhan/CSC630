# Import Libraries
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.metrics import categorical_accuracy
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
from keras.callbacks import CSVLogger
from keras.callbacks import Callback
from keras.utils import Sequence, to_categorical
from keras.models import Model

from keras.layers import GlobalMaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import MaxPooling2D
from keras.layers import UpSampling2D
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Conv2D
from keras.layers import Dense
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


# function to extract 'resnet_layers' number of layers from ResNet50
def resnet50_model(resnet_layers):
    resnet50 = ResNet50(weights='imagenet', include_top=False, input_shape=(None, None, 3))
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

# function to return the Classification layers as a Model
def classifier(encoder_name, resnet_layers, top_filters, reduction_layers, hidden_neurons, dropout, name):
    input_layer = Input(batch_shape=(None, None, None, 3),
                        name='{}_input'.format(name))

    encoder_layers = encoder(resnet_layers=resnet_layers,
                             top_filters=top_filters,
                             reduction_layers=reduction_layers,
                             name='enc')
    encoder_layers.load_weights(os.path.join('models', '{}.h5'.format(encoder_name)),
                                by_name=True)
    encoder_model = Model(inputs=encoder_layers.layers[0].input,
                          outputs=encoder_layers.layers[-1].output, name='enc_layers')

    autoencoder_layers = encoder_model(input_layer)
    globalmaxpool = GlobalMaxPooling2D(name='enc_pool')(autoencoder_layers)
    hidden_layer1 = Dense(units=hidden_neurons,
                          kernel_initializer='he_uniform',
                          bias_initializer='he_uniform',
                          name='hidden_layer_1')(globalmaxpool)
    hidden_layer1 = BatchNormalization(scale=False, name='hidden_batchnorm_1')(hidden_layer1)
    hidden_layer1 = Activation(activation='relu',
                               name='hidden_relu_1')(hidden_layer1)
    hidden_layer1 = Dropout(rate=dropout)(hidden_layer1)
    # for full and lfull models
    output = Dense(units=3, activation='softmax',
                   kernel_initializer='he_uniform',
                   bias_initializer='he_uniform',
                   name='output_layer')(hidden_layer1)
    # for xfull models, uncomment the below section and comment the above section
    # hidden_layer2 = Dense(units=hidden_neurons // 2,
    #                       kernel_initializer='he_uniform',
    #                       bias_initializer='he_uniform',
    #                       name='hidden_layer_2')(hidden_layer1)
    # hidden_layer2 = BatchNormalization(scale=False, name='hidden_batchnorm_2')(hidden_layer2)
    # hidden_layer2 = Activation(activation='relu',
    #                            name='hidden_relu_2')(hidden_layer2)
    # hidden_layer2 = Dropout(rate=dropout)(hidden_layer2)
    # output = Dense(units=3, activation='softmax',
    #                kernel_initializer='he_uniform',
    #                bias_initializer='he_uniform',
    #                name='output_layer')(hidden_layer2)
    return Model(inputs=input_layer, outputs=output, name=name)


# inspired from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
# class to feed data to the model during fit
class DataGenerator(Sequence):
    def __init__(self, list_IDs, mapping, batch_size=32, n_channel=3, shuffle=True, augment=True):
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
        self.mapping = mapping
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
        Y = []
        for i, ID in enumerate(list_IDs_temp):
            if not self.augment:
                X[i] = np.expand_dims(imread(ID), axis=0)
            else:
                seed = self.prng.randint(0, 1000000)
                X[i] = np.expand_dims(self.datagen.random_transform(imread(ID), seed=seed), axis=0)
            Y.append(self.mapping[ID.split(os.sep)[-1].split('-')[0]])
        return (preprocess_input(X), to_categorical(np.asarray(Y), num_classes=3))


# class for saving epoch counts as a Callback
class EpochCallback(Callback):
    def __init__(self, key):
        self.key = key

    def on_epoch_end(self, epoch, logs=None):
        d = {}
        if os.path.exists('clf-epochs.json'):
            d = json.load(open('clf-epochs.json'))
        d[self.key] = {}
        d[self.key]['trained'] = 'n'
        d[self.key]['epochs'] = epoch
        json.dump(d, open('clf-epochs.json', 'w'))


# function to train the parameterized model
def main(encoder_name, resnet_layers, top_filters, reduction_layers, neurons, dropout, name):
    mapping = {'DIPG': 0, 'MB': 1, 'EP': 2}
    train = None
    with open('train.txt', 'r') as fp:
        train = [x.strip() for x in fp.readlines()]
    test = None
    with open('test.txt', 'r') as fp:
        test = [x.strip() for x in fp.readlines()]
    key = name
    train_generator = DataGenerator(train, mapping, 17, 3, True, True)
    test_generator = DataGenerator(test, mapping, len(test), 3, True, True)

    K.clear_session()
    model = None
    if os.path.exists(os.path.join('models', '{}.h5'.format(key))):
        model = load_model(os.path.join('models', '{}.h5'.format(key)))
        print('Loading model from history...')
    else:
        model = classifier(encoder_name, resnet_layers, top_filters, reduction_layers, neurons, dropout, name)
        model.compile('adadelta', 'categorical_crossentropy', [categorical_accuracy])
        print('Created new model...')

    initial_epoch = None
    e = None
    if os.path.exists('clf-epochs.json'):
        e = json.load(open('clf-epochs.json'))
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
                                  patience=10,
                                  mode='min')
        tensorboard = TensorBoard(log_dir=os.path.join('logs', '{}_log'.format(key)),
                                  batch_size=13, write_graph=False,
                                  write_grads=False, write_images=True)
        csvlogger = CSVLogger(filename=os.path.join('logs', '{}.csv'.format(key)))
        history = model.fit_generator(generator=train_generator, epochs=100, verbose=1,
                                      callbacks=[epoch_cbk, csvlogger, earlystop, checkpoint, tensorboard],
                                      validation_data=test_generator,
                                      use_multiprocessing=True, workers=4,
                                      initial_epoch=initial_epoch + 1)
        e = json.load(open('clf-epochs.json'))
        e[key]['trained'] = 'y'
        json.dump(e, open('clf-epochs.json', 'w'))

    eval_results = model.evaluate_generator(generator=test_generator)
    d = {}
    if os.path.exists('clf-scores.json'):
        d = json.load(open('clf-scores.json'))
    d[key] = eval_results[1]
    json.dump(d, open('clf-scores.json', 'w'))


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]),
         int(sys.argv[5]), float(sys.argv[6]), sys.argv[7])
