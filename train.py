# %%

import numpy as np
import os
import glob
import time
from random import shuffle
from imageio import imread
import tensorflow as tf

import keras.backend as K
from keras.models import Model
from keras.layers import Conv2D, ZeroPadding2D, \
    BatchNormalization, Input, Dropout
from keras.layers import Conv2DTranspose, Activation, Cropping2D
from keras.layers import Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.initializers import RandomNormal
from keras.optimizers import Adam


# configure os environment
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# configure keras
K.set_image_data_format('channels_last')
CH_AXIS = -1

# configure tensorflow
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.Session(config=config)


# Hyper parameters
NITERS = 6  # total number of iterations
DISPLAY_ITERS = 2  # number of iterations before display

# the input data:
# (originally AIA or Atmospheric Imaging Assembly)
INPUT_DATA = 'TEST_INPUT'
# The data we want to reproduce:
# (originally HMI or Helioseismic and Magnetic Imager)
OUTPUT_DATA = 'TEST_OUTPUT'

ISIZE = 1024  # height of the image
NC_IN = 1  # number of input channels (1 for greyscale, 3 for RGB)
NC_OUT = 1  # number of output channels (1 for greyscale, 3 for RGB)
BATCH_SIZE = 1  # number of images in each batch
# number of layers
# 1 for 16, 2 for 34, 3 for 70, 4 for 142, and 5 for 286
MAX_LAYERS = 3

TRIAL_NAME = 'TEST' + str(MAX_LAYERS)

# %%

MODE = INPUT_DATA + '_to_' + OUTPUT_DATA  # folder name for saving the model

IMAGE_PATH_INPUT = './DATA/TRAIN/'+INPUT_DATA+'/*.png'  # input file path
IMAGE_PATH_OUTPUT = './DATA/TRAIN/'+OUTPUT_DATA+'/*.png'  # ouptut file path

# make a folder for the trial if it doesn't already exist
MODEL_PATH_MAIN = './MODELS/' + TRIAL_NAME + '/'
os.mkdir(MODEL_PATH_MAIN) if not os.path.exists(MODEL_PATH_MAIN) else None
MODEL_PATH = MODEL_PATH_MAIN + MODE + '/'
os.mkdir(MODEL_PATH) if not os.path.exists(MODEL_PATH) else None

# %%
# generates tensors with a normal distribution with (mean, standard deviation)
# this is used as a matrix of weights
CONV_INIT = RandomNormal(0, 0.02)
GAMMA_INIT = RandomNormal(1., 0.02)


# create a convolutional layer with f filters, and arguments a and k
def DN_CONV(f, *a, **k):
    return Conv2D(f, kernel_initializer=CONV_INIT, *a, **k)


# create a deconvolutional layer with f filters, and arguments a and k
def UP_CONV(f, *a, **k):
    return Conv2DTranspose(f, kernel_initializer=CONV_INIT, *a, **k)


# applies normalisation such that max is 1, and minimum is 0
def BATNORM():
    return BatchNormalization(
                              momentum=0.9,
                              axis=CH_AXIS,
                              epsilon=1.01e-5,
                              gamma_initializer=GAMMA_INIT
                              )


# leaky ReLU (y = alpha*x for x < 0, y = x for x > 0)
def LEAKY_RELU(alpha):
    return LeakyReLU(alpha)


#  the descriminator
def BASIC_D(ISIZE, NC_IN, NC_OUT, MAX_LAYERS):
    INPUT_A, INPUT_B = Input(shape=(ISIZE, ISIZE, NC_IN)),
    Input(shape=(ISIZE, ISIZE, NC_OUT))

    INPUT = Concatenate(axis=CH_AXIS)([INPUT_A, INPUT_B])

    if MAX_LAYERS == 0:
        N_FEATURE = 1
        L = DN_CONV(N_FEATURE,
                    kernel_size=1,
                    padding='same',
                    activation='sigmoid'
                    )(INPUT)

    else:
        N_FEATURE = 64
        L = DN_CONV(N_FEATURE,
                    kernel_size=4,
                    strides=2,
                    padding="same"
                    )(INPUT)
        L = LEAKY_RELU(0.2)(L)

        for i in range(1, MAX_LAYERS):
            N_FEATURE *= 2
            L = DN_CONV(N_FEATURE,
                        kernel_size=4,
                        strides=2,
                        padding="same"
                        )(L)
            L = BATNORM()(L, training=1)
            L = LEAKY_RELU(0.2)(L)

        N_FEATURE *= 2
        L = ZeroPadding2D(1)(L)
        L = DN_CONV(N_FEATURE, kernel_size=4, padding="valid")(L)
        L = BATNORM()(L, training=1)
        L = LEAKY_RELU(0.2)(L)

        N_FEATURE = 1
        L = ZeroPadding2D(1)(L)
        L = DN_CONV(N_FEATURE,
                    kernel_size=4,
                    padding="valid",
                    activation='sigmoid'
                    )(L)

    return Model(inputs=[INPUT_A, INPUT_B], outputs=L)


# The generator (based on the U-Net architecture)
def UNET_G(ISIZE, NC_IN, NC_OUT, FIXED_INPUT_SIZE=True):
    MAX_N_FEATURE = 64 * 8

    def BLOCK(X, S, NF_IN, USE_BATNORM=True, NF_OUT=None, NF_NEXT=None):
        assert S >= 2 and S % 2 == 0
        if NF_NEXT is None:
            NF_NEXT = min(NF_IN*2, MAX_N_FEATURE)
        if NF_OUT is None:
            NF_OUT = NF_IN
        X = DN_CONV(NF_NEXT,
                    kernel_size=4,
                    strides=2,
                    use_bias=(not (USE_BATNORM and S > 2)),
                    padding="same"
                    )(X)
        if S > 2:
            if USE_BATNORM:
                X = BATNORM()(X, training=1)
            X2 = LEAKY_RELU(0.2)(X)
            X2 = BLOCK(X2, S//2, NF_NEXT)
            X = Concatenate(axis=CH_AXIS)([X, X2])
        X = Activation("relu")(X)
        X = UP_CONV(NF_OUT,
                    kernel_size=4,
                    strides=2,
                    use_bias=not USE_BATNORM
                    )(X)
        X = Cropping2D(1)(X)
        if USE_BATNORM:
            X = BATNORM()(X, training=1)
        if S <= 8:
            X = Dropout(0.5)(X, training=1)
        return X

    S = ISIZE if FIXED_INPUT_SIZE else None
    X = INPUT = Input(shape=(S, S, NC_IN))
    X = BLOCK(X, ISIZE, NC_IN, False, NF_OUT=NC_OUT, NF_NEXT=64)
    X = Activation('tanh')(X)

    return Model(inputs=INPUT, outputs=[X])


# %%

NET_D = BASIC_D(ISIZE, NC_IN, NC_OUT, MAX_LAYERS)
NET_G = UNET_G(ISIZE, NC_IN, NC_OUT)

REAL_A = NET_G.input
FAKE_B = NET_G.output
REAL_B = NET_D.inputs[1]

OUTPUT_D_REAL = NET_D([REAL_A, REAL_B])
OUTPUT_D_FAKE = NET_D([REAL_A, FAKE_B])


def LOSS_FN(OUTPUT, TARGET):
    return -K.mean(K.log(OUTPUT+1e-12)*TARGET+K.log(1-OUTPUT+1e-12)*(1-TARGET))


# LOSS_FN = lambda OUTPUT, TARGET : -K.mean(K.log(OUTPUT+1e-12)*TARGET+K.log(1-OUTPUT+1e-12)*(1-TARGET))

LOSS_D_REAL = LOSS_FN(OUTPUT_D_REAL, K.ones_like(OUTPUT_D_REAL))
LOSS_D_FAKE = LOSS_FN(OUTPUT_D_FAKE, K.zeros_like(OUTPUT_D_FAKE))
LOSS_G_FAKE = LOSS_FN(OUTPUT_D_FAKE, K.ones_like(OUTPUT_D_FAKE))

LOSS_L = K.mean(K.abs(FAKE_B-REAL_B))


LOSS_D = LOSS_D_REAL + LOSS_D_FAKE
TRAINING_UPDATES_D = Adam(lr = 2e-4, beta_1 = 0.5).get_updates(NET_D.trainable_weights, [], LOSS_D)
NET_D_TRAIN = K.function([REAL_A, REAL_B], [LOSS_D/2.0], TRAINING_UPDATES_D)

LOSS_G = LOSS_G_FAKE + 100 * LOSS_L
TRAINING_UPDATES_G = Adam(lr = 2e-4, beta_1 = 0.5).get_updates(NET_G.trainable_weights, [], LOSS_G)
NET_G_TRAIN = K.function([REAL_A, REAL_B], [LOSS_G_FAKE, LOSS_L], TRAINING_UPDATES_G)

#%%

def LOAD_DATA(FILE_PATTERN):
    return glob.glob(FILE_PATTERN)

def READ_IMAGE(FN, NC_IN, NC_OUT):
    IMG_A = imread(FN[0])
    IMG_B = imread(FN[1])
    X, Y = np.random.randint(31), np.random.randint(31)

    if NC_IN != 1 :
        IMG_A = np.pad(IMG_A, ((15, 15), (15, 15), (0, 0)), 'constant')
        IMG_A = IMG_A[X:X + 1024, Y:Y + 1024,:] / 255.0 * 2 - 1
    else :
        IMG_A = np.pad(IMG_A, 15, 'constant')
        IMG_A = IMG_A[X:X + 1024, Y:Y + 1024] / 255.0 * 2 - 1

    if NC_OUT != 1 :
        IMG_B = np.pad(IMG_B, ((15, 15), (15, 15), (0, 0)), 'constant')
        IMG_B = IMG_B[X:X + 1024, Y:Y + 1024,:] / 255.0 * 2 - 1
    else :
        IMG_B = np.pad(IMG_B, 15, 'constant')
        IMG_B = IMG_B[X:X + 1024, Y:Y + 1024] / 255.0 * 2 - 1

    return IMG_A, IMG_B

def MINI_BATCH(DATA_AB, BATCH_SIZE, NC_IN, NC_OUT):
    LENGTH = len(DATA_AB)
    EPOCH = I = 0
    TMP_SIZE = None
    while True:
        SIZE = TMP_SIZE if TMP_SIZE else BATCH_SIZE
        if I + SIZE > LENGTH:
            shuffle(DATA_AB)
            I = 0
            EPOCH += 1
        DATA_A = []
        DATA_B = []
        for J in range(I, I + SIZE):
            IMG_A,IMG_B = READ_IMAGE(DATA_AB[J], NC_IN, NC_OUT)
            DATA_A.append(IMG_A)
            DATA_B.append(IMG_B)
        DATA_A = np.float32(DATA_A)
        DATA_B = np.float32(DATA_B)
        I += SIZE
        TMP_SIZE = yield EPOCH, DATA_A, DATA_B

#%%

LIST_INPUT = LOAD_DATA(IMAGE_PATH_INPUT)
LIST_OUTPUT = LOAD_DATA(IMAGE_PATH_OUTPUT)
assert len(LIST_INPUT) == len(LIST_OUTPUT)
LIST_TOTAL = list(zip(sorted(LIST_INPUT), sorted(LIST_OUTPUT)))
TRAIN_BATCH = MINI_BATCH(LIST_TOTAL, BATCH_SIZE, NC_IN, NC_OUT)

#%%

T0 = T1 = time.time()
GEN_ITERS = 0
ERR_L = 0
EPOCH = 0
ERR_G = 0
ERR_L_SUM = 0
ERR_G_SUM = 0
ERR_D_SUM = 0

while GEN_ITERS <= NITERS :
    EPOCH, TRAIN_A, TRAIN_B = next(TRAIN_BATCH)
    TRAIN_A = TRAIN_A.reshape((BATCH_SIZE, ISIZE, ISIZE, NC_IN))
    TRAIN_B = TRAIN_B.reshape((BATCH_SIZE, ISIZE, ISIZE, NC_OUT))

    ERR_D,  = NET_D_TRAIN([TRAIN_A, TRAIN_B])
    ERR_D_SUM += ERR_D

    ERR_G, ERR_L = NET_G_TRAIN([TRAIN_A, TRAIN_B])
    ERR_G_SUM += ERR_G
    ERR_L_SUM += ERR_L

    GEN_ITERS += 1

    if GEN_ITERS%DISPLAY_ITERS==0:
        print('[%d][%d/%d] LOSS_D: %5.3f LOSS_G: %5.3f LOSS_L: %5.3f T: %dsec/%dits, Total T: %d'
        % (EPOCH, GEN_ITERS, NITERS, ERR_D_SUM/DISPLAY_ITERS, ERR_G_SUM/DISPLAY_ITERS, ERR_L_SUM/DISPLAY_ITERS, time.time()-T1, DISPLAY_ITERS, time.time()-T0))

        ERR_L_SUM = 0
        ERR_G_SUM = 0
        ERR_D_SUM = 0
        DST_MODEL = MODEL_PATH+MODE+'_ITER'+'%07d'%GEN_ITERS+'.h5'
        NET_G.save(DST_MODEL)
        T1 = time.time()
