import numpy as np
import os
import glob
import time
from random import shuffle
from imageio import imread
import pandas as pd
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
import argparse

print(1)
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

# parse the optional arguments:
parser = argparse.ArgumentParser()
parser.add_argument("--model_name",
                    help="name of model",
                    default='trial_2'
                    )
parser.add_argument("--input",
                    help="folder name of input data",
                    default='AIA'
                    )
parser.add_argument("--display_iter",
                    help="number of iterations between each test",
                    type=int,
                    default=20000
                    )
parser.add_argument("--max_iter",
                    help="total number of iterations",
                    type=int,
                    default=500000
                    )
args = parser.parse_args()

# Hyper parameters
NITERS = args.max_iter  # total number of iterations
# number of iterations before display and model creation
DISPLAY_ITERS = args.display_iter

# the input data:
# (originally AIA or Atmospheric Imaging Assembly)
INPUT_DATA = args.input
# The data we want to reproduce:
# (originally HMI or Helioseismic and Magnetic Imager)
OUTPUT_DATA = 'HMI'

ISIZE = 1024  # height of the image
NC_IN = 1  # number of input channels (1 for greyscale, 3 for RGB)
NC_OUT = 1  # number of output channels (1 for greyscale, 3 for RGB)
BATCH_SIZE = 1  # number of images in each batch
# max layers in the discriminator not including sigmoid activation:
# 1 for 16, 2 for 34, 3 for 70, 4 for 142, and 5 for 286 (receptive field size)
MAX_LAYERS = 3

TRIAL_NAME = args.model_name


MODE = INPUT_DATA + '_to_' + OUTPUT_DATA  # folder name for saving the model

IMAGE_PATH_INPUT = './DATA/TRAIN/'+INPUT_DATA+'/*.png'  # input file path
IMAGE_PATH_OUTPUT = './DATA/TRAIN/'+OUTPUT_DATA+'/*.png'  # ouptut file path

# make a folder for the trial if it doesn't already exist
MODEL_PATH_MAIN = './MODELS/' + TRIAL_NAME + '/'
# os.mkdir(MODEL_PATH_MAIN) if not os.path.exists(MODEL_PATH_MAIN) else None
MODEL_PATH = MODEL_PATH_MAIN + MODE + '/'
os.makedirs(MODEL_PATH) if not os.path.exists(MODEL_PATH) else None


# generates tensors with a normal distribution with (mean, standard deviation)
# this is used as a matrix of weights
CONV_INIT = RandomNormal(0, 0.02)
GAMMA_INIT = RandomNormal(1., 0.02)

print(2)

# The loss function
def LOSS_FN(OUTPUT, TARGET):
    return -K.mean(K.log(OUTPUT+1e-12)*TARGET+K.log(1-OUTPUT+1e-12)*(1-TARGET))


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
    # combines the inputs from the generator and the desired input
    INPUT_A, INPUT_B = Input(shape=(ISIZE, ISIZE, NC_IN)),\
        Input(shape=(ISIZE, ISIZE, NC_OUT))

    INPUT = Concatenate(axis=CH_AXIS)([INPUT_A, INPUT_B])

    if MAX_LAYERS == 0:
        N_FEATURE = 1  # number of filters to use
        # apply sigmoid activation
        L = DN_CONV(N_FEATURE,
                    kernel_size=1,
                    padding='same',
                    activation='sigmoid'
                    )(INPUT)

    else:
        N_FEATURE = 64  # number of filters to use
        # apply convolution
        L = DN_CONV(N_FEATURE,
                    kernel_size=4,
                    strides=2,
                    padding="same"
                    )(INPUT)
        # Apply leaky ReLU activation with a slope of 0.2
        L = LEAKY_RELU(0.2)(L)

        # Apply convolution MAX_LAYERS times
        for i in range(1, MAX_LAYERS):
            N_FEATURE *= 2  # double the number of filters
            # Apply convolution
            L = DN_CONV(N_FEATURE,
                        kernel_size=4,
                        strides=2,
                        padding="same"
                        )(L)
            # normalise
            L = BATNORM()(L, training=1)
            # Apply leaky ReLU activation with a slope of 0.2
            L = LEAKY_RELU(0.2)(L)

        N_FEATURE *= 2  # double the number of filters
        L = ZeroPadding2D(1)(L)  # pads the model with 0s with a thickness of 1
        # Apply convolution
        L = DN_CONV(N_FEATURE, kernel_size=4, padding="valid")(L)
        # normalise
        L = BATNORM()(L, training=1)
        # Apply leaky ReLU activation with a slope of 0.2
        L = LEAKY_RELU(0.2)(L)

        N_FEATURE = 1
        L = ZeroPadding2D(1)(L)  # pads the model with 0s with a thickness of 1
        # Apply sigmoid activation
        L = DN_CONV(N_FEATURE,
                    kernel_size=4,
                    padding="valid",
                    activation='sigmoid'
                    )(L)

    return Model(inputs=[INPUT_A, INPUT_B], outputs=L)


# The generator (based on the U-Net architecture)
def UNET_G(ISIZE, NC_IN, NC_OUT, FIXED_INPUT_SIZE=True):
    MAX_N_FEATURE = 64 * 8  # max number of filters to use

    def BLOCK(X, S, NF_IN, USE_BATNORM=True, NF_OUT=None, NF_NEXT=None):
        # Encoder: (decreasing size)

        assert S >= 2 and S % 2 == 0
        if NF_NEXT is None:  # number of filters in the next layer?
            # set number of filters to twice the number of filters in the
            # input, if it isn't more than the max number of filters
            NF_NEXT = min(NF_IN*2, MAX_N_FEATURE)
        if NF_OUT is None:
            NF_OUT = NF_IN
        # Apply convolution
        X = DN_CONV(NF_NEXT,
                    kernel_size=4,
                    strides=2,
                    # don't use a bias if batch normalisation will be done
                    # later, or if s > 2
                    use_bias=(not (USE_BATNORM and S > 2)),
                    padding="same"
                    )(X)
        if S > 2:
            # apply batch normalisation
            if USE_BATNORM:
                X = BATNORM()(X, training=1)
            # apply leaky ReLU with a slope of 0,2
            X2 = LEAKY_RELU(0.2)(X)
            # continue recursion until size = 2, halving size each time
            X2 = BLOCK(X2, S//2, NF_NEXT)
            # combine X and X2
            # this gives the "skip connections" between the encoder layers
            # and decoder layers.
            X = Concatenate(axis=CH_AXIS)([X, X2])

        # Decoder: (Increasing size)
        # This happens only when the recursive encoder has reached its maximum
        # depth (size = 2)
        # Note the minimum layer size is actually s = 4, as encoding stops when
        # s = 2

        # Apply ReLU activation
        X = Activation("relu")(X)

        # Apply deconvolution
        X = UP_CONV(NF_OUT,
                    kernel_size=4,
                    strides=2,
                    use_bias=not USE_BATNORM
                    )(X)
        X = Cropping2D(1)(X)
        # Batch normalisation
        if USE_BATNORM:
            X = BATNORM()(X, training=1)
        # apply dropout
        # Randomly drops units which helps prevent overfitting
        if S <= 8:
            X = Dropout(0.5)(X, training=1)
        return X

    S = ISIZE if FIXED_INPUT_SIZE else None  # size
    X = INPUT = Input(shape=(S, S, NC_IN))  # The input
    # Apply the U-Net convolution, deconvolution (see above function)
    X = BLOCK(X, ISIZE, NC_IN, False, NF_OUT=NC_OUT, NF_NEXT=64)
    # Apply tanh activation
    X = Activation('tanh')(X)

    return Model(inputs=INPUT, outputs=[X])

print(3)
# The discriminator model
NET_D = BASIC_D(ISIZE, NC_IN, NC_OUT, MAX_LAYERS)
# The generator model
NET_G = UNET_G(ISIZE, NC_IN, NC_OUT)

# tensor placeholders?
REAL_A = NET_G.input  # generator input (AIA)
FAKE_B = NET_G.output  # generator output (fake HMI)
REAL_B = NET_D.inputs[1]  # descriminator input (real HMI)

# output of the discriminator for AIA and real HMI
OUTPUT_D_REAL = NET_D([REAL_A, REAL_B])
# output of the discriminator for AIA and fake HMI
OUTPUT_D_FAKE = NET_D([REAL_A, FAKE_B])

# set initial values for the loss
# ones_like creates a tensor of the same shape full of ones
# zeros_like creates a tensor of the same shape full of zeros
# as the discriminator gives the probability that the input is a real HMI
# picture, we want it to out put 1 when the input is real and 0 when the
# input is fake.
LOSS_D_REAL = LOSS_FN(OUTPUT_D_REAL, K.ones_like(OUTPUT_D_REAL))
LOSS_D_FAKE = LOSS_FN(OUTPUT_D_FAKE, K.zeros_like(OUTPUT_D_FAKE))
# while the generator, we want the discriminator to guess that the
# generator output is the real HMI, which corresponds to the discriminator
# outputting 1:
LOSS_G_FAKE = LOSS_FN(OUTPUT_D_FAKE, K.ones_like(OUTPUT_D_FAKE))

# total average difference between the real and generated HMIs
LOSS_L = K.mean(K.abs(FAKE_B-REAL_B))

# Total loss of the discriminator
LOSS_D = LOSS_D_REAL + LOSS_D_FAKE
# gives the updates for the discriminator training
TRAINING_UPDATES_D = Adam(lr=2e-4, beta_1=0.5
                          ).get_updates(NET_D.trainable_weights, [], LOSS_D)
# creates a function that trains the discriminator
NET_D_TRAIN = K.function([REAL_A, REAL_B], [LOSS_D/2.0], TRAINING_UPDATES_D)

# The total loss of G, which includes the difference between the real and
# generated HMIs, as well as the loss because of the descriminator
LOSS_G = LOSS_G_FAKE + 100 * LOSS_L

# operation to update the gradient of the generator using the adam optimizer
TRAINING_UPDATES_G = Adam(
                          lr=2e-4,
                          beta_1=0.5
                          ).get_updates(NET_G.trainable_weights, [], LOSS_G)
# function to train the generator
NET_G_TRAIN = K.function([REAL_A, REAL_B],
                         [LOSS_G_FAKE, LOSS_L],
                         TRAINING_UPDATES_G)

print(4)
# returns list of files that match FILE_PATTERN
def LOAD_DATA(FILE_PATTERN):
    return glob.glob(FILE_PATTERN)


def GET_DATE(file):
    filename = file.split("/")[-1]  # filename is at end of file path
    date_str = filename.split(".")[2]  # date string is after second "."
    date_str = date_str.replace("_", "")  # remove underscores
    date_str = date_str.replace("-", "")  # remove hyphens
    date_str = date_str.replace("TAI", "z")  # TAI and Z are both UTC
    date = pd.Timestamp(date_str)
    return date


def GET_TIMESTAMP(file):
    date = GET_DATE(file)
    return date.timestamp()


# FN = filenames, NC_IN = #channels in input, NC_OUT = #channels in output
# This function essentially reads the image, and shifts it slightly by up
# to 15 pixels any direction before returning it. This is probably to
# prevent overfitting
def READ_IMAGE(FN, NC_IN, NC_OUT):
    IMG_A = imread(FN[0])
    IMG_B = imread(FN[1])
    X, Y = np.random.randint(31), np.random.randint(31)
    if NC_IN != 1:
        IMG_A = np.pad(IMG_A, ((15, 15), (15, 15), (0, 0)), 'constant')
        IMG_A = IMG_A[X:X + 1024, Y:Y + 1024, :] / 255.0 * 2 - 1
    else:
        IMG_A = np.pad(IMG_A, 15, 'constant')
        IMG_A = IMG_A[X:X + 1024, Y:Y + 1024] / 255.0 * 2 - 1

    if NC_OUT != 1:
        IMG_B = np.pad(IMG_B, ((15, 15), (15, 15), (0, 0)), 'constant')
        IMG_B = IMG_B[X:X + 1024, Y:Y + 1024, :] / 255.0 * 2 - 1
    else:
        IMG_B = np.pad(IMG_B, 15, 'constant')
        IMG_B = IMG_B[X:X + 1024, Y:Y + 1024] / 255.0 * 2 - 1

    return IMG_A, IMG_B


# create mini batches for training (actually creates a generator
# that generates each element of the batch)
def MINI_BATCH(DATA_AB, BATCH_SIZE, NC_IN, NC_OUT):
    LENGTH = len(DATA_AB)
    EPOCH = i = 0
    TMP_SIZE = None
    while True:
        SIZE = TMP_SIZE if TMP_SIZE else BATCH_SIZE
        # if we reach the end of the data (which corresponds to an
        # epoch), shuffle data and begin again
        if i + SIZE > LENGTH:
            shuffle(DATA_AB)
            i = 0
            EPOCH += 1
        DATA_A = []
        DATA_B = []
        # make batches of length: SIZE
        for J in range(i, i + SIZE):
            IMG_A, IMG_B = READ_IMAGE(DATA_AB[J], NC_IN, NC_OUT)
            DATA_A.append(IMG_A)
            DATA_B.append(IMG_B)
        DATA_A = np.float32(DATA_A)
        DATA_B = np.float32(DATA_B)
        i += SIZE
        TMP_SIZE = yield EPOCH, DATA_A, DATA_B


# input data
LIST_INPUT = LOAD_DATA(IMAGE_PATH_INPUT)
# output data
LIST_OUTPUT = LOAD_DATA(IMAGE_PATH_OUTPUT)

# sort lists based on timestamp
LIST_OUTPUT = sorted(LIST_OUTPUT, key=GET_TIMESTAMP)
LIST_INPUT = sorted(LIST_INPUT, key=GET_TIMESTAMP)


i = 0  # index of LIST_INPUT
j = 0  # index of LIST_OUTPUT

print(5)
# only keep images that are in both input and output
while i < len(LIST_INPUT) and j < len(LIST_OUTPUT):
    input = LIST_INPUT[i]
    in_time = GET_DATE(input)
    output = LIST_OUTPUT[j]
    out_time = GET_DATE(output)
    # if input is after output, delete output:
    if in_time.date() > out_time.date():
        del(LIST_OUTPUT[j])
    # if input is before output, delete input:
    elif in_time.date() < out_time.date():
        del(LIST_INPUT[i])
    # if input is after output, delete output:
    elif in_time.hour > out_time.hour:
        del(LIST_OUTPUT[j])
    # if input is before output, delete input:
    elif in_time.hour < out_time.hour:
        del(LIST_INPUT[i])
    # else, date and hours are the same, so we have a pair!
    else:
        # increment both lists
        i += 1
        j += 1

# trim ends of lists so they are the same size
length = min(i, j)
LIST_INPUT = LIST_INPUT[:length]
LIST_OUTPUT = LIST_OUTPUT[:length]

assert len(LIST_INPUT) == len(LIST_OUTPUT)

# zips the data such that each element is a (input, output) pair
LIST_TOTAL = list(zip(sorted(LIST_INPUT), sorted(LIST_OUTPUT)))

print("Input Output Pairs:")
print(LIST_TOTAL)
# creates a generator to use for training
TRAIN_BATCH = MINI_BATCH(LIST_TOTAL, BATCH_SIZE, NC_IN, NC_OUT)

# initialise training variables
T0 = T1 = time.time()
GEN_ITERS = 0
ERR_L = 0
EPOCH = 0
ERR_G = 0
ERR_L_SUM = 0
ERR_G_SUM = 0
ERR_D_SUM = 0

# training:
while GEN_ITERS <= NITERS:
    EPOCH, TRAIN_A, TRAIN_B = next(TRAIN_BATCH)
    # input data set
    TRAIN_A = TRAIN_A.reshape((BATCH_SIZE, ISIZE, ISIZE, NC_IN))
    # output data set
    TRAIN_B = TRAIN_B.reshape((BATCH_SIZE, ISIZE, ISIZE, NC_OUT))

    # descriminator training and error
    ERR_D,  = NET_D_TRAIN([TRAIN_A, TRAIN_B])
    ERR_D_SUM += ERR_D

    # generator training and error
    ERR_G, ERR_L = NET_G_TRAIN([TRAIN_A, TRAIN_B])
    ERR_G_SUM += ERR_G
    ERR_L_SUM += ERR_L

    GEN_ITERS += 1

    # print training summary and save model
    if GEN_ITERS % DISPLAY_ITERS == 0:
        print('[%d][%d/%d] LOSS_D: %5.3f LOSS_G: %5.3f LOSS_L: %5.3f T:'
              '%dsec/%dits, Total T: %d'
              % (
                 EPOCH, GEN_ITERS, NITERS, ERR_D_SUM/DISPLAY_ITERS,
                 ERR_G_SUM/DISPLAY_ITERS, ERR_L_SUM/DISPLAY_ITERS,
                 time.time()-T1, DISPLAY_ITERS, time.time()-T0
                 )
              )

        ERR_L_SUM = 0
        ERR_G_SUM = 0
        ERR_D_SUM = 0
        DST_MODEL = MODEL_PATH+MODE+'_ITER'+'%07d' % GEN_ITERS+'.h5'
        NET_G.save(DST_MODEL)
        T1 = time.time()
