import os
import glob
from imageio import imread, imsave
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import keras.backend as K
import time
import tensorflow as tf

# initialise tensorflow
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.Session(config=config)

# initialise KERAS_BACKEND
K.set_image_data_format('channels_last')
channel_axis = -1

# initialise os environment
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def listdir_nohidden(path):
    return glob.glob(os.path.join(path, '*'))


SLEEP_TIME = 1000
DISPLAY_ITER = 2
MAX_ITER = 6

MODE = 'TEST_INPUT_to_TEST_OUTPUT'
TRIAL_NAME = 'TEST3'

INPUT = 'TEST_INPUT'  # input used while training
# testing input which has a corresponding output (near side AIA)
INPUT1 = 'TEST_INPUT1'
# testing input which does not have a corresponding (far side EUV)
INPUT2 = 'TEST_INPUT2'
# corresponding output for INPUT1 (near side HMI)
OUTPUT = 'TEST_OUTPUT'

ISIZE = 1024  # input size
NC_IN = 1  # number of channels in the output
NC_OUT = 1  # number of channels in the input
BATCH_SIZE = 1  # batch size

# used for calculating the total unsigned magnetic flux (TUMF)
RSUN = 392
SATURATION = 100
THRESHOLD = 10

OP1 = INPUT1 + '_to_' + OUTPUT
OP2 = INPUT2 + '_to_' + OUTPUT

IMAGE_PATH1 = './DATA/TEST/' + INPUT1 + '/*.png'  # INPUT1 data file path
IMAGE_PATH2 = './DATA/TEST/' + INPUT2 + '/*.png'  # INPUT2 data file path
IMAGE_PATH3 = './DATA/TEST/' + OUTPUT + '/*.png'  # OUTPUT data file path

# finds and sorts the filenames for INPUT1, INPUT2 and OUTPUT respectively
IMAGE_LIST1 = sorted(glob.glob(IMAGE_PATH1))
IMAGE_LIST2 = sorted(glob.glob(IMAGE_PATH2))
IMAGE_LIST3 = sorted(glob.glob(IMAGE_PATH3))

RESULT_PATH_MAIN = './RESULTS/' + TRIAL_NAME + '/'
# os.mkdir(RESULT_PATH_MAIN) if not os.path.exists(RESULT_PATH_MAIN) else None

# file path for the results of INPUT1 to OUTPUT (generating HMI from nearside)
RESULT_PATH1 = RESULT_PATH_MAIN + OP1 + '/'
os.makedirs(RESULT_PATH1) if not os.path.exists(RESULT_PATH1) else None

# file path for the results of INPUT2 to OUTPUT (generating HMI from farside)
RESULT_PATH2 = RESULT_PATH_MAIN + OP2 + '/'
os.makedirs(RESULT_PATH2) if not os.path.exists(RESULT_PATH2) else None

# file path for the figures
FIGURE_PATH_MAIN = './FIGURES/' + TRIAL_NAME + '/'
os.makedirs(FIGURE_PATH_MAIN) if not os.path.exists(FIGURE_PATH_MAIN) else None


# This is used for finding the TUMF value
def SCALE(DATA, RANGE_IN, RANGE_OUT):

    DOMAIN = [RANGE_IN[0], RANGE_OUT[1]]

    def INTERP(X):
        return RANGE_OUT[0] * (1.0 - X) + RANGE_OUT[1] * X

    def UNINTERP(X):
        B = 0
        if (DOMAIN[1] - DOMAIN[0]) != 0:
            B = DOMAIN[1] - DOMAIN[0]
        else:
            B = 1.0 / DOMAIN[1]
        return (X - DOMAIN[0]) / B

    return INTERP(UNINTERP(DATA))


# finds the TUMF value
def TUMF_VALUE(IMAGE, RSUN, SATURATION, THRESHOLD):
    VALUE_POSITIVE = 0
    VALUE_NEGATIVE = 0

    IMAGE_SCALE = SCALE(
                        IMAGE,
                        RANGE_IN=[0., 255.],
                        RANGE_OUT=[-SATURATION, SATURATION]
                        )

    SIZE_X, SIZE_Y = IMAGE_SCALE.shape[0], IMAGE_SCALE.shape[1]

    for I in range(SIZE_X):
        for J in range(SIZE_Y):
            if (I-SIZE_X/2) ** 2. + (J-SIZE_Y/2) ** 2. < RSUN ** 2.:
                if IMAGE_SCALE[I, J] > THRESHOLD:
                    VALUE_POSITIVE += IMAGE_SCALE[I, J]
                elif IMAGE_SCALE[I, J] < -THRESHOLD:
                    VALUE_NEGATIVE += IMAGE_SCALE[I, J]
                else:
                    None

    FACT = (695500./RSUN) * (695500./RSUN) * 1000 * 1000 * 100 * 100

    FLUX_POSITIVE = VALUE_POSITIVE * FACT
    FLUX_NEGATIVE = VALUE_NEGATIVE * FACT
    FLUX_TOTAL = FLUX_POSITIVE + abs(FLUX_NEGATIVE)

    return FLUX_POSITIVE, FLUX_NEGATIVE, FLUX_TOTAL


# during training, the model was saved every DISPLAY_ITER steps
# as such, test every DISPLAY_ITER.
ITER = DISPLAY_ITER
while ITER <= MAX_ITER:

    SITER = '%07d' % ITER  # string representing the itteration

    # file path for the model of current itteration
    MODEL_NAME = './MODELS/' + TRIAL_NAME + '/' + MODE + '/' + MODE + \
        '_ITER' + SITER + '.h5'

    # file path to save the generated outputs from INPUT1 (nearside)
    SAVE_PATH1 = RESULT_PATH1 + 'ITER' + SITER + '/'
    os.mkdir(SAVE_PATH1) if not os.path.exists(SAVE_PATH1) else None

    # file path to save the generated outputs from INPUT2 (farside)
    SAVE_PATH2 = RESULT_PATH2 + 'ITER' + SITER + '/'
    os.mkdir(SAVE_PATH2) if not os.path.exists(SAVE_PATH2) else None

    # file path to save the figures
    FIGURE_PATH = FIGURE_PATH_MAIN + 'ITER' + SITER

    EX = 0
    while EX < 1:
        if os.path.exists(MODEL_NAME):
            print('Starting Iter ' + str(ITER) + ' ...')
            EX = 1
        else:
            print('Waiting Iter ' + str(ITER) + ' ...')
            time.sleep(SLEEP_TIME)

    # load the model
    MODEL = load_model(MODEL_NAME)

    REAL_A = MODEL.input
    FAKE_B = MODEL.output
    # function that evaluates the model
    NET_G_GENERATE = K.function([REAL_A], [FAKE_B])

    # generates the output (HMI) based on input image (A)
    def NET_G_GEN(A):
        output = [NET_G_GENERATE([A[I:I+1]])[0] for I in range(A.shape[0])]
        return np.concatenate(output, axis=0)

    UTMF_REAL = []
    UTMF_FAKE = []

    for I in range(len(IMAGE_LIST1)):
        # input image (AIA nearside)
        IMG = np.float32(imread(IMAGE_LIST1[I]) / 255.0 * 2 - 1)
        # desired image (HMI nearside)
        REAL = np.float32(imread(IMAGE_LIST3[I]))
        INFO = IMAGE_LIST1[I].split('.')
        DATE = INFO[2].replace('-', '').replace('_', '').replace('T', '')[:10]
        # reshapes IMG tensor to (BATCH_SIZE, ISIZE, ISIZE, NC_IN)
        IMG.shape = (BATCH_SIZE, ISIZE, ISIZE, NC_IN)
        # output image (generated HMI)
        FAKE = NET_G_GEN(IMG)
        FAKE = ((FAKE[0] + 1) / 2.0 * 255.).clip(0, 255).astype('uint8')
        FAKE.shape = (ISIZE, ISIZE) if NC_IN == 1 else (ISIZE, ISIZE, NC_OUT)
        SAVE_NAME = SAVE_PATH1 + OP1 + '_' + DATE + '.png'
        imsave(SAVE_NAME, FAKE)

        RP, RN, RT = TUMF_VALUE(REAL, RSUN, SATURATION, THRESHOLD)
        FP, FN, FT = TUMF_VALUE(FAKE, RSUN, SATURATION, THRESHOLD)

        UTMF_REAL.append(RT)
        UTMF_FAKE.append(FT)

    for J in range(len(IMAGE_LIST2)):
        IMG = np.float32(imread(IMAGE_LIST2[J]) / 255.0 * 2 - 1)
        INFO = IMAGE_LIST2[I].split('.')
        DATE = INFO[2].replace('-', '').replace('_', '').replace('T', '')[:10]
        IMG.shape = (BATCH_SIZE, ISIZE, ISIZE, NC_IN)
        FAKE = NET_G_GEN(IMG)
        FAKE = ((FAKE[0] + 1) / 2.0 * 255.).clip(0, 255).astype('uint8')
        FAKE.shape = (ISIZE, ISIZE) if NC_IN == 1 else (ISIZE, ISIZE, NC_OUT)
        SAVE_NAME = SAVE_PATH2 + OP2 + '_' + DATE + '.png'
        imsave(SAVE_NAME, FAKE)

    def MAKE_FIGURE():
        INPUT1_IMAGE_LIST = listdir_nohidden('./DATA/TEST/'+INPUT1 + '/')
        I1 = np.array(imread(INPUT1_IMAGE_LIST[0]))
        I2 = np.array(imread(INPUT1_IMAGE_LIST[1]))
        I3 = np.array(imread(INPUT1_IMAGE_LIST[2]))
        I4 = np.array(imread(INPUT1_IMAGE_LIST[3]))

        OUTPUT_IMAGE_LIST = listdir_nohidden('./DATA/TEST/'+OUTPUT + '/')

        T1 = np.array(imread(OUTPUT_IMAGE_LIST[0]))
        T2 = np.array(imread(OUTPUT_IMAGE_LIST[1]))
        T3 = np.array(imread(OUTPUT_IMAGE_LIST[2]))
        T4 = np.array(imread(OUTPUT_IMAGE_LIST[3]))

        O1 = np.array(imread(SAVE_PATH1 + '/' + OP1 + '_0.png'))
        O2 = np.array(imread(SAVE_PATH1 + '/' + OP1 + '_1.png'))
        O3 = np.array(imread(SAVE_PATH1 + '/' + OP1 + '_2.png'))
        O4 = np.array(imread(SAVE_PATH1 + '/' + OP1 + '_3.png'))

        fig2 = plt.figure()

        ax211 = fig2.add_subplot(3, 4, 1)
        ax211.imshow(I1, cmap='gray')
        ax211.axis('off')

        ax212 = fig2.add_subplot(3, 4, 2)
        ax212.imshow(I2, cmap='gray')
        ax212.axis('off')

        ax213 = fig2.add_subplot(3, 4, 3)
        ax213.imshow(I3, cmap='gray')
        ax213.axis('off')

        ax214 = fig2.add_subplot(3, 4, 4)
        ax214.imshow(I4, cmap='gray')
        ax214.axis('off')

        ax221 = fig2.add_subplot(3, 4, 5)
        ax221.imshow(O1, cmap='gray')
        ax221.axis('off')

        ax222 = fig2.add_subplot(3, 4, 6)
        ax222.imshow(O2, cmap='gray')
        ax222.axis('off')

        ax223 = fig2.add_subplot(3, 4, 7)
        ax223.imshow(O3, cmap='gray')
        ax223.axis('off')

        ax224 = fig2.add_subplot(3, 4, 8)
        ax224.imshow(O4, cmap='gray')
        ax224.axis('off')

        ax231 = fig2.add_subplot(3, 4, 9)
        ax231.imshow(T1, cmap='gray')
        ax231.axis('off')

        ax232 = fig2.add_subplot(3, 4, 10)
        ax232.imshow(T2, cmap='gray')
        ax232.axis('off')

        ax233 = fig2.add_subplot(3, 4, 11)
        ax233.imshow(T3, cmap='gray')
        ax233.axis('off')

        ax234 = fig2.add_subplot(3, 4, 12)
        ax234.imshow(T4, cmap='gray')
        ax234.axis('off')

        fig2.savefig(FIGURE_PATH + '_FIGURE2.png')
        plt.close(fig2)

        CC = np.corrcoef(UTMF_REAL, UTMF_FAKE)[0, 1]
        fig3 = plt.figure()
        fig3.suptitle('CC : %6.3f' % (CC))
        ax3 = fig3.add_subplot(1, 1, 1)
        ax3.plot(UTMF_REAL, UTMF_FAKE, 'ro')
        fig3.savefig(FIGURE_PATH + '_FIGURE3.png')
        plt.close(fig3)

        INPUT2_IMAGE_LIST = listdir_nohidden('./DATA/TEST/'+INPUT2 + '/')
        INPUT_TRAIN_IMAGE_LIST = listdir_nohidden('./DATA/TRAIN/'+INPUT + '/')

        U1 = np.array(imread(INPUT2_IMAGE_LIST[0]))
        U2 = np.array(imread(INPUT2_IMAGE_LIST[1]))
        U3 = np.array(imread(INPUT_TRAIN_IMAGE_LIST[0]))
        U4 = np.array(imread(INPUT_TRAIN_IMAGE_LIST[1]))

        OUTPUT_TRAIN_IMAGE_LIST = listdir_nohidden('./DATA/TRAIN/' +
                                                   OUTPUT + '/')

        D1 = np.array(imread(SAVE_PATH2 + '/' + OP2 + '_0.png'))
        D2 = np.array(imread(SAVE_PATH2 + '/' + OP2 + '_1.png'))
        D3 = np.array(imread(OUTPUT_TRAIN_IMAGE_LIST[0]))
        D4 = np.array(imread(OUTPUT_TRAIN_IMAGE_LIST[1]))

        fig4 = plt.figure()

        ax411 = fig4.add_subplot(2, 4, 1)
        ax411.imshow(U1, cmap='gray')
        ax411.axis('off')

        ax412 = fig4.add_subplot(2, 4, 2)
        ax412.imshow(U2, cmap='gray')
        ax412.axis('off')

        ax413 = fig4.add_subplot(2, 4, 3)
        ax413.imshow(U3, cmap='gray')
        ax413.axis('off')

        ax414 = fig4.add_subplot(2, 4, 4)
        ax414.imshow(U4, cmap='gray')
        ax414.axis('off')

        ax421 = fig4.add_subplot(2, 4, 5)
        ax421.imshow(D1, cmap='gray')
        ax421.axis('off')

        ax422 = fig4.add_subplot(2, 4, 6)
        ax422.imshow(D2, cmap='gray')
        ax422.axis('off')

        ax423 = fig4.add_subplot(2, 4, 7)
        ax423.imshow(D3, cmap='gray')
        ax423.axis('off')

        ax424 = fig4.add_subplot(2, 4, 8)
        ax424.imshow(D4, cmap='gray')
        ax424.axis('off')

        fig4.savefig(FIGURE_PATH + '_FIGURE4.png')
        plt.close(fig4)

    MAKE_FIGURE()

    del MODEL
    K.clear_session()

    ITER += DISPLAY_ITER
