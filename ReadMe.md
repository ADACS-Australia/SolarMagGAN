## Pipeline:

1. run `get_data.py` to download AIA and HMI data in the .fits file format to the folder `FITS_DATA`. This also creates a pickle files with the data queries. **OR**
1. run `get_pickle_data.py` if there already exists a pickle file with the data queries to download the AIA and HMI data in the .fits file format to the folder `FITS_DATA`.
2. run `fits_to_png.py` to create normalised and centered AIA and HMI png files from the fits files, and put them into testing and training sets. This is done in the `DATA` folder.
3. run `train.py` to train the GAN on the data, saving the models in the folder `MODELS`.
4. run `test.py` to test the models on new data, and generate figures for the results. The output png files are saved in the `RESULTS` folder, while the generated figures are saved in the `FIGURES` folder.

### Environments
To run `train.py` and `test.py` with the correct packages we made use of a conda environment.

To create a similar conda environment for using a gpu use the command:

`conda create --name gpu_env python=3.6 tensorflow-gpu==1.12 keras-gpu imageio matplotlib`

Alternatively, for a normal cpu, use the command:

`conda create --name cpu_env python=3.6 tensorflow==1.12 Keras imageio matplotlib`

## Network architectures

### Generator architectures

The generator is consist of the encoder-decoder architecture:

encoder:

1. Conv2D(filers = 64, strides = 2), LeakyReLu(slope = 0.2)
2. Conv2D(filers = 128, strides = 2), BatchNorm, LeakyReLu(slope = 0.2)
3. Conv2D(filers = 256, strides = 2), BatchNorm, LeakyReLu(slope = 0.2)
4. Conv2D(filers = 512, strides = 2), BatchNorm, LeakyReLu(slope = 0.2)
5. Conv2D(filers = 512, strides = 2), BatchNorm, LeakyReLu(slope = 0.2)
6. Conv2D(filers = 512, strides = 2), BatchNorm, LeakyReLu(slope = 0.2)
7. Conv2D(filers = 512, strides = 2), BatchNorm, LeakyReLu(slope = 0.2)
8. Conv2D(filers = 512, strides = 2), BatchNorm, LeakyReLu(slope = 0.2)
9. Conv2D(filers = 512, strides = 2), BatchNorm, LeakyReLu(slope = 0.2)
10. Conv2D(filers = 512, strides = 2), ReLu

decoder:

1. Conv2DTranspose(filter = 512, strides = 2), BatchNorm, Dropout(rate = 0.5), ReLU
2. Conv2DTranspose(filter = 512, strides = 2), BatchNorm, Dropout(rate = 0.5), ReLU
3. Conv2DTranspose(filter = 512, strides = 2), BatchNorm, Dropout(rate = 0.5), ReLU
4. Conv2DTranspose(filter = 512, strides = 2), BatchNorm, ReLU
5. Conv2DTranspose(filter = 512, strides = 2), BatchNorm, ReLU
6. Conv2DTranspose(filter = 512, strides = 2), BatchNorm, ReLU
7. Conv2DTranspose(filter = 256, strides = 2), BatchNorm, ReLU
8. Conv2DTranspose(filter = 128, strides = 2), BatchNorm, ReLU
9. Conv2DTranspose(filter = 64, strides = 2), BatchNorm, ReLU
10. Conv2DTranspose(filter = 1, strides = 2), Tanh

Also, the generator has skip-connections between layers of the encoder and layers of the decoder like the U-Net architecture. 

skip-connection:

* encoder 1st layer - decoder 9th layer
* encoder 2nd layer - decoder 8th layer
* encoder 3rd layer - decoder 7th layer
* encoder 4th layer - decoder 6th layer
* encoder 5th layer - decoder 5th layer
* encoder 6th layer - decoder 4th layer
* encoder 7th layer - decoder 3rd layer
* encoder 8th layer - decoder 2nd layer
* encoder 9th layer - decoder 1st layer

### Discriminator architectures

The discriminator architecture is described in the following notation:
* Conv2D(filers = 64, strides = 2), LeakyReLu(slope = 0.2)
* Conv2D(filers = 128, strides = 2), BatchNorm, LeakyReLu(slope = 0.2)
* Conv2D(filers = 256, strides = 2), BatchNorm, LeakyReLu(slope = 0.2)
* Conv2D(filers = 512, strides = 1), BatchNorm, LeakyReLu(slope = 0.2)
* Conv2D(filers = 1, strides = 1), Sigmoid

The receptive field size used in our discriminator is 70 x 70.

## Hyperparameter

#### The loss configuration of the objective functions of the Generator
* Total loss = loss of J2 + 100 * loss of J1

#### Batch
* Batch iteration : 500,000
* Batch size : 1

#### Optimizer 
* Optimizer : Adam solver
* Learning rate : 0.0002
* momentum beta 1 parameter : 0.5
* momentum beta 2 parameter : 0.999

#### Initializer
* The Initializer of the Convolution Layers : normal distribution, mean : 0.0, stddev : 0.02
* The gamma initializer of the BatchNormalization layers : normal distribution, mean : 1.0, stddev : 0.02
