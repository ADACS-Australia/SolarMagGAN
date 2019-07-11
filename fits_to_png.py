from astropy.io import fits
import os
import matplotlib.pyplot as plt
import numpy as np
# from PIL import Image

m_max = 1500  # maximum value for magnetograms
m_min = -1500  # minimum value for magnetograms
a_min = 0
a_max = 150


def save_to_png(filename, min, max):
    hdul = fits.open("../data/ar1/" + filename, ext=0)
    hdul.verify("fix")
    image_data = hdul[1].data
    # clip data between (min, max):
    image_data = np.clip(image_data, min, max)
    # translate data so it's between (0, max-min):
    image_data -= min
    # normalise data so it's between (0, 1):
    image_data = image_data/(max - min)
    # set all nan values to 0
    image_data = np.nan_to_num(image_data)
    plt.imsave(
               fname="../png_data/ar1/" + filename[:-5] + ".png",
               arr=image_data,
               cmap='gray',
               origin='lower'
               )
    # image = Image.fromarray(image_data)
    # image = image.convert("L")
    # breakpoint()
    # image.save("../png_data/ar1/" + filename[:-5] + ".png")


for filename in os.listdir("../data/ar1/"):
    filename_info = filename.split(".")
    if filename_info[4] == "image_lev1":
        save_to_png(filename, a_min, a_max)

    elif filename_info[4] == "magnetogram":
        save_to_png(filename, m_min, m_max)
