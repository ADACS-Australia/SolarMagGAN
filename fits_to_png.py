from PIL import Image
from astropy.io import fits
import numpy as np
import os

input = 'AIA'
output = 'HMI'
w = h = 1024  # desired width and height of png
m_max = 1500  # maximum value for magnetograms
m_min = -1500  # minimum value for magnetograms
a_min = -10
a_max = 150


def resize_and_save_to_png(name, fits_path, png_path, min, max, w, h):
    print(name)
    hdul = fits.open(fits_path + name + ".fits", memmap=True, ext=0)
    hdul.verify("fix")
    image_data = hdul[1].data
    # clip data between (min, max):
    image_data = np.clip(image_data, min, max)
    # translate data so it's between (0, max-min):
    image_data -= min
    # normalise data
    image_data = image_data/(max - min)

    # format data, and convert to image
    image = Image.fromarray(np.uint8(image_data * 255), 'L')
    image = image.resize((w, h), Image.LANCZOS)
    # flip image to match original orientation.
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    image.save(png_path + name + ".png")


def main(data, min, max, w, h):
    fits_path = "FITS_DATA/" + data + '/'
    for filename in os.listdir(fits_path):
        file_info = filename.split('.')
        date = file_info[2].replace('-', '')
        month = date[4:6]
        if month == '09' or month == '10':
            png_path = "DATA/TEST/" + data + '/'
        else:
            png_path = "DATA/TRAIN/" + data + '/'
        os.makedirs(png_path) if not os.path.exists(png_path) else None
        resize_and_save_to_png(name=filename[:-5],
                               fits_path=fits_path,
                               png_path=png_path,
                               min=min,
                               max=max,
                               w=w,
                               h=h
                               )


main(data=input,
     min=a_min,
     max=a_max,
     w=w,
     h=h
     )
main(data=output,
     min=m_min,
     max=m_max,
     w=w,
     h=h
     )
