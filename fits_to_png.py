from PIL import Image
from astropy.io import fits
import numpy as np
import os

input = 'AIA'
output = 'HMI'
w = h = 1024  # desired width and height of png
m_max = 100  # maximum value for magnetograms
m_min = -100  # minimum value for magnetograms
a_min = 0
a_max = 150/4


def save_to_png(name, fits_path, png_path, min, max, w, h,
                normalise=False, rotate=False, abs=False,
                crop=False):
    print(name)
    hdul = fits.open(fits_path + name + ".fits", memmap=True, ext=0)
    hdul.verify("fix")
    image_data = hdul[1].data
    if abs:
        image_data = np.abs(image_data)
        min = np.max([0, min])
    if normalise:
        int_time = hdul[1].header['DATAMEDN']
        image_data = image_data/int_time

    # clip data between (min, max):
    image_data = np.clip(image_data, min, max)
    # translate data so it's between (0, max-min):
    image_data -= min
    # normalise data
    image_data = image_data/(max - min)

    # format data, and convert to image
    image = Image.fromarray(np.uint8(image_data * 255), 'L')
    # crop to diameter of sun
    if crop:
        if "R_SUN" in hdul[1].header:
            radius = hdul[1].header["R_SUN"]
            # coordinates of center of sun
            x = 2096 - 40
            y = 2096 - 50
            # cropping coords:
            left = x - radius
            right = x + radius
            top = y - radius
            bottom = y + radius

        else:
            image_values = np.nan_to_num(image_data)
            row = 0  # row
            row_sum = np.sum(image_values[row])
            while row_sum == 0:  # itterate through until we find the sun
                row += 1
                row_sum = np.sum(image_values[row])
            top = row
            while row_sum != 0:  # itterate through the sun
                row += 1
                row_sum = np.sum(image_values[row])
            bottom = row

            col = 0  # column
            col_sum = np.sum(image_values[:, col])
            while col_sum == 0:  # itterate through until we find the sun
                col += 1
                col_sum = np.sum(image_values[:, col])
            left = col
            while col_sum != 0:  # itterate through the sun
                col += 1
                col_sum = np.sum(image_values[:, col])
            right = col

        image = image.crop((left, top, right, bottom))

    image = image.resize((w, h), Image.LANCZOS)
    # flip image to match original orientation.
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    # rotate images to match
    if rotate:
        image = image.transpose(Image.ROTATE_180)

    image.save(png_path + name + ".png")


def main(data, min, max, w, h, normalise=False, rotate=False,
         abs=False, crop=False):
    fits_path = "FITS_DATA/" + data + '/'
    if abs:
        data = 'ABS_' + data
    if crop:
        data = 'CROPPED_' + data
    for filename in os.listdir(fits_path):
        file_info = filename.split('.')
        date = file_info[2].replace('-', '')
        month = date[4:6]
        if month == '09' or month == '10':
            png_path = "DATA/TEST/" + data + '/'
        else:
            png_path = "DATA/TRAIN/" + data + '/'
        os.makedirs(png_path) if not os.path.exists(png_path) else None
        save_to_png(name=filename[:-5],
                    fits_path=fits_path,
                    png_path=png_path,
                    min=min,
                    max=max,
                    w=w,
                    h=h,
                    normalise=normalise,
                    rotate=rotate,
                    abs=abs,
                    crop=crop
                    )


if __name__ == "__main__":
    # aia:
    # main(data=input,
    #     min=a_min,
    #     max=a_max,
    #     w=w,
    #     h=h,
    #     normalise=True
    #     )
    # hmi:
    # main(data=output,
    #     min=m_min,
    #     max=m_max,
    #     w=w,
    #     h=h,
    #     rotate=True,
    #     )
    # abs hmi:
    # main(data=output,
    #      min=m_min,
    #      max=2000,
    #      w=w,
    #      h=h,
    #      rotate=True,
    #      abs=True
    #      )
    # cropped AIA
    main(data=input,
         min=a_min,
         max=a_max,
         w=w,
         h=h,
         normalise=True,
         crop=True
         )
    # cropped HMI
    main(data=output,
         min=m_min,
         max=m_max,
         w=w,
         h=h,
         rotate=True,
         crop=True
         )
