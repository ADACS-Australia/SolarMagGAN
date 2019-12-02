#!/usr/bin/env python3
import os
import sunpy
import sunpy.map
from astropy.io import fits
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from PIL import Image


def save_to_png(name, fits_path, png_path, min, max, w, h, top_right=None,
                bottom_left=None):

    print(name)
    filename = fits_path + name + ".fits"
    filename = fits_path + name + ".fits"
    hdul = fits.open(filename, memmap=False, ext=0)
    hdul.verify("fix")
    image_data = hdul[0].data

    # find median
    med = np.median(image_data)

    # Cropping to desired range
    map = sunpy.map.Map(filename)
    image_data = map.submap(bottom_left, top_right).data

    # translate data so it starts at 0
    image_data -= min

    # normalise to keep consistant
    image_data = image_data/(med-min)

    # clip data between (0, (max - min)):
    image_data = np.clip(image_data, 0, (max - min))

    # normalise data between 0 and 1
    image_data = image_data/((max - min))

    # format data, and convert to image
    image = Image.fromarray(np.uint8(image_data * 255), 'L')
    # crop to diameter of sun
    image = image.resize((w, h), Image.LANCZOS)
    # flip image to match original orientation.
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    # rotate images to match

    image.save(png_path + "STEREO.euvi304." + name + ".png")


data = "STEREO"
w = h = 1024
# min and maxed were chosen so that the result was as similar to
# aia between a min of 0 and a max of 150
min = 700
max = 900
fits_path = "FITS_DATA/STEREO/"
png_path = "DATA/TEST/STEREO/"
os.makedirs(png_path) if not os.path.exists(png_path) else None
filename = "./" + fits_path + os.listdir(fits_path)[0]
map_ref = sunpy.map.Map(filename)
top_right = SkyCoord(920 * u.arcsec, 920 * u.arcsec,
                     frame=map_ref.coordinate_frame)
bottom_left = SkyCoord(-920 * u.arcsec, -920 * u.arcsec,
                       frame=map_ref.coordinate_frame)

for filename in os.listdir(fits_path):
    save_to_png(name=filename[:-5],
                fits_path=fits_path,
                png_path=png_path,
                min=min,
                max=max,
                w=w,
                h=h,
                top_right=top_right,
                bottom_left=bottom_left,
                )
