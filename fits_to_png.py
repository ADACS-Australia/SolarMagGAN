from PIL import Image
from astropy.io import fits
import numpy as np
import os
import sunpy
import sunpy.map
import astropy.units as u
from astropy.coordinates import SkyCoord

input = 'AIA'
output = 'HMI'
w = h = 1024  # desired width and height of png
m_max = 100  # maximum value for magnetograms
m_min = -100  # minimum value for magnetograms
a_min = 0
a_max = 150/4


def save_to_png(name, fits_path, png_path, min, max, w, h,
                normalise=False, rotate=False, abs=False,
                crop=False, top_right=None, bottom_left=None):
    print(name)
    filename = fits_path + name + ".fits"
    hdul = fits.open(filename, memmap=True, ext=0)
    hdul.verify("fix")
    if not crop:
        image_data = hdul[1].data
    else:
        # Cropping to desired range
        map = sunpy.map.Map(filename)
        if rotate:
            map = map.rotate(angle=180 * u.deg)

        # Cropping
        image_data = map.submap(bottom_left, top_right).data

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
    image = image.resize((w, h), Image.LANCZOS)
    # flip image to match original orientation.
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    # rotate images to match
    if rotate and not crop:
        image = image.transpose(Image.ROTATE_180)

    image.save(png_path + name + ".png")


def main(data, min, max, w, h, normalise=False, rotate=False,
         abs=False, crop=False, top_right=None, bottom_left=None):
    fits_path = "FITS_DATA/" + data + '/'
    if abs:
        data = 'ABS_' + data
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
                    crop=crop,
                    top_right=top_right,
                    bottom_left=bottom_left
                    )


if __name__ == "__main__":
    # -1000 to 1000 arcsec results in a full disk image.
    # Slightly lower values will be closerto the actual limb of the Sun
    filename = "./FITS_DATA/HMI/" + os.listdir("FITS_DATA/HMI/")[0]
    map_ref = sunpy.map.Map(filename)
    top_right = SkyCoord(1000 * u.arcsec, 1000 * u.arcsec,
                         frame=map_ref.coordinate_frame)
    bottom_left = SkyCoord(-1000 * u.arcsec, -1000 * u.arcsec,
                           frame=map_ref.coordinate_frame)
    # AIA:
    main(data=input,
         min=a_min,
         max=a_max,
         w=w,
         h=h,
         normalise=True,
         crop=True,
         top_right=top_right,
         bottom_left=bottom_left
         )
    # HMI:
    main(data=output,
         min=m_min,
         max=m_max,
         w=w,
         h=h,
         rotate=True,
         crop=True,
         top_right=top_right,
         bottom_left=bottom_left
         )
