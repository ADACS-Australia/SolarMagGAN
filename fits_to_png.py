from PIL import Image
from astropy.io import fits
import numpy as np
import os
import sunpy
import sunpy.map
import astropy.units as u
from astropy.coordinates import SkyCoord
import argparse
import random

# parse the optional arguments:
parser = argparse.ArgumentParser()
parser.add_argument(
    "--min", help="lower bound cutoff pixel value for AIA", type=int, default=0
)
parser.add_argument(
    "--max", help="upper bound cutoff pixel value for AIA", type=int, default=150
)

parser.add_argument(
    "--random",
    help="randomly assign upper bound pixel value for AIA",
    action="store_true",
)

parser.add_argument(
    "--name", help="name of folder for AIA to be saved in", default="AIA"
)

args = parser.parse_args()


input = "AIA"
output = "HMI"
w = h = 1024  # desired width and height of png
m_max = 100  # maximum value for magnetograms
m_min = -100  # minimum value for magnetograms
a_min = args.min
a_max = args.max
AIA = True
HMI = False


def save_to_png(
    name,
    fits_path,
    png_path,
    min,
    max,
    w,
    h,
    normalise=False,
    rotate=False,
    abs=False,
    crop=False,
    top_right=None,
    bottom_left=None,
):
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

    # clip data between (min, max):
    image_data = np.clip(image_data, min, max)

    if normalise:
        med = hdul[1].header["DATAMEDN"]
        # make sure median is between min and max:
        np.clip(med, min, max)
        image_data = image_data / med

    if abs:
        image_data = np.abs(image_data)
        min = np.max([0, min])

    # translate data so it's between (0, max-min):
    image_data -= min
    # normalise data so it's between (0, 1):
    image_data = image_data / (max - min)

    # format data, and convert to image
    image = Image.fromarray(np.uint8(image_data * 255), "L")
    # crop to diameter of sun
    image = image.resize((w, h), Image.LANCZOS)
    # flip image to match original orientation.
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    # rotate images to match
    if rotate and not crop:
        image = image.transpose(Image.ROTATE_180)

    image.save(png_path + name + ".png")


if __name__ == "__main__":
    # -1000 to 1000 arcsec results in a full disk image.
    # Slightly lower values will be closerto the actual edge of the Sun
    filename = "./FITS_DATA/HMI/" + os.listdir("FITS_DATA/HMI/")[0]
    map_ref = sunpy.map.Map(filename)
    top_right = SkyCoord(
        1000 * u.arcsec, 1000 * u.arcsec, frame=map_ref.coordinate_frame
    )
    bottom_left = SkyCoord(
        -1000 * u.arcsec, -1000 * u.arcsec, frame=map_ref.coordinate_frame
    )
    # AIA:
    if AIA:
        fits_path = "FITS_DATA/" + input + "/"
        name = args.name
        test_path = "DATA/TEST/" + name + "/"
        train_path = "DATA/TRAIN/" + name + "/"
        # make directories if they don't exist
        os.makedirs(test_path) if not os.path.exists(test_path) else None
        os.makedirs(train_path) if not os.path.exists(train_path) else None

        for filename in os.listdir(fits_path):
            file_info = filename.split(".")
            date = file_info[2].replace("-", "")
            month = date[4:6]
            if month == "09" or month == "10":
                png_path = test_path
            else:
                png_path = train_path

            if args.random:
                a_max = random.random() * 1800 + 200
            save_to_png(
                name=filename[:-5],
                fits_path=fits_path,
                png_path=png_path,
                min=a_min,
                max=a_max,
                w=w,
                h=h,
                normalise=False,
                crop=True,
                top_right=top_right,
                bottom_left=bottom_left,
            )
    # HMI:
    if HMI:
        fits_path = "FITS_DATA/" + output + "/"
        test_path = "DATA/TEST/" + output + "/"
        train_path = "DATA/TRAIN/" + output + "/"
        # make directories if they don't exist
        os.makedirs(test_path) if not os.path.exists(test_path) else None
        os.makedirs(train_path) if not os.path.exists(train_path) else None

        for filename in os.listdir(fits_path):
            file_info = filename.split(".")
            date = file_info[2].replace("-", "")
            month = date[4:6]
            if month == "09" or month == "10":
                png_path = test_path
            else:
                png_path = train_path
            save_to_png(
                name=filename[:-5],
                fits_path=fits_path,
                png_path=png_path,
                min=m_min,
                max=m_max,
                w=w,
                h=h,
                rotate=True,
                abs=abs,
                crop=True,
                top_right=top_right,
                bottom_left=bottom_left,
            )
