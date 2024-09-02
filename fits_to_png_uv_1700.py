from PIL import Image
import numpy as np
import os
import sunpy
import sunpy.map
import astropy.units as u
from astropy.coordinates import SkyCoord

dataset = "AIA_1700"
w = h = 1024  # desired width and height of png


def save_to_png(
    name,
    fits_path,
    png_path,
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
    # -1000 to 1000 arcsec results in a full disk image.
    # Slightly lower values will be closer to the actual edge of the Sun
    swap_map = sunpy.map.Map(filename)
    top_right = SkyCoord(
        1000 * u.arcsec, 1000 * u.arcsec, frame=swap_map.coordinate_frame
    )
    bottom_left = SkyCoord(
        -1000 * u.arcsec, -1000 * u.arcsec, frame=swap_map.coordinate_frame
    )

    # Cropping
    image_data = swap_map.submap(bottom_left, top_right).data

    # minimum value of data:
    min = np.min(image_data)

    # clip above 95th percentile
    image_data = np.clip(image_data, min, np.percentile(image_data, 99))

    # translate data so it's between (1, e):

    image_data -= min
    # max value of data
    max = np.max(image_data)

    image_data = image_data / (max)
    image_data = image_data * (np.e - 1)
    image_data += 1

    # take log of data
    image_data = np.log(image_data)

    # format data, and convert to image
    image = Image.fromarray(np.uint8(image_data * 255), "L")
    # crop to diameter of sun
    image = image.resize((w, h), Image.LANCZOS)
    # flip image to match original orientation.
    image = image.transpose(Image.FLIP_TOP_BOTTOM)

    image.save(png_path + name + ".png")


if __name__ == "__main__":
    # AIA:

    fits_path = "FITS_DATA/" + dataset + "/"
    test_path = "DATA/TEST/" + dataset + "/"
    train_path = "DATA/TRAIN/" + dataset + "/"
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
            w=w,
            h=h,
            crop=True,
        )
