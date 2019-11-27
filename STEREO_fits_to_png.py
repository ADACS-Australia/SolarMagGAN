#!/usr/bin/env python3
from fits_to_png import save_to_png
import os
import sunpy
import sunpy.map
import astropy.units as u
from astropy.coordinates import SkyCoord

data = "STEREO"
w = h = 1024
min = 0
max = 150/4
fits_path = "FITS_DATA/STEREO/"
png_path = "DATA/TEST/STEREO/"
filename = "./" + fits_path + os.listdir(fits_path)[0]
map_ref = sunpy.map.Map(filename)
top_right = SkyCoord(1000 * u.arcsec, 1000 * u.arcsec,
                     frame=map_ref.coordinate_frame)
bottom_left = SkyCoord(-1000 * u.arcsec, -1000 * u.arcsec,
                       frame=map_ref.coordinate_frame)

for filename in os.listdir(fits_path):
    os.makedirs(png_path) if not os.path.exists(png_path) else None
    save_to_png(name=filename[:-5],
                fits_path=fits_path,
                png_path=png_path,
                min=min,
                max=max,
                w=w,
                h=h,
                normalise=True,
                crop=True,
                top_right=top_right,
                bottom_left=bottom_left,
                index=0
                )
def save_to_png(name, fits_path, png_path, min, max, w, h,
                normalise=False, rotate=False, abs=False,
                crop=False, top_right=None, bottom_left=None,
                index=1):
    print(name)
    filename = fits_path + name + ".fits"
    hdul = fits.open(filename, memmap=index, ext=0)
    hdul.verify("fix")
    image_data = hdul[index].data

    # find median before cropping
    if normalise and index == 1:
        med = hdul[index].header['DATAMEDN']
    elif normalise and index == 0:
        med = np.median(image_data)

    if abs:
        image_data = np.abs(image_data)
        min = np.max([0, min])

    if crop:
        # Cropping to desired range
        map = sunpy.map.Map(filename)
        if rotate:
            map = map.rotate(angle=180 * u.deg)

        # Cropping
        image_data = map.submap(bottom_left, top_right).data

    # normalise
    if normalise:
        image_data = image_data/med

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
