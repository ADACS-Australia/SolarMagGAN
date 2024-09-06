import argparse
import os
import random

import astropy.units as u
import numpy as np
import sunpy
import sunpy.map
from astropy.coordinates import SkyCoord
from astropy.io import fits
from PIL import Image


def fits_to_png(
    fits_file,
    output_file,
    min_clip=-100,
    max_clip=100,
    width=1024,
    height=1024,
    normalise=False,
    rotate=False,
    absolute=False,
    crop=False,
):

    cf = sunpy.map.Map(fits_file).coordinate_frame
    top_right = SkyCoord(1000 * u.arcsec, 1000 * u.arcsec, frame=cf)
    bottom_left = SkyCoord(-1000 * u.arcsec, -1000 * u.arcsec, frame=cf)

    hdul = fits.open(fits_file, memmap=True, ext=0)
    hdul.verify("fix")
    if not crop:
        image_data = hdul[1].data
    else:
        # Cropping to desired range
        map = sunpy.map.Map(fits_file)
        if rotate:
            map = map.rotate(angle=180 * u.deg)

        # Cropping
        image_data = map.submap(bottom_left, top_right).data

    # clip data between (min, max):
    image_data = np.clip(image_data, min_clip, max_clip)

    if normalise:
        med = hdul[1].header["DATAMEDN"]
        # make sure median is between min and max:
        np.clip(med, min_clip, max_clip)
        image_data = image_data / med

    if absolute:
        image_data = np.abs(image_data)
        min_clip = np.max([0, min_clip])

    # translate data so it's between (0, max-min):
    image_data -= min_clip
    # normalise data so it's between (0, 1):
    image_data = image_data / (max_clip - min_clip)

    # format data, and convert to image
    image = Image.fromarray(np.uint8(image_data * 255), "L")
    # crop to diameter of sun
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    # flip image to match original orientation.
    image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    # rotate images to match
    if rotate and not crop:
        image = image.transpose(Image.ROTATE_180)

    image.save(output_file)


def main():
    parser = argparse.ArgumentParser(description="Convert a FITS file to a PNG image.")
    parser.add_argument("fits_files", type=str, nargs="+", help="The input FITS files.")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="The output directory. Defaults to the current directory.",
        default=".",
    )
    parser.add_argument(
        "--random",
        help="randomly assign upper bound pixel value (use this for AIA images). 'max' argument will be ignored.",
        action="store_true",
        default=False,
    )
    parser.add_argument("--min", type=int, help="Minimum pixel value", default=-100)
    parser.add_argument("--max", type=int, help="Maximum pixel value", default=100)
    parser.add_argument(
        "--normalise", help="Normalise the image", action="store_true", default=False
    )
    parser.add_argument(
        "--rotate", help="Rotate the image", action="store_true", default=False
    )
    parser.add_argument(
        "--abs", help="Absolute value of the image", action="store_true", default=False
    )
    parser.add_argument(
        "--crop", help="Crop the image", action="store_true", default=False
    )
    args = parser.parse_args()

    # check if the output directory exists
    if not os.path.isdir(args.output):
        print(f"Directory {args.output} does not exist.")
        exit(1)

    for fits_file in args.fits_files:

        # check if the file exists
        if not os.path.isfile(fits_file):
            print(f"File {fits_file} does not exist.")
            continue

        # get the name of the input file, minus the path
        basename = os.path.basename(fits_file)

        # if extension is either .fits or .fit or .fts , replace it with .png
        if basename.endswith(".fits"):
            output_file = basename.strip(".fits") + ".png"
        elif basename.endswith(".fit"):
            output_file = basename.strip(".fit") + ".png"
        elif basename.endswith(".fts"):
            output_file = basename.strip(".fts") + ".png"
        else:
            output_file = basename + ".png"

        # join the output directory with the output file
        output_file = os.path.join(args.output, output_file)

        if args.random:
            max_clip = random.random() * 1800 + 200
        else:
            max_clip = args.max

        fits_to_png(
            fits_file,
            output_file,
            min_clip=args.min,
            max_clip=max_clip,
            normalise=args.normalise,
            rotate=args.rotate,
            absolute=args.abs,
            crop=args.crop,
        )
        print(f"Converted {fits_file} to {output_file}")


if __name__ == "__main__":
    main()
