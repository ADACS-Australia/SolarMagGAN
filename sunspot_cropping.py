#!/usr/bin/env python3
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
import os
import matplotlib.pyplot as plt
import argparse


# arguements to be specefied when running script
parser = argparse.ArgumentParser()

parser.add_argument("--fits_path",
                    help="directory of fit data",
                    default="FITS_DATA/Dopplergrams")
parser.add_argument("--save_path",
                    help="directory to save cropped pngs",
                    default="FITS_DATA/Dopplergrams_cropped")
parser.add_argument("--crop_size",
                    help="width and height of crop length in pixels",
                    type=int,
                    default=512)
args = parser.parse_args()

# parameters
fits_path = args.fits_path
save_path = args.save_path
length = args.crop_size
size = (length, length)
# create save path if it doesn't already exist
os.makedirs(save_path) if not os.path.exists(save_path) else None

# for each fits file, find all sunspot coordinates and crop around them
for file in os.listdir(fits_path):
    sunspot_coords = [(1000, 2000), (2048, 2048)]
    filename = fits_path + "/" + file
    hdul = fits.open(filename, memmap=False, ext=0, ignore_missing_end=True)
    hdul.verify("fix")
    data = hdul[1].data
    header = hdul[1].header
    wcs = WCS(header)

    # plt.imshow(hdul[1].data, origin="lower")
    # plt.show()
    for position in sunspot_coords:
        cutout = Cutout2D(data, position, size, wcs=wcs)
        # update data and header
        # hdul[1].header.update(cutout.wcs.to_header())
        # plt.imshow(hdul[1].data, origin="lower")
        # plt.show()
        position_string = str(position[0]) + "_" + str(position[1])

        # new hdu:
        hdu = fits.PrimaryHDU()
        hdu.data = cutout.data

        # remove checksum and datasum of old header
        # header = header[:-4]
        # update old header with new info:
        header.update(hdu.header)
        # update with cropping info:
        header.update(cutout.wcs.to_header())

        # update new header
        hdu.header.update(header)
        hdu.update_header()
        print(hdu.header)

        hdu.verify("fix")

        hdu.writeto(save_path + "/" + position_string + file,
                    output_verify='ignore',
                    overwrite=True
                    )
