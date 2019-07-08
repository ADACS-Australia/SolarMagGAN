from astropy.io import fits
import os
from PIL import Image


for filename in os.listdir("../data/ar1/"):
    hdul = fits.open("../data/ar1/" + filename, ext=0)
    hdul.verify("fix")
    image_data = hdul[1].data
    image = Image.fromarray(image_data)
    image = image.convert("L")
    image.save("../png_data/ar1/" + filename[:-5] + ".png")
