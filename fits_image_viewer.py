import matplotlib.pyplot as plt
from astropy.io import fits
import matplotlib.colors as colors
import os
import numpy as np

hdul = fits.open("test.fits")
hdul.verify("fix")
image_data = hdul[1].data


plt.figure()
plt.imshow(image_data, cmap="gray", vmin=0, vmax=50)
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(image_data, norm=colors.LogNorm(vmin=10, vmax=50))
plt.colorbar()
plt.show()

for filename in os.listdir("../data/ar1/"):
    hdul = fits.open("../data/ar1/" + filename, ext=0)
    hdul.verify("fix")
    image_data = hdul[1].data
    max = np.max(image_data)
    min = np.min(image_data)
    plt.figure(filename)
    plt.imshow(image_data, cmap="gray", vmin=-50, vmax=50)
    plt.colorbar()
    plt.show()
