import astropy.units as u
from astropy.coordinates import SkyCoord

import sunpy.map
import matplotlib.pyplot as plt

path = "./FITS_DATA/Dopplergrams/"
path = path + "hmi.v_45s.20150101_000045_TAI.2.Dopplergram.fits"
smap = sunpy.map.Map(path)

fig = plt.figure()
# Provide the Map as a projection, which creates a WCSAxes object
ax = plt.subplot(projection=smap)

im = smap.plot()

# Prevent the image from being re-scaled while overplotting.
ax.set_autoscale_on(False)

long = [106.4 - 5, 106.4+5]*u.deg
lat = [18.1 - 5, 18.1 + 5]*u.deg
coords = SkyCoord(long, lat, frame="heliographic_carrington")
smap.plot()

p = ax.plot_coord(coords, 'o')
plt.savefig('test.png', dpi=600)
plt.show()
