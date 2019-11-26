#!/usr/bin/env python3
import urllib.request
f = open('FITS_DATA/STEREO/test.fits', "wb")
data = urllib.request.urlopen("https://stereo-ssc.nascom.nasa.gov/data" +
                              "/ins_data/secchi/L0/a/img/euvi/20090116" +
                              "/20090116_000530_n4euA.fts")
f.write(data.read())
f.close()
