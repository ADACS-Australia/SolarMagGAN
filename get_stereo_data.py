#!/usr/bin/env python3
import urllib.request
# first part of url before date
url1 = "https://stereo-ssc.nascom.nasa.gov/data/ins_data/secchi/L0/a/img/euvi/"
# second part of url after date
url2 = "_115615_n4euA"

dates = ["20140604", "20140605", "20140606", "20140607",
         "20140608", "20140609", "20140610"]

for date in dates:
    fname = date + url2  # filename
    print(fname)
    url = url1 + date + "/" + fname + ".fts"
    f = open('FITS_DATA/STEREO/' + fname + ".fits", "wb")
    data = urllib.request.urlopen(url)
    f.write(data.read())
    f.close()
