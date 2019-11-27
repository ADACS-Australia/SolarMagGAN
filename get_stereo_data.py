#!/usr/bin/env python3
import urllib.request
# first part of url before date
url_1 = "https://stereo-ssc.nascom.nasa.gov/data/ins_data/secchi/L0/a/img/euvi/"
# second part of url after date
url_2 = "_115615_n4euA.fts"

dates = ["20140604", "20140605", "20140606", "20140607", "20140608", "20140609", "20140610"]

for date in dates:
    fname = date + url_2 # filename
    url = url_1 + date + "/" + fname
    f = open('FITS_DATA/STEREO/' + fname , "wb")
    data = urllib.request.urlopen(url)
    f.write(data.read())
    f.close()
