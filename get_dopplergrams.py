#!/usr/bin/env python3
# Python code for retrieving HMI and AIA data
from sunpy.net import Fido, attrs as a
import os
import argparse
# TODO: do many short queryies rather than one large one
email = 'csmi0005@student.monash.edu'
# can make multiple queries and save the details to files based on the AR and
# retrieve the data later

parser = argparse.ArgumentParser()

parser.add_argument("--start",
                    help="start date for data collection",
                    default='2015/01/01 00:00:00')
parser.add_argument("--end",
                    help="end date for data collection",
                    default='2015/01/01 01:00:00')

parser.add_argument("--path",
                    help="directory to store data",
                    default='./FITS_DATA/Dopplergrams'
                    )
args = parser.parse_args()

# query duration:
start = args.start
end = args.end
path = args.path
res = Fido.search(a.Time(start, end),
                  a.jsoc.Notify(email),
                  a.jsoc.Series('hmi.V_45s'),
                  )

print('Start: ' + start + '\nEnd: ' + end)
print(res)
os.makedirs(path) if not os.path.exists(path) else None
downloaded_files = Fido.fetch(res, path=path)
