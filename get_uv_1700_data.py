from sunpy.net import Fido, attrs as a
from sunpy.time import TimeRange
import astropy.units as u  # for AIA
import os
import pandas as pd
# import argparse
# TODO: do many short queryies rather than one large one
email = 'csmi0005@student.monash.edu'
path = "./FITS_DATA/AIA_1700"
os.makedirs(path) if not os.path.exists(path) else None
start_date = pd.Timestamp('2011/01/01 00:00:00')
end_date = pd.Timestamp('2011/02/01 00:00:00')
date = start_date
range = TimeRange(start_date, 24*u.s)
time_series = a.Time(range)
date += pd.Timedelta(hours=12)
while date < end_date:
    range = TimeRange(date, 24*u.s)
    time_series = time_series | a.Time(range)
    date += pd.Timedelta(hours=12)

res_aia = Fido.search(time_series,
                      a.jsoc.Notify(email),
                      a.jsoc.Series('aia.lev1_uv_24s'),
                      a.jsoc.Segment('image'),
                      a.jsoc.Wavelength(1700*u.AA),
                      )

print(res_aia)
downloaded_files = Fido.fetch(res_aia, path=path)
