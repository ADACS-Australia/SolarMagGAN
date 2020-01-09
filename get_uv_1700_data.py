from sunpy.net import Fido, attrs as a
import astropy.units as u  # for AIA
import os
import pandas as pd
# import argparse
# TODO: do many short queryies rather than one large one
email = 'csmi0005@student.monash.edu'
path = "./FITS_DATA/AIA_1700"
os.makedirs(path) if not os.path.exists(path) else None

start_date = pd.Timestamp('2011/01/01 00:00:14')
end_date = pd.Timestamp('2017/01/01 00:00:00')
batch_span = 30*24  # download files in groups spanning 30 days

batch_start = start_date
batch_end = batch_start + pd.Timedelta(hours=batch_span)

while batch_end <= end_date:
    res_aia = Fido.search(a.Time(batch_start, batch_end),
                          a.jsoc.Notify(email),
                          a.jsoc.Series('aia.lev1_uv_24s'),
                          a.jsoc.Segment('image'),
                          a.jsoc.Wavelength(1700*u.AA),
                          a.Sample(12*u.hour)
                          )

    print(res_aia)

    # start and end dates of next batch
    end_date = str(res_aia.get_response(0)['T_REC'][-1])  # end of this batch
    batch_start = pd.Timestamp(end_date)
    batch_end = batch_start + pd.Timedelta(batch_span)

    # download files
    downloaded_files = Fido.fetch(res_aia, path=path)
