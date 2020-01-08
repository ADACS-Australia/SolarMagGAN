from sunpy.net import Fido, attrs as a
import astropy.units as u  # for AIA
import os
# import argparse
# TODO: do many short queryies rather than one large one
email = 'csmi0005@student.monash.edu'
path = "./FITS_DATA/AIA_1700"
os.makedirs(path) if not os.path.exists(path) else None
res_aia = Fido.search(a.Time('2011/01/01 00:00:00', '2017/01/01 00:00:00'),
                      a.jsoc.Notify(email),
                      a.jsoc.Series('aia.lev1_uv_24s'),
                      a.jsoc.Segment('image'),
                      a.jsoc.Wavelength(1700*u.AA),
                      a.Sample(12*u.hour)
                      )

print(res_aia)
downloaded_files = Fido.fetch(res_aia, path=path)
