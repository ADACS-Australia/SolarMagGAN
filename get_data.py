# Python code for retrieving HMI and AIA data
from sunpy.net import Fido, attrs as a
import astropy.units as u  # for AIA
import pickle  # for saving query to file
import os

email = 'csmi0005@student.monash.edu'
# can make multiple queries and save the details to files based on the AR and
# retrieve the data later

# query duration:
start = '2011/01/01 00:00:00'
end = '2017/01/01 00:00:00'

cadence = 300*u.day  # 300 days
wavelength = 304*u.AA  # 304 Angstroms

res_aia = Fido.search(a.Time(start, end),
                      a.jsoc.Notify(email),
                      a.jsoc.Series('aia.lev1_euv_12s'),
                      a.jsoc.Segment('image'),
                      a.jsoc.Wavelength(wavelength),
                      a.Sample(cadence)
                      )

print(res_aia)

# magnetogram results
res_hmi = Fido.search(a.Time(start, end),
                      a.jsoc.Notify(email),
                      a.jsoc.Series('hmi.m_45s'),
                      a.Sample(cadence)
                      )

print(res_hmi)

# save the the query details to file
with open('data_query.pkl', 'wb') as f:
    pickle.dump([start, end, res_aia, res_hmi], f)

# download data
path1 = './FITS_DATA/AIA'
path2 = './FITS_DATA/HMI'
os.makedirs(path1) if not os.path.exists(path1) else None
downloaded_files = Fido.fetch(res_aia, path='./FITS_DATA/AIA')
os.makedirs(path2) if not os.path.exists(path2) else None
downloaded_files = Fido.fetch(res_hmi, path='./FITS_DATA/HMI')
