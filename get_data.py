# Python code for retrieving HMI and AIA data
from sunpy.net import Fido, attrs as a
import astropy.units as u  # for AIA
import pickle  # for saving query to file
import os

email = 'csmi0005@student.monash.edu'
# can make multiple queries and save the details to files based on the AR and
# retrieve the data later

# query duration:
start = '2015/01/01 00:00:00'
end = '2015/01/01 00:00:00'

AIA = True
AIA_path = './FITS_DATA/AIA'
HMI = True
HMI_path = './FITS_DATA/HMI'
cadence = 12*u.hour  # take images every 12 hours
wavelength = 304*u.AA  # 304 Angstroms

if AIA:
    res_aia = Fido.search(a.Time(start, end),
                          a.jsoc.Notify(email),
                          a.jsoc.Series('aia.lev1_euv_12s'),
                          a.jsoc.Segment('image'),
                          a.jsoc.Wavelength(wavelength),
                          a.Sample(cadence)
                          )

    print(res_aia)

    # save the the query details to file
    with open('data_query_aia.pkl', 'wb') as f:
        pickle.dump([start, end, res_aia], f)

if HMI:
    res_hmi = Fido.search(a.Time(start, end),
                          a.jsoc.Notify(email),
                          a.jsoc.Series('hmi.m_45s'),
                          a.Sample(cadence)
                          )

    # save the the query details to file
    with open('data_query_hmi.pkl', 'wb') as f:
        pickle.dump([start, end, res_hmi], f)
    print(res_hmi)


if AIA:
    print('AIA\nStart: ' + start + '\nEnd: ' + end)
    print(res_aia)
    os.makedirs(AIA_path) if not os.path.exists(AIA_path) else None
    downloaded_files = Fido.fetch(res_aia, path=AIA_path)

if HMI:
    print('HMI\nStart: ' + start + '\nEnd: ' + end)
    print(res_hmi)
    os.makedirs(HMI_path) if not os.path.exists(HMI_path) else None
    downloaded_files = Fido.fetch(res_hmi, path=HMI_path)
