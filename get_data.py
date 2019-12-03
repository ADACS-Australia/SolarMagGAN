# Python code for retrieving HMI and AIA data
from sunpy.net import Fido, attrs as a
import astropy.units as u  # for AIA
import pickle  # for saving query to file
import os
# TODO: do many short queryies rather than one large one
email = 'csmi0005@student.monash.edu'
# can make multiple queries and save the details to files based on the AR and
# retrieve the data later

# query duration:
AIA_start = HMI_start = '2015/01/01 00:00:00'
AIA_end = HMI_end = '2015/01/01 00:00:00'
STEREO_start = '2007-01-01'
STEREO_end = '2007-01-01T01:00:00'

AIA = False
AIA_path = './FITS_DATA/AIA'
HMI = False
HMI_path = './FITS_DATA/HMI'
STEREO = True
STEREO_path = './FITS_DATA/STEREO'
cadence = 12*u.hour  # take images every 12 hours
wavelength = 304*u.AA  # 304 Angstroms

if AIA:
    res_aia = Fido.search(a.Time(AIA_start, AIA_end),
                          a.jsoc.Notify(email),
                          a.jsoc.Series('aia.lev1_euv_12s'),
                          a.jsoc.Segment('image'),
                          a.jsoc.Wavelength(wavelength),
                          a.Sample(cadence)
                          )

    print(res_aia)

    # save the the query details to file
    with open('data_query_aia.pkl', 'wb') as f:
        pickle.dump([AIA_start, AIA_end, res_aia], f)

if HMI:
    res_hmi = Fido.search(a.Time(HMI_start, HMI_end),
                          a.jsoc.Notify(email),
                          a.jsoc.Series('hmi.m_45s'),
                          a.Sample(cadence)
                          )

    # save the the query details to file
    with open('data_query_hmi.pkl', 'wb') as f:
        pickle.dump([HMI_start, HMI_end, res_hmi], f)
    print(res_hmi)

if STEREO:
    res_stereo = Fido.search(a.Wavelength(wavelength),
                             a.vso.Source('STEREO_B'),
                             a.Instrument('EUVI'),
                             a.Time(STEREO_start, STEREO_end),
                             )


if AIA:
    print('AIA\nStart: ' + AIA_start + '\nEnd: ' + AIA_end)
    print(res_aia)
    os.makedirs(AIA_path) if not os.path.exists(AIA_path) else None
    downloaded_files = Fido.fetch(res_aia, path=AIA_path)

if HMI:
    print('HMI\nStart: ' + HMI_start + '\nEnd: ' + HMI_end)
    print(res_hmi)
    os.makedirs(HMI_path) if not os.path.exists(HMI_path) else None
    downloaded_files = Fido.fetch(res_hmi, path=HMI_path)

if STEREO:
    print('STEREO\nStart: ' + STEREO_start + '\nEnd: ' + STEREO_end)
    print(res_stereo)
    os.makedirs(STEREO_path) if not os.path.exists(STEREO_path) else None
    downloaded_files = Fido.fetch(res_stereo, path=STEREO_path)
