#!/usr/bin/env python3
# Python code for retrieving HMI and AIA data
from sunpy.net import Fido, attrs as a
import astropy.units as u  # for AIA
import pickle  # for saving query to file
import os
import argparse
# TODO: do many short queryies rather than one large one
email = 'csmi0005@student.monash.edu'
# can make multiple queries and save the details to files based on the AR and
# retrieve the data later

parser = argparse.ArgumentParser()
parser.add_argument("--AIA",
                    help="download AIA if set",
                    action="store_true"
                    )
parser.add_argument("--HMI",
                    help="download HMI if set",
                    action="store_true"
                    )
parser.add_argument("--STEREO",
                    help="download STEREO if set",
                    action="store_true"
                    )
parser.add_argument("--start",
                    help="start date for AIA and HMI",
                    default='2011/01/01 00:00:00')
parser.add_argument("--end",
                    help="end date for AIA and HMI",
                    default='2017/01/01 00:00:00')

parser.add_argument("--cadence",
                    help="AIA and HMI cadence in hours",
                    type=int,
                    default=12
                    )
parser.add_argument("--AIA_path",
                    help="directory to store AIA data",
                    default='./FITS_DATA/AIA'
                    )
parser.add_argument("--HMI_path",
                    help="directory to store HMI data",
                    default='./FITS_DATA/HMI'
                    )
parser.add_argument("--STEREO_path",
                    help="directory to store STEREO data",
                    default='./FITS_DATA/STEREO'
                    )
parser.add_argument("--STEREO_start",
                    help="start time for STEREO",
                    default='2011/01/01 00:00:00')
parser.add_argument("--STEREO_end",
                    help="end date for STEREO",
                    default='2017/01/01 00:00:00')
parser.add_argument("--STEREO_cadence",
                    help="download 1 in every n images",
                    type=int,
                    default=100
                    )
args = parser.parse_args()

# query duration:
AIA_start = HMI_start = args.start
AIA_end = HMI_end = args.end
STEREO_start = args.STEREO_start
STEREO_end = args.STEREO_end

AIA = args.AIA
AIA_path = args.AIA_path
HMI = args.HMI
HMI_path = args.HMI_path
STEREO = args.STEREO
STEREO_path = args.STEREO_path
cadence = args.cadence*u.hour  # take images every 12 hours
wavelength = 304*u.AA  # 304 Angstroms
stereo_cadence = args.STEREO_cadence  # use 1 in every n images

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
    print(res_stereo[0, 0:-1:100])

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
    print(res_stereo[0, 0:-1:100])
    os.makedirs(STEREO_path) if not os.path.exists(STEREO_path) else None
    downloaded_files = Fido.fetch(res_stereo[0, 0:-1:100], path=STEREO_path)
