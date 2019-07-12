# Python code for retrieving HMI and AIA data

from sunpy.net import Fido, attrs as a
# from sunpy.net import jsoc
import astropy.units as u  # for AIA
import pickle  # for saving query to file

email = 'csmi0005@student.monash.edu'
# can make multiple queries and save the details to files based on the AR and
# retrieve the data later
time_ar = ['2019-03-07T19:00:00', '2019-03-07T19:30:00']  # AR NOAA 12734

res_ic = Fido.search(
                     a.Time(time_ar[0], time_ar[1]),
                     a.jsoc.Series('hmi.ic_45s'),
                     a.jsoc.Notify(email)
                     )
# res_m = Fido.search(
#                     a.Time(time_ar[0], time_ar[1]),
#                     a.jsoc.Series('hmi.m_45s'),
#                     a.jsoc.Notify(email)
#                     )
res_304 = Fido.search(
                      a.Time(time_ar[0], time_ar[1]),
                      a.jsoc.Notify(email),
                      a.jsoc.Series('aia.lev1_euv_12s'),
                      a.jsoc.Segment('image'),
                      a.jsoc.Wavelength(304*u.AA)
                      )

# save the the query details to file
with open('ar1.pkl', 'wb') as f:
    pickle.dump([time_ar, res_ic, res_304], f)


# Wait about 15 minutes for the requests to be processed
# if starting a new python session load the packages above

# load query info
with open('ar1.pkl', 'rb') as f:
    time_ar, res_ic, res_m, res_304 = pickle.load(f)

# download data
# wait before running these commands, as request nets to be processed
downloaded_files = Fido.fetch(res_ic, path='./ar1/')
# downloaded_files = Fido.fetch(res_m, path='./ar1/')
downloaded_files = Fido.fetch(res_304, path='./ar1/')
