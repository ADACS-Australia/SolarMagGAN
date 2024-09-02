# Python code for retrieving HMI and AIA data
from sunpy.net import Fido
import pickle  # for saving query to file
import os

AIA = True
HMI = True
email = "csmi0005@student.monash.edu"


# download data
if AIA:
    with open("data_query_aia.pkl", "rb") as f:
        start, end, res_aia = pickle.load(f)
    print("AIA\nStart: " + start + "\nEnd: " + end)
    print(res_aia)
    path1 = "./FITS_DATA/AIA"
    os.makedirs(path1) if not os.path.exists(path1) else None
    downloaded_files = Fido.fetch(res_aia, path="./FITS_DATA/AIA")
if HMI:
    with open("data_query_hmi.pkl", "rb") as f:
        start, end, res_hmi = pickle.load(f)
    print("HMI\nStart: " + start + "\nEnd: " + end)
    print(res_hmi)
    path2 = "./FITS_DATA/HMI"
    os.makedirs(path2) if not os.path.exists(path2) else None
    downloaded_files = Fido.fetch(res_hmi, path="./FITS_DATA/HMI")
