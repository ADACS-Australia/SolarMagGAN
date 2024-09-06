import argparse
import os

import astropy.units as u
from sunpy.net import Fido
from sunpy.net import attrs as a

parser = argparse.ArgumentParser()
parser.add_argument(
    "dataset",
    help="type of data to download",
    choices=["AIA", "HMI", "STEREO"],
    default="AIA",
)
parser.add_argument("output", help="output directory for data")
parser.add_argument("--email", help="email address for JSOC downloads")
parser.add_argument(
    "--start", help="start date for AIA and HMI", default="2011/01/01 00:00:00"
)
parser.add_argument(
    "--end", help="end date for AIA and HMI", default="2017/01/01 00:00:00"
)
parser.add_argument("--cadence", type=int, default=12)
parser.add_argument(
    "--wavelength", help="wavelength of AIA images in angstroms", type=int, default=304
)

args = parser.parse_args()

# Check the email looks like an email
if args.email and "@" not in args.email:
    print("Email doesn't look like an address.")
    exit(1)

# Check the output directory exists
if not os.path.exists(args.output):
    print("Output directory doesn't exist.")
    exit(1)

# Check the output directory is writable
if not os.access(args.output, os.W_OK):
    print("Output directory isn't writable.")
    exit(1)

if args.dataset in ["AIA", "HMI"] and not args.email:
    print("Email address required for AIA and HMI.")
    exit(1)


if args.dataset == "AIA" or args.dataset == "HMI":
    cadence = args.cadence * u.hour  # take images every n hours
else:
    cadence = args.cadence  # use 1 in every n images

dataset = args.dataset
wavelength = args.wavelength * u.AA  # wavelength in angstroms
start = args.start
end = args.end

# Set up search parameters
search = [a.Time(start, end)]

print(f"===> Searching for {dataset} data:")
print("Start:", start)
print("End:", end)

if dataset == "AIA":
    print("Wavelength:", wavelength)
    print("Cadence:", cadence)
    search += [
        a.jsoc.Notify(args.email),
        a.jsoc.Series("aia.lev1_euv_12s"),
        a.jsoc.Segment("image"),
        a.jsoc.Wavelength(wavelength),
        a.Sample(cadence),
    ]
elif dataset == "HMI":
    print("Cadence:", cadence)
    search += [
        a.jsoc.Notify(args.email),
        a.jsoc.Series("hmi.m_45s"),
        a.Sample(cadence),
    ]
elif dataset == "STEREO":
    print("Cadence:", cadence)
    print("Wavelength:", wavelength)
    search += [
        a.vso.Source("STEREO_B"),
        a.Instrument("EUVI"),
        a.Wavelength(wavelength),
    ]

result = Fido.search(*search)

if dataset == "STEREO":
    result = result[0, ::cadence]

print(result)

Fido.fetch(result, path=args.output)
