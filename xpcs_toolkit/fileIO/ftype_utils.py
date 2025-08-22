import h5py
import numpy
import sys
import os


def isNeXusFile(filename):
    with h5py.File(filename, "r") as f:
        if "/entry/instrument/bluesky/metadata/" in f:
            return True
    return False


def isLegacyFile(filename):
    with h5py.File(filename, "r") as f:
        if "/xpcs/Version" in f:
            return True


def get_ftype(filename: str):
    if not os.path.isfile(filename):
        return False

    if isLegacyFile(filename):
        return 'legacy'
    elif isNeXusFile(filename):
         return 'nexus' 
    else:
        return False
