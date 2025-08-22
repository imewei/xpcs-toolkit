import h5py
import os
from typing import Union


def isNeXusFile(filename: str) -> bool:
    with h5py.File(filename, "r") as f:
        if "/entry/instrument/bluesky/metadata/" in f:
            return True
    return False


def isLegacyFile(filename: str) -> bool:
    with h5py.File(filename, "r") as f:
        if "/xpcs/Version" in f:
            return True
    return False


def get_ftype(filename: str) -> Union[str, bool]:
    if not os.path.isfile(filename):
        return False

    if isLegacyFile(filename):
        return 'legacy'
    elif isNeXusFile(filename):
         return 'nexus' 
    else:
        return False
