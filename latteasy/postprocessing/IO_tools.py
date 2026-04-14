from subprocess import run
import os
from os import mkdir
from shutil import copy, move
from argparse import Namespace

import pandas as pd
import numpy as np

import pickle
from copy import copy as cp

from skimage import measure
from scipy.ndimage import distance_transform_edt as edist




def read_permeability(file_path="pore.txt"):
    """
    Reads the permeability value from a ``pore.txt`` file.

    The function retrieves the permeability value, which is expected to be found after the first 
    equal sign ('=') and before the substring '\nName' on the fourth line from the end of the file.

    It uses the pandas library to read and handle the data file. If the permeability value cannot be 
    converted to a float or if the file ``pore.txt`` cannot be found, exceptions will be raised.
    
    Parameters
    -----------
        file_path : str, optional
            Location of the ``pore.txt`` file. Defaults to the current
            directory.
        
    Returns
    --------
        perm : float
            The permeability value parsed from ``pore.txt``.

    Notes
    ------
        None
    
    Raises
    -------
        ValueError
            If the permeability value is not a valid float.
        FileNotFoundError
            If ``pore.txt`` cannot be found.
    """
    # Read the data file using pandas
    file = pd.read_csv(file_path, header=None)

    # Find the index of the first equal sign in the last fourth line
    ind1 = str(file.iloc[-4]).find('=')+2
    # Find the index of the substring '\nName' in the last fourth line
    ind2 = str(file.iloc[-4]).find('\nName')

    # Extract the permeability value from the string between the two indices and convert it to float
    perm = float(str(file.iloc[-4])[ind1:ind2])
    return perm



               