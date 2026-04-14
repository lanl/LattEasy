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



    


def create_geom_edist(rock, args):
    """
    Modifies a given rock matrix, calculates its Euclidean distance, creates a geometry file based on the 
    modified matrix and returns it.

    This function primarily modifies an input rock matrix according to several configuration options provided 
    in the 'args' parameter, calculates the Euclidean distance for the modified matrix, and writes it to a .dat 
    file. Additionally, the function ensures that all the boundary conditions have bounding box nodes, and it 
    pads the matrix if the 'num_slices' argument is provided.

    Parameters
    ----------
        rock : np.array
            The input rock matrix to be modified.
        args : argparse.Namespace
            An object containing various configuration options. This includes the 'swapXZ', 'scale_2', 'add_mesh', 
            'num_slices', 'print_size' and 'name' flags, and 'loc' directory path.

    Returns
    -------
        erock : np.array
            The modified rock matrix after the Euclidean distance calculation.

    Raises
    ------
        NotImplementedError
            If the 'scale_2' or 'add_mesh' features are attempted to be used as they are not yet implemented.

    Notes
    -----
        The function uses numpy's 'transpose', 'pad' and 'tofile' functions, as well as a custom 'edist' function 
        for Euclidean distance calculation.

    Examples
    --------
        >>> import numpy as np
        >>> import argparse
        >>> rock = np.array([[[0, 1, 0], [1, 0, 1], [0, 1, 0]], [[1, 0, 1], [0, 1, 0], [1, 0, 1]], [[0, 1, 0], [1, 0, 1], [0, 1, 0]]])
        >>> args = argparse.Namespace(swapXZ=False, scale_2=False, add_mesh=False, num_slices=2, print_size=True, name='geom', loc='./')
        >>> mod_rock = create_geom_edist(rock, args)
    """
    # If the 'swapXZ' flag is set, transpose the rock matrix accordingly
    if args.swapXZ:
        rock = rock.transpose([2, 1, 0])

    # If the 'scale_2' flag is set, raise an error as this feature is not yet implemented
    if args.scale_2:
        raise NotImplementedError('Feature not yet implemented')

    # Calculate the Euclidean distance for the rock matrix
    erock = edist(rock)
    
    # Ensure all the BCs have BB nodes
    erock[0,:,:] = erock[:,0,:] = erock[:,:,0] = 1
    erock[-1,:,:] = erock[:,-1,:] = erock[:,:,-1] = 1
    
    # Reopen the pores
    erock[rock==0] = 0
    
    # Get the final matrix with values [0,1,2]
    erock[(erock > 0) * (erock < 2)] = 1
    erock[erock > 1] = 2
    
    # If the 'add_mesh' flag is set, raise an error as this feature is not yet implemented
    if args.add_mesh:
        raise NotImplementedError('Feature not yet implemented')
    
    # If the 'num_slices' argument is provided, pad the 'erock' array accordingly
    if args.num_slices:
        erock = np.pad(erock, [(args.num_slices, args.num_slices), (0, 0), (0, 0)])
    
    # Determine the geometry name based on the 'print_size' flag
    if args.print_size:
        size = erock.shape
        geom_name = f'{args.name}_{size[0]}_{size[1]}_{size[2]}'
    else:
        geom_name = args.name
    
    # Modify the 'erock' array's data type and values for final output
    erock = erock.astype(np.int16)
    erock[erock==0] = 2608
    erock[erock==1] = 2609
    erock[erock==2] = 2610
    
    # Write the 'erock' array to a .dat file
    erock.flatten().tofile(f'{args.loc}/input/{geom_name}.dat')
    
    return erock




def erase_regions(rock):
    """
    Identifies and erases isolated regions within a given rock matrix.

    The function employs the `label` function from the skimage.measure module to detect connected components 
    in the input rock matrix. Isolated regions are identified as those components which are not connected to 
    the main body of the matrix (where the corresponding element in the labeled matrix is greater than 1). 
    These regions are then removed (set to 0) from the rock matrix.

    Parameters
    ----------
        rock : np.array
            The input rock matrix to be processed.

    Returns
    -------
        rock : np.array
            The modified rock matrix with isolated regions removed.

    Notes
    -----
        The function uses skimage.measure's 'label' function for connected components detection. The background 
        is set to 1, and connectivity is set to 1 to consider only orthogonally adjacent points. 

    Examples
    --------
    >>> import numpy as np
    >>> rock = np.array([[[0, 1, 0], [1, 0, 1], [0, 1, 0]], [[1, 0, 1], [0, 1, 0], [1, 0, 1]], [[0, 1, 0], [1, 0, 1], [0, 1, 0]]])
    >>> mod_rock = erase_regions(rock)
    """
    # Use the measure.label function from the skimage.measure module to identify connected components in the rock matrix
    # Here, the background is set to 1, and connectivity is set to 1 to consider only orthogonally adjacent points
    blobs_labels = measure.label(rock, background=1, connectivity=1)
    
    # Erase all isolated regions within the rock matrix
    # This is done by setting all elements in 'rock' that are part of an isolated region (where the corresponding element in 'blobs_labels' is greater than 1) to 0
    rock[blobs_labels > 1] = 0
    
    return rock