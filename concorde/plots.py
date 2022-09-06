import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt

def scatter_interpolate(x,y,res=17,method='cubic',normalise=True, return_res = False):
    '''Interpolates scatter data on a 2D histogram based on density. This is a proxy of the KDE estimator, with lower (still acceptable) resolution but much faster.
    
    Parameters:
        x: Pandas series or array
            Series with X data
        y: Pandas series or array
            Series with Y data
        res: Int (Default=17)
            Grid resolution.
        method: Str (Default='cubic')
            Scipy griddata interpolation method.
        normalise: Boolean (Default= False)
            True: Normalize data to probability density Probability/unit_X*unit_y
            False: Point density per unit area Points/unit_X*unit_y
        return_res: bool (default: False)
            If True: returns gridZ, dx, dy
            If False: returns gridZ
    ------------------------------------------------------------------------------------------------
    Returns:
        gridZ: Array
            Array with the colors based on histogram density
    '''
    

    # Setup a lin-grid for interpolation
    ## Centre points
    cx = np.linspace(min(x), max(x), res)
    cy = np.linspace(min(y), max(y), res)
    
    ## Resolution
    dx = cx[1]-cx[0]
    dy = cy[1]-cy[0]
    
    ## Edege points
    ex = np.linspace(min(x)-dx*0.5, max(x)+dx*0.5, res+1)
    ey = np.linspace(min(y)-dy*0.5, max(y)+dy*0.5, res+1)
        
    # Numpy method for calculating 2D histogram
    histoData, exh, eyh = np.histogram2d(x,y,[ex,ey],normed=normalise)

    # EXTRACT HISTOGRAM VALIUES
    # unpacking data
    h = []
    for j in range(len(cy)):
        for i in range(len(cx)):
            h.append(histoData[i,j])        
    
    # PRE-PROCESS GRID DATA
    xg, yg = np.meshgrid(cx, cy)
    xg = xg.ravel()
    yg = yg.ravel()
    
    ## Interpolate histogram density data to original scatter
    gridZ = interpolate.griddata((xg, yg),h, (x,y),method=method,rescale=True)
        
    if return_res:
        return gridZ, dx, dy
    else:
        return gridZ
