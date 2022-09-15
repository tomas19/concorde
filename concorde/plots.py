import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
import matplotlib as mpl
import contextily as cxt
from sklearn.neighbors import KDTree

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

def plot2D(nc, var, levels, ncvec = None, dxvec = None, dyvec = None, 
           vecsc = None, veccolor = 'k', xlims = None, ylims = None, 
           cbar = False, gdf = None, ts = None, ax = None, fig = None, 
           cmap = 'viridis', fsize = (8, 6), cb_shrink = 1, cb_label = None):
    ''' Funtion to create 2D plots from netcdf files. WIP
        Parameters:
            nc: netcdf object
                adcirc file already loaded to memory
            var: string
                name of the variable to plot. E.g. 'zeta', 'zeta_max'
            levels: numpy array
                contours to plot
            xlims, ylims: list
                limits of the plot
    '''
    
    tri = mpl.tri.Triangulation(nc['x'][:].data, nc['y'][:].data, nc['element'][:,:] - 1)
    if ts == None:
        aux = nc[var][:].data
    else:
        aux = nc[var][ts, :].data
    aux = np.nan_to_num(aux, nan = -99999.0).reshape(-1)
    if ax == None:
        fig, ax = plt.subplots(figsize = fsize)
    
    contours = ax.tricontourf(tri, aux, levels = levels, cmap = cmap)
    
    if ncvec is not None:
        ## plot vectors
        if dxvec is None and dyvec is None:
            ax.quiver(ncvec['x'], ncvec['y'], ncvec['windx'][ts, :], ncws['windy'][ts, :], scale=vecsc)
        else:
            nodes_to_plot = res_nodes_for_plot_vectors(ncvec, dxvec, dyvec)
            ax.quiver(ncvec['x'][nodes_to_plot], ncvec['y'][nodes_to_plot], 
                      ncvec['windx'][ts, nodes_to_plot], ncvec['windy'][ts, nodes_to_plot], scale=vecsc, color = veccolor)
    
    if xlims is not None:
        ax.set_xlim(xlims)
    if ylims is not None:
        ax.set_ylim(ylims)
    cxt.add_basemap(ax, crs = 'EPSG:4326', source=cxt.providers.Stamen.Terrain)
    ax.set_xlabel('Longitude [deg]')
    ax.set_ylabel('Latitude [deg]')
    if cbar == True:
        cb = fig.colorbar(contours, shrink = cb_shrink, extend = 'both')
        cb.set_label(cb_label)
    if gdf is not None:
        gdf.plot(ax = ax, color = 'r')
    return ax
    
def res_nodes_for_plot_vectors(ncObj, dx, dy):
    '''
    '''
    maxx = np.max(ncObj['x'][:].data)
    minx = np.min(ncObj['x'][:].data)
    maxy = np.max(ncObj['y'][:].data)
    miny = np.min(ncObj['y'][:].data)
    xs = np.arange(minx, maxx + dx, dx)
    ys = np.arange(miny, maxy + dy, dy)
    X,Y = np.meshgrid(xs, ys)
    xravel = X.ravel()
    yravel = Y.ravel()
    
    nodes = list(zip(ncObj['x'][:].data, ncObj['y'][:].data))
    tree = KDTree(nodes)
    dist, ind = tree.query(list(zip(xravel, yravel)))
    ind = ind.reshape(-1)
    
    return ind
