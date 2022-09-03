import os
import matplotlib as mpl
import numpy as np
import geopandas as gpd
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon, Point
from scipy import interpolate
from tqdm import tqdm
import pandas as pd
#import cv2

def animate_list_imgs(imgs, fps, anim_name):
    ''' Function to animate list of images.
        Parameters
            list_imgs: list
                list with complete path of all imgs
            fps: int
                frames per second
            anim_name: str
                output name, only name without path. The video will be saved in the notebook path. For the moment, only avi extension has been tested
        Returns
            None
    '''
    im = cv2.imread(imgs[0])
    h, w, l = im.shape
    size = (w, h)
    
    out = cv2.VideoWriter(anim_name, cv2.VideoWriter_fourcc(*'DIVX'), 10, size)
    
    for i in tqdm(imgs):
        im = cv2.imread(i)
        out.write(im)
    out.release()

def get_list(directory, ends = None, starts = None, contains = None, not_contains = None):
    '''Generates list with all the files path (path + file name) inside directory which
        starts with "starts" and ends with "ends"

    Parameters:
        directory: string
            Folder path
        ends: string. Optional, default None
            e.g. the file extension, '.csv', '.txt', etc
        starts: string. Optional, default None
            e.g. the proyect number 'A2022'
        contains= string. Optional, default None
            e.g. 'time series' in the middle of the name
    ---------------------------------------------------------------------------------------
    Returns:
        List with paths of the files which satisfy the conditions
    '''
    listfiles = []
    if (ends == None and starts == None):
        for cur, dirs, files in os.walk(directory): #genera lista con la direccion de cada archivo gz dentro de la carpeta NCfiles
            for f in files:
                listfiles.append(os.path.join(cur, f))
    elif (ends != None and starts == None):
        for cur, dirs, files in os.walk(directory): #genera lista con la direccion de cada archivo gz dentro de la carpeta NCfiles
            for f in files:
                if f.endswith(ends):
                    listfiles.append(os.path.join(cur, f))
    elif (ends == None and starts != None):
        for cur, dirs, files in os.walk(directory): #genera lista con la direccion de cada archivo gz dentro de la carpeta NCfiles
            for f in files:
                if f.startswith(starts):
                    listfiles.append(os.path.join(cur, f))
    else:
        for cur, dirs, files in os.walk(directory): #genera lista con la direccion de cada archivo gz dentro de la carpeta NCfiles
            for f in files:
                if (f.endswith(ends) and f.startswith(starts)):
                    listfiles.append(os.path.join(cur, f))
    if contains != None:
        listfiles = [x for x in listfiles if contains in x]
    if not_contains != None:
        listfiles = [x for x in listfiles if not_contains not in x]
        
    if len(listfiles) == 0:
        sys.exit('The list is empty')
        return None
    else:
        return listfiles

def pntFromNC(ncObj, pnts, n = 3, variable = 'zeta_max'):
    ''' Interpolate adcirc results from the 3 nodes that forms the triangle in which
        a point lies in. Only work for time-constant netcdf files.
        Parameters
            ncObj: etCDF4._netCDF4.Dataset
                Adcirc input file
            pnts: list
                list with ziped coordinates. Eg. [(x0, y0), (x1, y1), ....]
            n: int, default 3
                number of triangles considered to look in which one the point
                is contained.
        Returns
            lznew: list
                list of interpolated results
    '''
    ## triangles
    nv = ncObj['element'][:,:] - 1 ## triangles starts from 1
    ## x and y coordinates
    x = ncObj['x'][:].data
    y = ncObj['y'][:].data
    ## matplotlib triangulation
    tri = mpl.tri.Triangulation(x, y, nv)
    ## get the x and y coordinate of the triangle elements in the right order
    xvertices = x[tri.triangles[:]]
    yvertices = y[tri.triangles[:]]
    ## add x and y togheter
    listElem = np.stack((xvertices, yvertices), axis = 2)
    ## vertex number of each node
    v1 = nv.data[:, 0]
    v2 = nv.data[:, 1]
    v3 = nv.data[:, 2]
    v = np.array((v1, v2, v3)).T  
    ## get centroids
    centx = xvertices.mean(axis = 1)
    centy = yvertices.mean(axis = 1)
    ## compute distance from all centroids to the requested points
    mdist = cdist(list(zip(centx, centy)), pnts)
    ## iterate through each point to find in what triangle is contained
    lnewz = []
    for i in range(len(pnts)):
        ## selected only the n closest centroids
        a = np.where(mdist[:, i] < sorted(mdist[:, i])[n])[0]
        for ni in range(n):
            ## define polygon
            pol = Polygon(listElem[a[ni], :, :])
            ## find the polygon that contains the point
            if pol.contains(Point(pnts[i])):
                vs = v[a[ni]]
                break
        
        x = ncObj['x'][vs].data
        y = ncObj['y'][vs].data 
        ## variable to interpolate
        z = ncObj[variable][vs].data
        ## masked values to 0
        z[z < -1000] = np.nan
        ## define interpolation function
        f = interpolate.LinearNDInterpolator(list(zip(x, y)), z)
        ## interpolate
        newz = float(f(pnts[i][0], pnts[i][1]))
        lnewz.append(newz)
        
    return lnewz
    
def closestPointIndex2D(xarray, yarray, xp, yp):
    ''' Find index of closest point inside an array
        of the requested points
        Parameters
            xarray, yarray: numpy arrays
                coordinates of the array where to look
            xp, yp: numpy arrays or floats
                coordinates of the requested point(s) to
                look for.
        Return
            mdi:int
                index of the closest point(s)
        
    '''
    arr = np.array(list(zip(xarray, yarray)))
    p = np.array([xp, yp])
    p = np.reshape(p, (1, len(p)))
    dist = cdist(arr, p)
    mdi = np.argmin(dist)
    return mdi

def curvature(x, y):
    ''' Curvature is a measure of deviance of a curve from being a straight line.
        Based on: https://www.delftstack.com/howto/numpy/curvature-formula-numpy
        Parameters
            x, y: array
                coordinates
        Returns
            curvature_val: array
                values of the curvature
    '''
#     coordinates = zip(x, y)
    x_t = np.gradient(x)
    y_t = np.gradient(y)
    vel = list(zip(x_t, y_t))
    speed = np.sqrt(x_t **2 + y_t ** 2)
    
    ss_t = np.gradient(speed)
    xx_t = np.gradient(x_t)
    yy_t = np.gradient(y_t)

    curvature_val = np.abs(xx_t * y_t - x_t * yy_t) / (x_t * x_t + y_t * y_t)**1.5
    
    return curvature_val
    
def tsFromNC(ncObj, pnts, n = 3, variable = 'zeta'):
    ''' Interpolate adcirc results from the 3 nodes that forms the triangle in which 
        a point lies in for all timesteps
        Parameters
            ncObj: etCDF4._netCDF4.Dataset
                Adcirc input file
            pnts: list
                list with ziped coordinates. Eg. [(x0, y0), (x1, y1), ....]
            n: int, default 3
                number of triangles considered to look in which one the point
                is contained.
        Returns
            dfout: pandas dataframe
                df with of interpolated results
    '''
    ## triangles
    nv = ncObj['element'][:,:] - 1 ## triangles starts from 1
    ## x and y coordinates
    x = ncObj['x'][:].data
    y = ncObj['y'][:].data
    ## matplotlib triangulation
    tri = mpl.tri.Triangulation(x, y, nv)
    ## get the x and y coordinate of the triangle elements in the right order
    xvertices = x[tri.triangles[:]]
    yvertices = y[tri.triangles[:]]
    ## add x and y togheter
    listElem = np.stack((xvertices, yvertices), axis = 2)
    ## vertex number of each node
    v1 = nv.data[:, 0]
    v2 = nv.data[:, 1]
    v3 = nv.data[:, 2]
    v = np.array((v1, v2, v3)).T  
    ## get centroids
    centx = xvertices.mean(axis = 1)
    centy = yvertices.mean(axis = 1)
    ## compute distance from all centroids to the requested points
    mdist = cdist(list(zip(centx, centy)), pnts)
    ## iterate through each point to find in what triangle is contained
    t0 = pd.to_datetime(ncObj['time'].units.split('since ')[1])
    dates = [t0 + pd.Timedelta(seconds = float(x)) for x in ncObj['time'][:]]
    dfout = pd.DataFrame(columns = [f'Pnt{x:03d}' for x in range(len(pnts))], index = dates)
    
    for i in tqdm(range(len(pnts))):
        a = np.where(mdist[:, i] < sorted(mdist[:, i])[n])[0]
        for ni in range(n):
            lnewzti = []
            ## define polygon
            pol = Polygon(listElem[a[ni], :, :])
            ## find the polygon that contains the point
            if pol.contains(Point(pnts[i])):
                vs = v[a[ni]]
                break

        x = ncObj['x'][vs].data
        y = ncObj['y'][vs].data 
        ## variable to interpolate
        z = ncObj[variable][:, vs].data
        for zi in z:
            f = interpolate.LinearNDInterpolator(list(zip(x, y)), zi)
            newz = float(f(pnts[i][0], pnts[i][1]))
            lnewzti.append(newz)
        dfout[f'Pnt{i:03d}'] = lnewzti
        dfout = dfout.replace(-99999.000000, np.nan)
    
    return dfout
