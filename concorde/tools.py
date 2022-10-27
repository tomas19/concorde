import os
import sys
import netCDF4 as netcdf
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import geopandas as gpd
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon, Point
from scipy import interpolate
from tqdm import tqdm
import pandas as pd
import calendar
import datetime
from pathlib import Path
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

def checkTimeVarying(ncObj):
    ''' Check if an adcirc input is time-varying or not.
        Parameters
            ncObj: netCDF4._netCDF4.Dataset
                Adcirc input file
        Returns
            timeVar: int
                1 if time-varying, 0 if not
    '''
    if ncObj['time'].shape[0] <= 1:
        ## not time-varying
        timeVar = 0
    elif (ncObj['time'][-1].data - ncObj['time'][0].data).astype(int) == 0:
        ## time variable has lenght 2 but dates are the same --> not time-varying
        timeVar = 0
    else:
        ## time-varying file
        timeVar = 1
    
    return timeVar
    
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
    
def tsFromNC(ncObj, pnts, n = 3, variable = 'zeta', extractOut = False):
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
            extractOut: boolean
                True for extract data if the points are outside the domain. Defalt False,
                nan will be returned in this case.
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
    dfout = pd.DataFrame(columns = [f'{variable}_pnt{x:03d}' for x in range(len(pnts))], index = dates)
    ## check if nc file is time-varying
    tvar = checkTimeVarying(ncObj)
    if tvar == 1:
        z = ncObj[variable][:, :].data
    else:
        ## reshape to add an extra dimension
        z = ncObj[variable][:].data.reshape((1, ncObj[variable].size))
    ## loop through points
    for i in range(len(pnts)):
        ## get the n centroid nearest to the point i
        a = np.where(mdist[:, i] < sorted(mdist[:, i])[n])[0]
        ## iterate through each element to see is the point is inside
        for ni in range(n):
            lnewzti = []
            ## define polygon
            pol = Polygon(listElem[a[ni], :, :])
            ## find the polygon that contains the point
            if pol.contains(Point(pnts[i])):
                vs = v[a[ni]]
                break
        ## point is inside the mesh
        if 'vs' in locals():
            print(f'Point {i:03d} is inside the domain! data was interpolated.')
            xs = ncObj['x'][vs].data
            ys = ncObj['y'][vs].data 
            ## variable to interpolate
            zs = z[:, vs]
            for zi in zs:
                f = interpolate.LinearNDInterpolator(list(zip(xs, ys)), zi)
                newz = float(f(pnts[i][0], pnts[i][1]))
                lnewzti.append(newz)
            dfout[f'{variable}_pnt{i:03d}'] = lnewzti
            del vs
            
        else:
            ## point is outside the domain
            if extractOut == True:
                print(f'Point {i:03d} is outside the domain! data from nearest node was exported.')
                ## find nearest node to the requested point
                mdist2 = cdist(list(zip(x[v[a[0]]], y[v[a[0]]])), np.reshape(pnts[i], (1, 2)))
                clnode = mdist2.argmin()
                newz = z[:, v[a[0]][clnode]]
                lnewzti = newz.copy()
                dfout[f'{variable}_pnt{i:03d}'] = lnewzti
            else:
                print(f'Point {i:03d} is outside the domain! Returning nan.')
                dfout[f'{variable}_pnt{i:03d}'] = np.nan
        
    dfout = dfout.replace(-99999.000000, np.nan)
    
    return dfout
    
def ascii_replace(filein, fileout, olds, news):
    ''' Writes a new ascii file replacing fields from a template file
    
        Parameters
            filein: str
                path of the template file (path + name)
            fileout: str
                path of the new ascii file (path + name)
            olds: list of strings
                list with strings to replace
            news: list of strings
                list with the new strings which will be written
                
        Returns
            None                
    '''
    if len(olds) != len(news):
        sys.exit("The lists 'olds' and 'news' must have the same length")
    newlines = []
    index = []
    with open(filein, 'r') as fin:
        lines = fin.readlines()
    
    idx = 0
    for line in lines:
        replace = False
        for old, new in zip(olds, news):
            if old in line:
                replace = True
                line = line.replace(old, new)
        if replace==True:
            index.append(idx)
            newlines.append(line)
        idx = idx+1
        
    if len(newlines) == 0:
        pass
    else:
        for i in range(len(newlines)):
            lines[index[i]] = newlines[i]
        with open(fileout, 'w') as fout:
            fout.writelines(lines)

def checkAcircLog(run, mtype = 'padcirc'):
    ''' Read padcirc.XXXXX file to find the run time and the MPI status. 
        The last edited file will be analyzed.
        Parameters
            run: str
                complete path of the adcirc run
            mtype: str
                padcirc or padcswan
        Returns
            dt: float
                number of hours the run took
            status: boolean
                0 if finished correctly, 1 if not
    '''
    run  = Path(run)
    months = list(calendar.month_abbr)[1:]
    logs = [run/x for x in os.listdir(run) if x.startswith(f'{mtype}.') and '.csh' not in x]
    if len(logs) == 0:
        dt = 'empty'
        status = 'not run'
                                                           
    ## sort by modification date
    else:
        logs.sort(key = lambda x: os.path.getmtime(x))
        last_log = logs[-1]
        with open(run/last_log, 'r') as fin:
            lines = fin.readlines()
            for line in lines:
                if line.startswith('Started at'):
                    startline = line.split()
                    stime = startline[-2].split(':')
                    sdate = datetime.datetime(int(startline[-1]), int(months.index(startline[3])), int(startline[4]), 
                              int(stime[0]), int(stime[1]), int(stime[2]))
                elif line.startswith('Terminated at'):
                    endline = line.split()
                    etime = endline[-2].split(':')
                    edate = datetime.datetime(int(endline[-1]), int(months.index(endline[3])), int(endline[4]), 
                              int(etime[0]), int(etime[1]), int(etime[2]))
                elif line.startswith(' MPI terminated with Status = '):
                    statusline = line.split()
                    if statusline[-1] == '0':
                        status = 0
                    else:
                        status = 1
                else:
                    pass
        dt = (edate - sdate).total_seconds()/3600
    
    return dt, status
