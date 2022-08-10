import os
import matplotlib as mpl
import numpy as np
import geopandas as gpd
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon, Point
from scipy import interpolate
from tqdm import tqdm
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

def interpOut(ncObj, pnts, n = 3):
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
        z = ncObj['zeta_max'][vs].data
        ## masked values to 0
        z[z < -1000] = np.nan
        ## define interpolation function
        f = interpolate.LinearNDInterpolator(list(zip(x, y)), z)
        ## interpolate
        newz = float(f(pnts[i][0], pnts[i][1]))    
        lnewz.append(newz)
        
    return lnewz
