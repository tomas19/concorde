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
from itertools import islice
from sklearn.neighbors import KDTree
import tarfile

def animate_list_imgs(imgs, fps, anim_name):
    ''' Function to animate list of images.
        Parameters
            list_imgs: list
                list with complete path of all imgs
            fps: int
                frames per second
            anim_name: str
                output name, only name without path. The video will be saved in the notebook path
        Returns
            None
    '''
    im = cv2.imread(imgs[0])
    h, w, l = im.shape
    size = (w, h)
    
    if anim_name.endswith('.avi'):
        out = cv2.VideoWriter(anim_name, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    elif anim_name.endswith('.mp4'):
        out = cv2.VideoWriter(anim_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    else:
        print('Only avi and mp4 extensions are supported')
        sys.exit(-1)
    
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
    
def tsFromNC(ncObj, pnts, n=5, variable='zeta', extractOut=False, closestIfDry=False):
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
            rep: list
                strings with information about how the data was extracted
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

    z[z < -9999] = np.nan
    ## loop through points
    rep = []
    # for i in tqdm(range(len(pnts))):
    for i in range(len(pnts)):
        # print(pnts[i])
        ## get the nearest n centroid to point i
        # print(mdist.shape)
        a = np.where(mdist[:, i] < sorted(mdist[:, i])[n])[0]
        #print(a)
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
            xs = ncObj['x'][vs].data
            ys = ncObj['y'][vs].data 
            ## variable to interpolate
            zs = z[:, vs]

            ## 3 nodes are dry for the entire simulation, extract values from the closest node with any wet time step
            if np.isnan(zs).all() and closestIfDry == True:
                mdist2 = cdist(list(zip(x, y)), np.reshape(pnts[i], (1, 2)))
                mdist2 = np.concatenate((mdist2, np.arange(len(mdist2)).reshape((-1, 1))), axis = 1)
                mdist2_sorted = np.take(mdist2, np.argsort(mdist2[:, 0]), axis = 0)

                for k in mdist2_sorted[:, 1]:
                    zsk = z[:, int(k)]
                    if np.isnan(zsk).all() == False:
                        newz = zsk
                        rep.append(f"Point {i:03d} is inside the domain! All element's nodes are dry, so data from the closest wet node ({int(k)}) was extracted.")
                        break
                try:
                    lnewzti.extend(newz)
                except:
                    lnewzti.append(newz)

            elif np.isnan(zs).any() and closestIfDry == True:
                ## get values from the closest node with wet time steps from the same element
                
                mdist2 = cdist(list(zip(x[v[a[ni]]], y[v[a[ni]]])), np.reshape(pnts[i], (1, 2)))
                mdist2 = np.concatenate((mdist2, np.arange(len(mdist2)).reshape((-1, 1))), axis = 1)
                mdist2_sorted = np.take(mdist2, np.argsort(mdist2[:, 0]), axis = 0)

                for k in mdist2_sorted[:, 1]:
                    zsk = zs[:, int(k)]
                    if np.isnan(zsk).all() == False:
                        newz = zsk
                        rep.append(f"Point {i:03d} is inside the domain! There are dry nodes in the element, so data from the closest wet node ({v[a[int(k)]]}) in the element was extracted.")
                        break
                try:
                    lnewzti.extend(newz)
                except:
                    lnewzti.append(newz)
            else:
                rep.append(f"Point {i:03d} is inside the domain! data was interpolated using the corresponding element's nodes.")
                ## all nodes are wet or, if one node of the element is dry output will be nan
                for zi in zs:
                    f = interpolate.LinearNDInterpolator(list(zip(xs, ys)), zi)
                    newz = float(f(pnts[i][0], pnts[i][1]))
                    lnewzti.append(newz)
                    
            dfout[f'{variable}_pnt{i:03d}'] = lnewzti
            del vs
                
        else:
            ## point is outside the domain
            if extractOut == True:
                ## find nearest node to the requested point
                mdist2 = cdist(list(zip(x[v[a[0]]], y[v[a[0]]])), np.reshape(pnts[i], (1, 2)))
                clnode = mdist2.argmin()
                rep.append(f'Point {i:03d} is outside the domain! data from  node {clnode} (closest) was exported.')
                newz = z[:, v[a[0]][clnode]]
                lnewzti = newz.copy()
                dfout[f'{variable}_pnt{i:03d}'] = lnewzti
            else:
                rep.append(f'Point {i:03d} is outside the domain! Returning nan.')
                dfout[f'{variable}_pnt{i:03d}'] = np.nan
    
    return dfout, rep
    
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

def checkAdcircLog(run, mtype = 'padcirc'):
    ''' Read padcirc.XXXXX file to find the run time and the MPI status. 
        The last edited file will be analyzed. TODO: add error if the only log file
        doesn't have date info
        Parameters
            run: str
                complete path of the adcirc run
            mtype: str
                padcirc or padcswan
        Returns
            dt: float
                number of hours the run took
            status: boolean
                0 if finished correctly, 1 if crashed, 2 if time limit
                reached and 'Still running'
            log_out: string
                name of the log file check, usefull in the case there are multiple files
    '''
    def getStartDate(lines2):
        for line in lines2:
            if line.startswith('Started at'):
                startline = line.split()
                stime = startline[-2].split(':')
                sdate = datetime.datetime(int(startline[-1]), 1 + int(months.index(startline[3])), 
                            int(startline[4]), int(stime[0]), int(stime[1]), int(stime[2]))
        try:
            return sdate
        except NameError:
            return datetime.datetime(1900, 1, 1, 0, 0)
    
    run  = Path(run)
    months = list(calendar.month_abbr)[1:]
    ## check if there are log files, only if run is a folder
    if os.path.isdir(run):
        logs = [run/x for x in os.listdir(run) if x.startswith(f'{mtype}.') and 
                '.csh' not in x and '.sh' not in x]
        if len(logs) == 0:
            dt = 'empty'
            status = 'not ran'
            log_out = 'no log'
        else:
            sdates = []
            for log in logs:
                sdates.append(os.path.getmtime(log))
            last_sdate = max(sdates)
            last_log = logs[np.argmax(sdates)]
                
            with open(last_log, 'r') as fin:
                erroraux = 0
                lines = fin.readlines()
                last_sdate = getStartDate(lines)
                log_out = last_log
    else:
        with tarfile.open(run, 'r') as tar:
            logs = [x for x in tar.getmembers() if os.path.basename(x.name).startswith(f'{mtype}.') and 
                    '.csh' not in os.path.basename(x.name) and 
                    '.sh' not in os.path.basename(x.name)]
            sdates = []
            for log in logs:
                lines = tar.extractfile(log).read()
                lines = lines.decode('utf-8').split('\n')
                sd = getStartDate(lines)
                sdates.append(sd)

            last_sdate = sorted(sdates)[-1]
            last_sdate_index = sdates.index(last_sdate)
            last_log = logs[last_sdate_index]
            lines = tar.extractfile(last_log).read()
            lines = lines.decode('utf-8').split('\n')
            erroraux = 0
            log_out = last_log.name
    
    if 'lines' in locals() and isinstance(lines, list):
        startWarning = 0
        for iline, line in enumerate(lines):
            if line.startswith('Terminated at'):
                endline = line.split()
                etime = endline[-2].split(':')
                edate = datetime.datetime(int(endline[-1]), 1 + int(months.index(endline[3])), int(endline[4]), 
                          int(etime[0]), int(etime[1]), int(etime[2]))
            elif '100.00% COMPLETE' in line:
                status = 'complete'
            elif line.startswith('User defined signal 2'):
                status = 'Time limit reached'
            elif line.startswith('=   EXIT CODE:'):
                status = line[4:-1]
                erroraux = 1
            elif line.startswith('  ** ERROR: Elevation.gt.ErrorElev, ADCIRC stopping. **'):
                status = 'ADCIRC blow-up'
                erroraux = 1
            elif '** WARNING: Elevation.gt.WarnElev **' in line and startWarning == 0:
                nodeElevWarning = int(line.split()[5])
                nodeSpeedWarning = int(line.split()[11])
                perWarning = lines[iline - 3].split('COMPLETE')[0].split()[-1]
                startWarning = 1
            elif line.startswith('forrtl: No space left on device'):
                status = 'No space left on device'
                erroraux = 1
            elif line.startswith("INFO: openFileForRead: The file './fort.22' was not found."):
                status = 'fort.22 not found'
                erroraux = 1
            elif 'ADCIRC terminating' in line:
                status = 'Run failed'
                erroraux = 1
            else:
                pass
        if line[-1].startswith(' TIME STEP') or line.startswith('  ELMAX'):
            status = 'Still running '
        try:
            dt = (edate - last_sdate).total_seconds()/3600
        except NameError:
            dt = 0
        try:
            nodeElevWarning
        except NameError:
            nodeElevWarning = None
            nodeSpeedWarning = None
            perWarning = None
        try:
            status
        except NameError:
            status = 'Error no catched, check log manually' 
        
        return dt, status, os.path.basename(log_out), nodeElevWarning, nodeSpeedWarning, perWarning
    else:
        return 'noLogFile', 'noLogFile', 'noLogFile', 'noLogFile', 'noLogFile', 'noLogFile'

def NNfort13(fort14_old, fort14_new, fort13_old, fort13_new, attrs):
    ''' Function to interpolate the fort.13 from one mesh to another using
        nearest neighbor
        Parameters
            fort14_old: str
                full path of the original fort.14
            fort13_old: str
                full path of the original fort.13
            fort14_new: str
                full path of the new fort.14
            fort13_new: str
                full path of the new fort.13, a new file will be created
            attrs: dictionary
                attributes to consider in the new fort.13
                Currently the keys of the are **the exact same name** of the attributes, be careful
                with empty spaces (this will be fixed soon, WIP).
                The items are integers with the number of lines per attribute in the hader information.
                (WIP: As far as I know, all attrs always have three lines In case this is always
                true this input needs to be changed to a list). 
                E.g.
                attrs = {
                         'surface_directional_effective_roughness_length': 3,
                         'surface_canopy_coefficient': 3,
                         'mannings_n_at_sea_floor': 3,
                         'primitive_weighting_in_continuity_equation': 3,
                         'average_horizontal_eddy_viscosity_in_sea_water_wrt_depth': 3,
                         'elemental_slope_limiter': 3
                        }
        Return
            None
    '''
    ## open old fort.14 to get number of nodes and elements
    with open(fort14_old) as fin:
        head_old = list(islice(fin, 2))
        data_old = [int(x) for x in head_old[1].split()]
    ## read only nodes as an array of x,y
    nodes_old = np.loadtxt(fort14_old, skiprows = 2, max_rows = data_old[1], 
                           usecols = (1, 2))
    
    ## idem for new fort.14
    with open(fort14_new) as fin:
        head_new = list(islice(fin, 2))
        data_new = [int(x) for x in head_new[1].split()]
    nodes_new = np.loadtxt(fort14_new, skiprows = 2, max_rows = data_new[1], 
                           usecols = (1, 2))
    
    ## nearest neighbor interpolation --> the closest node of the new mesh is
    ## assign to each of the nodes of the old mesh.
    tree = KDTree(nodes_old)
    dist, ind = tree.query(nodes_new)
    ## dataframe with new nodes and the closest old node assigned to each one
    dfnew = pd.DataFrame({'x': nodes_new[:, 0], 'y': nodes_new[:, 1], 'old_id':ind.reshape(-1)})
    
    ## open the old fort.13 in read mode to read the data
    with open(fort13_old, 'r') as fin:
        ## open the new fort.13 in writing mode to dump the interpolated information
        with open(fort13_new, 'w') as fout:
            ## write header in to the new fort.13: titile, number of nodes and number of
            ## attributes.
            fout.write(f'Spatial attributes descrption. File generated with NNfort13 on {datetime.date.today()} using {fort14_old} and {fort13_old} as basefiles, and {fort14_new} as the new mesh\n')
            fout.write(f'{data_new[1]}\n')
            fout.write(f'{len(attrs.keys())}\n')## write
            
            ## Inside this for loop we are writing the name, default value and other
            ## parameters of each of the selected attributes
            lines = fin.readlines()[3:]
            for key in attrs.keys():
                ## index of the attribute
                ind = lines.index(key+'\n')
                ## write default values and extra info per attribute
                ## lines from index of attr to the int in the item of that specific
                ## key es written.
                fout.writelines(lines[ind:ind+1+attrs[key]])
            
            ## From this line the value of the attrs for each node es written.
            for key in tqdm(attrs.keys()):
                ## get index of 1st and 2nd time the attr key appears in the file
                inds = [i for i, n in enumerate(lines) if n == key+'\n'][:2]
                try:
                    ## read default value, try and except block is due to the 
                    ## surface_directional_effective_roughness_length, which is a typically
                    ## a list of 12 values
                    defval = lines[inds[0]+3].split()
                    ## convert from str to float
                    defval = [float(x) for x in defval]
                except:
                    ## in case the values are not a list
                    defval = [float(lines[inds[0]+3][:-1])]
                
                ## index where the values will be dumped
                indi = inds[1] + 2
                ## number of nodes with non default value
                nnondef = int(lines[int(inds[1] + 1)][:-1])
                ## index where the nodes of the attr finish
                indf = indi + nnondef - 1
                ## read data only if there are more than 0 non default vertices
                if nnondef > 0:
                    ## read the lines between previous defined indices as dataframes
                    olds = pd.read_csv(fort13_old, skiprows = indi + 3, nrows = indf - indi + 1, header = None, sep = ' ', index_col = 0)
                    ## problem when first column of the org fort 13 has whitespaces, the index is nan and an extra column
                    ## with the vertex id is added to the dataframe
                    if np.isnan(olds.index).all() == True:
                        olds.index = olds.iloc[:, 0]
                        olds = olds.drop([1], axis = 1)
                        olds.columns = range(1, len(olds.columns) + 1)
                    else:
                        pass
                    ## not sure why this dataframe is not writable: olds_all_aux.values.flags will show the array is not writable. Fixed with copy
                    ## array for store the value of all nodes, not only the non-default
                    olds_all_aux = pd.DataFrame(columns = olds.columns, index = range(1, data_old[1] + 1),
                                                data = np.broadcast_to(np.array(defval), (data_old[1], len(defval))))
                    olds_all = olds_all_aux.copy()
                    ## add info of nodes with default value
                    olds_all.loc[olds.index, :] = olds.values
                    ## dataframe with attr values for the nodes of the new mesh
                    ## this is done selecting the data of the old data for the closest old
                    ## node associated to each of the new nodes
                    news_all = olds_all.loc[dfnew['old_id'] + 1, :]
                    news_all.index = range(1, len(news_all) + 1)
                    ## get the nodes with default value
                    dfdef = news_all[news_all == defval].dropna()
                    ## get nodes with non-default value
                    dfnondef = news_all[news_all != defval].dropna()
                    dfnondef = dfnondef.sort_index()
                    dfnondef['dummy'] = '\n'
                    ## write attribute name
                    fout.write(key + '\n')
                    ## write number of non default nodes
                    fout.write(str(len(dfnondef)) + '\n')
                    ## format the data to write it to the new fort.13 file
                    dfaux = pd.DataFrame({'x': dfnondef.index}, index = dfnondef.index)
                    new_lines = pd.concat([dfaux, dfnondef], axis = 1)
                    new_lines2 = [' '.join([str(x) for x in new_lines.loc[i, :]]) for i in new_lines.index]
                    fout.writelines(new_lines2)
                else:
                    ## write attr with only default values
                    fout.write(key + '\n')
                    fout.write('0\n')

def from_mag_to_uv(vel, direction,meteo=True):
    ''' Function to transform velocity direction and magnitude into directional (vectorial) velocities U,V. 

    Parameters:
        vel: value, series or array 
            Velocity magnitude (can be wind, waves, currents, etc)
        dir:  value, series or array 
            Velocity direction (can be wind, waves, currents, etc)
        meteo: boolean (optional)
            Value to indicate if meteorological convention (True) or non-meteorological convention (False) is wanted for directional (vectorial) output. 
                (Default) Meteorological convention is the direction where the wind is blowing from, or where waves come from. 
                Non-meteorological convention is the direction where the currents are going to.

    Returns:
        u,v: tuple
            Tuple with values (can be an array or pandas series) of  velocities in x and y axis (u,v). This is given in the form (u1,u2,u3...uN),(v1,v2,v3...vN)
        
    '''
    # direction=np.asarray(direction) #Change values to arrays
    # vel=np.asarray(vel) #Change values to arrays
    
    if meteo==False:
        direction=direction+180
    else:
        pass    
    
    if len(vel)==1:
        u = -1*np.abs(vel)*np.sin(np.pi*np.mod(direction, 360)/180)
        v = -1*np.abs(vel)*np.cos(np.pi*np.mod(direction, 360)/180)
    else:
        u = -1*np.abs(list(vel))*np.sin(np.pi*np.mod(list(direction), 360)/180)
        v = -1*np.abs(list(vel))*np.cos(np.pi*np.mod(list(direction), 360)/180)
    return u,v
       
def from_uv_to_mag(u,v,meteo=True):
    ''' Function to transform directional velocities U,V into velocity direction and magnitude. 

    Parameters:
        u: float, int, list, series or array 
             value or values of X velocities
        v: float, in, series or array 
             value or values of X velocities
        meteo: boolean (optional)
            Value to indicate if meteorological convention (True) or non-meteorological convention (False) is wanted for directional output. 
                (Default) Meteorological convention is the direction where the wind is blowing from, or where waves come from. 
                Non-meteorological convention is the direction where the currents are going to.

    Returns:
        vel,d: tuple
            Tuple with values (can be an int, float, array o pandas series) of  velocities (magnitudes) and directions
        
    '''
    vel = (u**2+v**2)**0.5
    d = np.arctan2(u/vel, v/vel)*180/np.pi 
    if meteo==True:
        d = np.mod(d+180, 360)
    else:
        d = np.mod(d,360)
    return vel, d
    
def ssScale(x):
    ''' Saffir-Simpson scale in meters per second
    '''
    if x < 33:
        cat = 0
    elif 33 <= x < 43:
        cat = 1
    elif 43 <= x < 49:
        cat = 2
    elif 49 <= x < 58:
        cat = 3
    elif 58 <= x < 70:
        cat = 4
    else:
        cat = 5

    return cat
