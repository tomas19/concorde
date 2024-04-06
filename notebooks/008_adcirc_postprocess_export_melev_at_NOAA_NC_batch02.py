# %%
import os
import numpy as np
import netCDF4 as netcdf
import pandas as pd
from concorde.tools import get_list
from tqdm import tqdm
from concorde.tools import tsFromNC

# %%
pathin = r'/media/tacuevas/Extreme SSD/batch02'
files = sorted(get_list(pathin, ends='maxele.63.nc'))

ys = [36.183, 35.795, 35.209, 34.717, 34.227, 34.213]
xs = [-75.745, -75.548, -75.704, -76.67, -77.953, -77.787]
names = ['Duck', 'Oregon', 'Hatteras', 'Beaufort', 'Wilmington', 'Wrightsville']

maxelev = []
for fi, f in tqdm(enumerate(files)):
    melev = netcdf.Dataset(f)
    ts, rep = tsFromNC(melev, list(zip(xs, ys)), n = 5, variable = 'zeta_max',
                         closestIfDry = True)
    #ts, rep = tsFromNC(melev, list(zip(xs, ys)), n = 5, variable = 'time_of_zeta_max',
    #                    closestIfDry = True)
    ts.index = [os.path.dirname(f).split('/')[-1]]
    maxelev.append(ts)
    
dfout = pd.concat(maxelev, axis = 0)
dfout.columns = names
dfout.to_csv(r'/mnt/drive1/Insyncs/NCSU/thesis/models/adcirc/concorde/batch02/_postprocessing/max_water_level_at_NC_NOAA_stations.pkl')
