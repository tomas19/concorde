# %%
import os
import pickle
import netCDF4 as netcdf
from concorde.tools import get_list
from tqdm import tqdm
from concorde.tools import tsFromNC

# %%
pathin = r'/media/tacuevas/Extreme SSD/batch02'
files = sorted(get_list(pathin, ends='fort.63.nc'))

ys = [36.183, 35.795, 35.209, 34.717, 34.227, 34.213]
xs = [-75.745, -75.549, -75.704, -76.67, -77.953, -77.787]
names = ['Duck', 'Oregon', 'Hatteras', 'Beaufort', 'Wilmington', 'Wrightsville']

dct = {}
for fi, f in tqdm(enumerate(files[:-1])):
    f63 = netcdf.Dataset(f)
    r = os.path.dirname(f).split('/')[-1]
    ts, rep = tsFromNC(f63, list(zip(xs, ys)), n = 5, variable = 'zeta',
                         closestIfDry = True)
    print(type(r))
    print(type(ts))
    dct[r] = ts 
    
with open(r'../models/adcirc/concorde/batch02/_postprocessing/time_series_water_level_at_NOAA_NC_closest.pkl', 'wb') as handle:
    pickle.dump(dct, handle, protocol=pickle.HIGHEST_PROTOCOL)
# %%
