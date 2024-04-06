# %%
import os
import geopandas as gpd
import netCDF4 as netcdf
import pandas as pd
from concorde.tools import get_list
from tqdm import tqdm
from kalpana.export import pointsInsidePoly

# %%
# pathin = Path(r'/mnt/drive1/Insyncs/NCSU/thesis/models/adcirc/concorde_NC9/_max_results')
pathin = r'/media/tacuevas/Extreme SSD/batch02'
files = sorted(get_list(pathin, ends='maxele.63.nc'))

# %%
bbox_file = r'../gis/gpkg/bounding_box_max_flooding.gpkg'
bbox = gpd.read_file(bbox_file)

# %%
bbox = list(bbox.geometry.iloc[0].boundary.coords)

# %%
maxelev = []
runs = []
for fi, f in tqdm(enumerate(files)):
    melev = netcdf.Dataset(f)
    if fi == 0:
        points = list(zip(melev['x'][:].data, melev['y'][:].data))
        bo = pointsInsidePoly(points, bbox)
    
    df = pd.DataFrame({'x': melev['x'][:].data[bo.nonzero()[0]],
                        'y': melev['y'][:].data[bo.nonzero()[0]],
                        'zeta_max': melev['zeta_max'][:].data[bo.nonzero()[0]]})
    
    maxelev.append(df.loc[df['zeta_max'].idxmax(), :].values)
    runs.append(os.path.dirname(f).split('/')[-1])

dfout = pd.DataFrame(index = runs, data = maxelev, columns = ['x', 'y', 'zeta_max'])
dfout.to_pickle(r'../models/adcirc/concorde/batch02/_postprocessing/max_water_level_NC.pkl')
