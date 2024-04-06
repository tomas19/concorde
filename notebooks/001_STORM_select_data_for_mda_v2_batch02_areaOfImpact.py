import pandas as pd
import geopandas as gpd
import dask
from tqdm.dask import TqdmCallback
import numpy as np
from shapely.geometry import Point
import warnings
warnings.filterwarnings("ignore")

dfs = pd.read_pickle(r'../data/STORM/processed/batch02/STORM_NA_R5_v1.pkl')

def radImpact(vmax, rmax, vr = 33, beta = 1):
    r1 = rmax
    while True:
        r = rmax * (vr / vmax)**(-2 / beta) * (np.exp(1 - (rmax / r1)**beta))**(1 / beta)
        # print(f'r ini: {r1:0.5f} - r fin: {r:0.5f}')
        if np.abs(r - r1) < 1e-5:
            break
        r1 = r
    if r == 0:
        return rmax
    else:
        return r
    
@dask.delayed
def radImpactAll(df, i):
    dftrack = df[df.tc_number == i]
    ri = [radImpact(dftrack.loc[i, 'max_ws'], dftrack.loc[i, 'rad_to_max_ws']) for i in dftrack.index]
    pols = []
    for i, r in zip(dftrack.index, ri):
        pnt = Point((dftrack.loc[i, 'lon'], dftrack.loc[i, 'lat']))
        pol = pnt.buffer(r/110)
        pols.append(pol)
    gdfPol = gpd.GeoDataFrame(geometry=pols, crs = 4326)
    gdfPol2 = gpd.GeoDataFrame(geometry = [gdfPol.unary_union], crs = 4326)
    
    return gdfPol2

tasks = [radImpactAll(dfs, i) for i in dfs['tc_number'].unique()]

with TqdmCallback(desc = "Computing distance"):
    aux = dask.compute(tasks, scheduler = 'threads')

gdfAoi = pd.concat(aux[0])
gdfAoi.to_file(r'../data/STORM/processed/batch02/STORM_NA_R5_area_of_impact.gpkg')
