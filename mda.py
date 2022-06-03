import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull, Voronoi, voronoi_plot_2d, cKDTree
from tqdm import tqdm

def fnorm(vec, maxi = 1, mini = 0):
    ''' function to normalize a vector regarding the maximum value.
  
        Parameters
            vec: array
                values to normalize
            maxi: float (default 1)
                max value of the normalized vector
            mini: float (default 0)
                min value of the normalized vector
  
        Returns
            norm: array
                normalized vector
    '''
    norm = (vec-np.min(vec))/(np.max(vec)-np.min(vec))*(maxi-mini)+mini
    return norm

def fnorm_inv(vec, norm, maxi = 1, mini = 0):
    ''' Function to undo a normalization regarding the max value of the original values. Inverse function of "fnorm".
  
        Parameters
            vec: array
                orginal vector
            norm: array
                normalized vector
            maxi: float (default 1)
                max value of the normalized vector
            mini: float (default 0)
                min value of the normalized vector
  
        Returns
            vec2: array
                vector "unnormalize"
    '''
    vec2 = (norm-mini)*(np.max(vec)-np.min(vec))/(maxi-mini)+np.min(vec)
    return vec2
  
def ecdist_mda(vec0, vec1, index):
    ''' Computes euclidean distance between two normalized tuples regarding Camus et al. 2011 formulation (MDA methodology)
  
        Parameters
            vec0: tuple
                It doesn't matter the tuples dimension, it could be Nx1.
            vec1: tuple
                It doesn't matter the tuples dimension, it could be Nx1.
            index: list
                list with binary values. The list length must be equals to the number of parameters 
                in the cluster analysis. If the value is equal to 1, the distance of the tuple elements
                is calculated as (x-y)**2. And if the value is equal to 0, the distance
                is computed as (min(|x-y|, 2-|x-y|))**2. This formula is only applied for directions.
                For summarize, index (input list) must have 1 at index where dataframe columns 
                aren't directions and 0 where the columns are directions.
  
        Returns
            float: distance
    '''   
    listpow = []
    for i, j in zip(index, range(len(index))):
        if i == 1:
            listpow.append((vec0[j]-vec1[j])**2)
        else:
            listpow.append((min(np.abs(vec0[j]-vec1[j]), 2-np.abs(vec0[j]-vec1[j])))**2)
    return np.sum(listpow)**0.5

def normalize_df(df, index, maxi = 1, mini = 0, dirnorm = 180):
    '''Apply fnorm function (in TOOLS package) to a pandas dataframe.
  
        Parameters
            df: dataframe
                input data
            index: list
                list with binary values. The list length must be equals to the number of parameters 
                in the cluster analysis. If the value is equal to 1, the normalization is computed
                with the minmaxscaler method (fnorm function), and if the value is equal to 0, the 
                normalization is made dividing the value by 180, this is only for directions.
                For summarize, index (input list) must has 1 at index where dataframe columns 
                aren't directions and 0 where the columns are directions.
            maxi: float
                max value of the normalized array
            mini: float
                min value of the normalized array
            dirnorm = float
                value used for the normalization of direction arrays
  
        Return
            dfnorm: dataframe
                normalized dataframe, it has the same dimensions that df
    '''
    dfnorm = pd.DataFrame(columns = df.columns, index = df.index)
    for i,j in zip(index, range(len(index))):
        if i == 1:
            dfnorm[df.columns[j]] = fnorm(df.iloc[:,j].values, maxi, mini)
        else:
            dfnorm[df.columns[j]] = df.iloc[:,j].values/dirnorm
    return dfnorm

def unnormalize_df(df, clusters_norm, index, maxi = 1, mini = 0, dirnorm = 180):
    ''' Apply inverse normalization function (fnorm_inv defined in tools package) to a dataframe
  
        Parameters
            df: dataframe
                original dataframe it's the one which was normalized in the first place.
                The dimension of the dataframe can be MxN.
            clusters_norm: dataframe
                dataframe to unnormalize. The dimension of the dataframe can be KxN (K << M).
            inde: list
                list with binary values. The list length must be equals to the number of parameters 
                in the cluster analysis. If the value is equal to 1, the normalization is computed
                with the minmaxscaler method (fnorm function), and if the value is equal to 0, the 
                normalization is made dividing the value by 180, this is only for directions.
                For summarise, index (input list) must has 1 at index where dataframe columns 
                arent directions and 0 where the columns are directions.
            maxi: float
                max value of the normalized array
            mini: float
                min value of the normalized array
            dirnorm = float
                value used for the normalization of direction arrays
  
        Returns
            clusters_unnorm: dataframe unnormalized. The dimension of the dataframe is be KxN.
            
    '''
    clusters_unnorm = pd.DataFrame(columns = df.columns)
    for i, j in zip(index, range(len(index))):
        if i == 1:
            clusters_unnorm[df.columns[j]] = fnorm_inv(df.iloc[:,j].values, 
                                                clusters_norm.iloc[:,j].values, maxi, mini)
        else:
            clusters_unnorm[df.columns[j]] = clusters_norm.iloc[:,j].values*180
    return clusters_unnorm

def anti_neighbors(df, k, index, maxi = 1, mini = 0, dirnorm = 180, nkmeans = 0, voronoi_weight = True):
    '''Search the k farthest neighbors. This fx calls "ecdist_mda", "normalize_df" and "unnormalize_df" as auxiliary functions
  
        Parameters
            df: dataframe
                input data from where the clusters will be extracted
            k: int
                number of clusters wanted
            index: list
               list with binary values. It must has 1 at index where dataframe columns 
                arent directions and 0 where the columns are directions.
            maxi: float
                max value of the normalized array
            mini: float
                min value of the normalized array
            dirnorm = float
                value used for the normalization of direction arrays
  
        Returns
            clusters: dataframe
            
        FOR MORE HELP SEE THE DOCSTRINGS OF ecdist_mda, normalize_df and unnormalize_df functios!
            
    '''
    norm = normalize_df(df, index)
        
    points = [tuple(x) for x in norm.values]
    dummy_index = list(df.index.values)
    remaining_points = points[:]
    solution_set = []
    index_solution_set = []
    
    tree = cKDTree(norm.values)
    meanvalue = list(norm.mean())
    ix_meanvalue = tree.query(meanvalue)[1]
    
    solution_set.append(remaining_points.pop(ix_meanvalue))
    index_solution_set.append(dummy_index.pop(ix_meanvalue))
    
    for _ in tqdm(range(k-1)):
        distances = [ecdist_mda(p, solution_set[0], index) for p in remaining_points]
        for i, p in enumerate(remaining_points):
            for j, s in enumerate(solution_set):
                distances[i] = min(distances[i], ecdist_mda(p, s, index))
        
        dummy = distances.index(max(distances))
        index_solution_set.append(dummy_index.pop(dummy))
        solution_set.append(remaining_points.pop(dummy))
        
    normclusters = pd.DataFrame(columns = df.columns, data = solution_set)
    clusters = unnormalize_df(df, normclusters, index)
    clusters['index_cluster'] = index_solution_set
    
    if nkmeans > 0:
        kmeans = KMeans(n_clusters = nkmeans, random_state = 42).fit(norm.values)
        ix_kmeans = tree.query(kmeans.cluster_centers_)[1]
        
        dfkmeans = df.iloc[ix_kmeans, :]
        dfkmeans['index_cluster'] = dfkmeans.index
        dfkmeans.index = range(k, k + nkmeans)
        clusters = clusters.append(dfkmeans)
        
    if voronoi_weight == True:
        voronoi_kdtree = cKDTree(clusters.iloc[:, :-1].values)
        points_dist, points_regions = voronoi_kdtree.query(df.values)
        region_percentage = []

        for x in range(len(clusters)):
            no_points_in_region = [y for y in points_regions if y == x]
            no_points_in_region = len(no_points_in_region)
            region_percentage.append(no_points_in_region/len(points_regions))
        clusters['weight'] = region_percentage
    
    return clusters
