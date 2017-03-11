import numpy as np
import pandas as pd

#from hdbscan import HDBSCAN
from hdbscan import HDBSCAN, approximate_predict
from geopy.distance import great_circle
from shapely.geometry import MultiPoint
from sklearn import metrics
import time


# from http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html
# All lat/long must be converted to radians, i.e. multiplied by pi/180
#output is also in rad

def tr_clustering(train,test, y, folds, cache_file):
    def _get_centermost_point(cluster):
        centroid = (MultiPoint(cluster).centroid.x, MultiPoint(cluster).centroid.y)
        centermost_point = min(cluster, key=lambda point: great_circle(point, centroid).m)
        lat,lon,label = centermost_point
        return lat,lon
    
    def _cluster_train(df):
        
        start_time = time.time()
        # Note: Issue #88 open, prediction_data cannot be used with Haervsine metrics https://github.com/scikit-learn-contrib/hdbscan/issues/88
        db = HDBSCAN(min_samples=1,
                    metric='haversine',
                    core_dist_n_jobs=-1,
                    memory='./__pycache__/',
                    prediction_data=True 
                   )
        
        coords = df[['latitude','longitude']] * np.pi/180
        
        df = df.assign( cluster = db.fit_predict(coords) )
        # get the number of clusters
        num_clusters = db.labels_.max()
        message = 'Clustered {:,} points down to {:,} clusters, for {:.1f}% compression in {:,.2f} seconds'
        print(message.format(len(df), num_clusters, 100*(1 - float(num_clusters) / len(df)), time.time()-start_time))
        
        # Get the list of the point most in the center of each clusters
        cluster_centers = df[['cluster', 'latitude','longitude']].groupby('cluster')['latitude','longitude'].agg(lambda x: _get_centermost_point(x.values))
                
        df = df.merge(cluster_centers, left_on='cluster', right_index=True,how='left',suffixes=('', '_cluster'))        
        return db, cluster_centers, df
    
    def _cluster_test(hdbscan, cluster_centers, df):
        coords = df[['latitude','longitude']] * np.pi/180
        
        df = df.assign(cluster = approximate_predict(hdbscan, coords)[0])
        df = df.merge(cluster_centers, left_on='cluster', right_index=True,how='left',suffixes=('', '_cluster'))
        
        return df
    
    hdbscan, cluster_centers, trn = _cluster_train(train)
    tst = _cluster_test(hdbscan, cluster_centers, test)
    return trn,tst, y, folds, cache_file