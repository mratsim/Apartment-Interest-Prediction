import math
import numpy as np

def cart2rho(x, y):
    rho = np.sqrt(x**2 + y**2)
    return rho


def cart2phi(x, y):
    phi = np.arctan2(y, x)
    return phi


def rotation_x(row, alpha):
    x = row['latitude']
    y = row['longitude']
    return x*math.cos(alpha) + y*math.sin(alpha)


def rotation_y(row, alpha):
    x = row['latitude']
    y = row['longitude']
    return y*math.cos(alpha) - x*math.sin(alpha)


def add_rotation(degrees, df):
    namex = "coord_" + str(degrees) + "_X"
    namey = "coord_" + str(degrees) + "_Y"

    df[namex] = df.apply(lambda row: rotation_x(row, math.pi/(180/degrees)), axis=1)
    df[namey] = df.apply(lambda row: rotation_y(row, math.pi/(180/degrees)), axis=1)

    return df

def operate_on_coordinates(tr_df, te_df):
    for df in [tr_df, te_df]:
        #polar coordinates system
        df["rho_centralpark"] = df.apply(lambda x: cart2rho(x["latitude"] - 40.78222222, x["longitude"]+73.96527777), axis=1)
        df["phi_centralpark"] = df.apply(lambda x: cart2phi(x["latitude"] - 40.78222222, x["longitude"]+73.96527777), axis=1)
        #rotations
        for angle in [15,30,45,60]:
            df = add_rotation(angle, df)

    return tr_df, te_df

def tr_rotation_around_central_park(train,test, y, folds, cache_file):
    train, test = operate_on_coordinates(train, test)
    
    return train,test, y, folds, cache_file

def tr_dist_to_main_centers(train,test, y, folds, cache_file): # That is external data and probably out scope of the competition
    def _trans(df):
        location_dict = {
            'manhattan_loc': [40.728333, -73.994167],
            'brooklyn_loc': [40.624722, -73.952222],
            'bronx_loc': [40.837222, -73.886111],
            'queens_loc': [40.75, -73.866667],
            'staten_loc': [40.576281, -74.144839]}

        for location in location_dict.keys():
            dlat = location_dict[location][0] - df['latitude']
            dlon = (location_dict[location][1] - df['longitude']) * np.cos(np.deg2rad(41))  #  adjust for NYC latitude
            df['distance_' + location] = np.sqrt(dlat ** 2 + dlon ** 2) * 60     # distance in nautical miles
        return df
    return _trans(train),_trans(test), y, folds, cache_file