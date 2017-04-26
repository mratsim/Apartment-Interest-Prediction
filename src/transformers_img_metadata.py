import pandas as pd

# This transformer attach the "magic" feature (folder creation time) to each images
def tr_magic_folder_time(train, test, y, folds, cache_file):
    df_time = pd.read_csv('./data_preprocessed/listing_image_time.csv')
    def _trans(df, df_time):
        # Merged columns are Listing_Id and time_stamp
        return df.merge(df_time,how='left',left_on='listing_id',right_on='Listing_Id') 
    return _trans(train, df_time), _trans(test, df_time), y, folds, cache_file