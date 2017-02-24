import pandas as pd
import numpy as np

# This transformer extracts the date/month/year and timestamp in a neat package
def tr_datetime(train,test, y, cache_file):
    def _isWeekend(i):
        if i in [5,6]:
            return 1
        else:
            return 0
        
    def _trans(df):
        df = df.assign(
            Created_TS = pd.to_datetime(df["created"])
        )
        return df.assign(
            Created_Year = df["Created_TS"].dt.year,
            Created_Month = df["Created_TS"].dt.month,
            Created_Day = df["Created_TS"].dt.day,
            Created_Hour = df["Created_TS"].dt.hour,
            Created_DayOfWeek = df["Created_TS"].dt.dayofweek,
            Created_DayOfYear = df["Created_TS"].dt.dayofyear,
            Created_WeekOfYear = df["Created_TS"].dt.weekofyear,
            
            Created_D_cos = np.cos(((df["Created_TS"].dt.day -1)/31)*2*np.pi),
            Created_D_sin = np.sin(((df["Created_TS"].dt.day -1)/31)*2*np.pi),
            Created_DoW_cos = np.cos(((df["Created_TS"].dt.dayofweek -1)/7)*2*np.pi),
            Created_DoW_sin = np.sin(((df["Created_TS"].dt.dayofweek -1)/7)*2*np.pi),
            Created_DoY_cos = np.cos(((df["Created_TS"].dt.dayofyear -1)/365)*2*np.pi),
            Created_DoY_sin = np.sin(((df["Created_TS"].dt.dayofyear -1)/365)*2*np.pi),
            Created_WoY_cos = np.cos(((df["Created_TS"].dt.weekofyear -1)/52)*2*np.pi),
            Created_WoY_sin = np.sin(((df["Created_TS"].dt.weekofyear -1)/52)*2*np.pi),
            Created_Weekend = df["Created_TS"].dt.dayofyear.apply(_isWeekend)
            
            )
    return _trans(train), _trans(test), y, cache_file