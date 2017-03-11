import pandas as pd
import numpy as np
from workalendar.usa import NewYork
import datetime

# This transformer extracts the date/month/year and timestamp in a neat package
def tr_datetime(train,test, y, folds, cache_file):
    def _isWeekend(i):
        if i in [5,6]:
            return 1
        else:
            return 0
        
    def _trans(df):
        df = df.assign(
            Created_TS = pd.to_datetime(df["created"])
        )
        cal = NewYork()
        school_holidays = ["2016-01-01",
                   "2016-01-18",
                   "2016-02-08",
                   "2016-02-15","2016-02-16","2016-02-17","2016-02-18","2016-02-19",
                   "2016-03-25",
                   "2016-04-25","2016-04-26","2016-04-27","2016-04-28","2016-04-29",
                   "2016-05-30",
                   "2016-06-09",
                  "2016-09-12",
                  "2016-10-03","2016-10-04",
                  "2016-10-10",
                  "2016-10-12",
                  "2016-11-08",
                  "2016-11-11",
                  "2016-11-24","2016-11-25",
                  "2016-12-26","2016-12-27","2016-12-28","2016-12-29","2016-12-30"
                  ]
        start_summer = datetime.date(2016,6, 29)
        end_summer = datetime.date(2016,9, 7)
        summer = [start_summer +
          datetime.timedelta(days=x)
          for x in range((end_summer-start_summer).days + 1)]
        school_holidays.extend([d.strftime("%Y-%m-%d") for d in summer])
        
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
            Created_H_cos = np.cos(((df["Created_TS"].dt.hour -1)/24)*2*np.pi),
            Created_H_sin = np.sin(((df["Created_TS"].dt.hour -1)/24)*2*np.pi),
            Created_DoW_cos = np.cos(((df["Created_TS"].dt.dayofweek -1)/7)*2*np.pi),
            Created_DoW_sin = np.sin(((df["Created_TS"].dt.dayofweek -1)/7)*2*np.pi),
            Created_DoY_cos = np.cos(((df["Created_TS"].dt.dayofyear -1)/365)*2*np.pi),
            Created_DoY_sin = np.sin(((df["Created_TS"].dt.dayofyear -1)/365)*2*np.pi),
            Created_WoY_cos = np.cos(((df["Created_TS"].dt.weekofyear -1)/52)*2*np.pi),
            Created_WoY_sin = np.sin(((df["Created_TS"].dt.weekofyear -1)/52)*2*np.pi),
            Created_Weekend = df["Created_TS"].dt.dayofyear.apply(_isWeekend),
            Time_passed = (df["Created_TS"].max() - df["Created_TS"])/np.timedelta64(-1, 'D'),
            Is_Holiday = df["Created_TS"].dt.date.map(cal.is_holiday),
            Is_SchoolHoliday = df["Created_TS"].dt.strftime('%Y-%m-%d').isin(school_holidays)
            
            )
    return _trans(train), _trans(test), y, folds, cache_file