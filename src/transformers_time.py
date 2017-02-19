import pandas as pd

# This transformer extracts the date/month/year and timestamp in a neat package
def tr_datetime(train,test, y):
    def _trans(df):
        df = df.assign(
            Created_TS = pd.to_datetime(df["created"])
        )
        return df.assign(
            Created_Year = df["Created_TS"].dt.year,
            Created_Month = df["Created_TS"].dt.month,
            Created_Day = df["Created_TS"].dt.day,
            Created_Hour = df["Created_TS"].dt.hour,
            Created_DayOfWeek = df["Created_TS"].dt.dayofweek
            )
    return _trans(train), _trans(test), y