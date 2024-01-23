from create_featurevector import featurevector_timeseries
import garminconnect
import datetime
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import os
from getpass import getpass
from datetime import date, timedelta
import pandas as pd


pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

email = input("Enter email address: ")
password = getpass("Enter password: ")
garmin = garminconnect.Garmin(email, password)
garmin.login()
garmin.display_name

# retrieve the value of the GARTH_HOME environment variable 
# (defaulting to ~/.garth if not set) and then uses the 
# garmin.garth.dump function to process that path.

GARTH_HOME = os.getenv("GARTH_HOME", "~/.garth")
garmin.garth.dump(GARTH_HOME)


# Define your target end date
end_date = date(2023, 12, 3) # miriam
#end_date = date(2023, 10, 17) #liza


# Calculate the start date as 2 weeks (14 days) before the end date
start_date = end_date - timedelta(days=14) # miriam
#start_date = end_date - timedelta(days=14) #liza

datelist = []

current_date = start_date
while current_date <= end_date:
    datelist.append(current_date)
    current_date += timedelta(days=1)

datelist_new = [x.isoformat() for x in datelist]

dataframes_list = []  # List to store each day's DataFrame

for day in datelist_new:
    current_entry = garmin.get_sleep_data(day)
    current_entry_df = pd.DataFrame([current_entry])  # Convert the dictionary to a DataFrame
    dataframes_list.append(current_entry_df)

# Concatenate all the DataFrames in the list
week_sleep_df = pd.concat(dataframes_list, ignore_index=True)

# data from only one day
sleep_one_day = week_sleep_df.iloc[0]
sleep_one_day

def delete_untracked_nights(df):
    """Delete the untracked nights by using the restlessMomentsCount"""
    return df.dropna(subset=["restlessMomentsCount"]).reset_index(drop=True) # reset the index 

week_sleep_df = delete_untracked_nights(week_sleep_df)

featurevector_timeseries_pandas, featurevector_timeseries_numpy = featurevector_timeseries(week_sleep_df)

print(featurevector_timeseries_numpy)
print(type(featurevector_timeseries_numpy))
print(featurevector_timeseries_numpy.shape)