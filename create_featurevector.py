import garminconnect
import datetime
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import os
from getpass import getpass
from datetime import date, timedelta
import pandas as pd


# pd.set_option("display.max_rows", None)
# pd.set_option("display.max_columns", None)

# email = input("Enter email address: ")
# password = getpass("Enter password: ")
# garmin = garminconnect.Garmin(email, password)
# garmin.login()
# garmin.display_name

# # retrieve the value of the GARTH_HOME environment variable 
# # (defaulting to ~/.garth if not set) and then uses the 
# # garmin.garth.dump function to process that path.

# GARTH_HOME = os.getenv("GARTH_HOME", "~/.garth")
# garmin.garth.dump(GARTH_HOME)


# # Define your target end date
# end_date = date(2023, 12, 3) # miriam
# #end_date = date(2023, 10, 17) #liza


# # Calculate the start date as 2 weeks (14 days) before the end date
# start_date = end_date - timedelta(days=14) # miriam
# #start_date = end_date - timedelta(days=14) #liza

# datelist = []

# current_date = start_date
# while current_date <= end_date:
#     datelist.append(current_date)
#     current_date += timedelta(days=1)

# datelist_new = [x.isoformat() for x in datelist]

# dataframes_list = []  # List to store each day's DataFrame

# for day in datelist_new:
#     current_entry = garmin.get_sleep_data(day)
#     current_entry_df = pd.DataFrame([current_entry])  # Convert the dictionary to a DataFrame
#     dataframes_list.append(current_entry_df)

# # Concatenate all the DataFrames in the list
# week_sleep_df = pd.concat(dataframes_list, ignore_index=True)

# # data from only one day
# sleep_one_day = week_sleep_df.iloc[0]
# sleep_one_day

# def delete_untracked_nights(df):
#     """Delete the untracked nights by using the restlessMomentsCount"""
#     return df.dropna(subset=["restlessMomentsCount"]).reset_index(drop=True) # reset the index 

# week_sleep_df = delete_untracked_nights(week_sleep_df)

def featurevector_timeseries(week_sleep_df):
    #preliminary_featurevector_allnights = pd.DataFrame()
    preliminary_featurevector_allnights = pd.DataFrame(index=["sleepRestlessMoments", "hrvData", "sleepStress", "sleepBodyBattery", "sleepHeartRate", "wellnessEpochRespirationDataDTOList", "sleepLevels"])
    # iterate over indeces of week_sleep_df to only process one day at a time
    for i in week_sleep_df.index:
        # initialize empty Dataframe. This will be filled line by line with each iteration, adding a different feature in each row
        preliminary_featurevector = pd.DataFrame()
        sleep_one_day = week_sleep_df.iloc[i]
        # columns that use "startGMT" as name for their timecolumn
        columns_startGMT = ["sleepRestlessMoments", "hrvData", "sleepStress", "sleepBodyBattery", "sleepHeartRate"]
        # # columns that have one single value for the whole night
        # static_columns = ["remSleepData", "restlessMomentsCount", "avgOvernightHrv", "restingHeartRate"]
        # iterate over different columns
        for c in columns_startGMT:
            column_df = [pd.json_normalize(item) for item in sleep_one_day[c]]
            column_df = pd.concat(column_df, ignore_index=True)
            
            column_df["Real Time"] = [datetime.datetime.fromtimestamp(i) for i in column_df["startGMT"] / 1000]
            column_df["Real Time"] = pd.to_datetime(column_df["Real Time"])
            # make Real Time the index of the dataframe
            column_df.index = column_df["Real Time"]
            del column_df["Real Time"]
            del column_df["startGMT"]
            column_df = column_df.rename(columns = {column_df.columns[0]: c})
                    
            # resample: insert timesteps for each minute
            column_df_interpolate = column_df.resample('T').mean()
            # interpolate values to fill the new, empty time steps
            column_df_interpolate[c] = column_df_interpolate[c].interpolate()
            # drop unneeded columns, swap axes
            column_df_interpolate_transposed = column_df_interpolate.T
            # add to feature vector
            preliminary_featurevector = pd.concat([preliminary_featurevector, column_df_interpolate_transposed], axis=0)

            #preliminary_featurevector = preliminary_featurevector.rename(index={((len(preliminary_featurevector.index))-1): c})
            
        # sleep_one_day["wellnessEpochRespirationDataDTOList"] uses "startTimeGMT" as name for their time columns, so it has to be processed separately
        wellnessEpochRespiration_df = [pd.json_normalize(item) for item in sleep_one_day["wellnessEpochRespirationDataDTOList"]]
        wellnessEpochRespiration_df = pd.concat(wellnessEpochRespiration_df, ignore_index=True)
        
        wellnessEpochRespiration_df["Real Time"] = [datetime.datetime.fromtimestamp(i) for i in wellnessEpochRespiration_df["startTimeGMT"] / 1000]
        wellnessEpochRespiration_df["Real Time"] = pd.to_datetime(wellnessEpochRespiration_df["Real Time"])
        wellnessEpochRespiration_df.index = wellnessEpochRespiration_df["Real Time"]
        del wellnessEpochRespiration_df["Real Time"]
        del wellnessEpochRespiration_df["startTimeGMT"]
        wellnessEpochRespiration_df = wellnessEpochRespiration_df.rename(columns = {wellnessEpochRespiration_df.columns[0]: "wellnessEpochRespirationDataDTOList"})
        
        wellnessEpochRespiration_df_interpolate = wellnessEpochRespiration_df.resample('T').mean()
        wellnessEpochRespiration_df_interpolate['wellnessEpochRespirationDataDTOList'] = wellnessEpochRespiration_df_interpolate['wellnessEpochRespirationDataDTOList'].interpolate()
        wellnessEpochRespiration_df_interpolate_transposed = wellnessEpochRespiration_df_interpolate.T
        preliminary_featurevector = pd.concat([preliminary_featurevector, wellnessEpochRespiration_df_interpolate_transposed], axis=0)
        
        # # add the features with static values
        # for s in static_columns:
        #     preliminary_featurevector.loc[len(preliminary_featurevector.index)] = sleep_one_day[s]
        #     preliminary_featurevector = preliminary_featurevector.rename(index={((len(preliminary_featurevector.index))-1): s})


        # in sleep_one_day['sleepLevels'] the sleep levels are displayed in intervalls from startGMT to endGMT
        # .ffil was used for forward filling, so that all timesteps (per minute) in the interval would have the same value
        sleepLevels_df = [pd.json_normalize(item) for item in sleep_one_day['sleepLevels']]
        sleepLevels_df = pd.concat(sleepLevels_df, ignore_index=True)
        
        sleepLevels_df["startGMT"] = pd.to_datetime(sleepLevels_df["startGMT"])
        sleepLevels_df.index = sleepLevels_df["startGMT"]
        del sleepLevels_df["startGMT"]
        del sleepLevels_df["endGMT"]
        sleepLevels_df = sleepLevels_df.rename(columns = {sleepLevels_df.columns[0]: "sleepLevels"})
        #.ffil -> forward fill, replace all NaN values with the last numeric value that occured
        sleepLevels_ffil = sleepLevels_df.resample("T").ffill()
        sleepLevels_ffil = sleepLevels_ffil.T
        preliminary_featurevector = pd.concat([preliminary_featurevector, sleepLevels_ffil], axis=0)
        preliminary_featurevector_allnights = preliminary_featurevector_allnights.merge(preliminary_featurevector, left_index=True, right_index=True)
        # preliminary_featurevector_allnights = pd.concat([preliminary_featurevector_allnights, preliminary_featurevector], axis=1)

    #preliminary_featurevector_allnights = preliminary_featurevector_allnights.dropna(axis=1)

    # replace all NaNs with 0
    preliminary_featurevector_allnights = preliminary_featurevector_allnights.fillna(0)
    # convert to numpy for usage as feature vector
    preliminary_featurevector_allnights_np = preliminary_featurevector_allnights.to_numpy()

    return preliminary_featurevector_allnights, preliminary_featurevector_allnights_np

# featurevector_timeseries_pandas, featurevector_timeseries_numpy = featurevector_timeseries(week_sleep_df)

# print(featurevector_timeseries_numpy)
# print(type(featurevector_timeseries_numpy))