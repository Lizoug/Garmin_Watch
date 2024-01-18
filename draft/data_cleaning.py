import datetime
import pandas as pd
import numpy as np

def normalize_and_rename_columns(df, column_name, prefix_to_remove=None):
    # Normalize the specified column
    expanded_df = pd.json_normalize(df[column_name])

    # Optionally remove a specified prefix from column names
    if prefix_to_remove:
        expanded_df.columns = [col.replace(prefix_to_remove + '.', '') for col in expanded_df.columns]

    return expanded_df

def convert_num(input_value):
    """
    Function that converts the string qualifiers to numerical value.  
    """
    score_dict = {"EXCELLENT": 4, "GOOD" : 3, "FAIR" : 2, "POOR" : 1}
    for key,value in score_dict.items():
        if input_value == key:
            input_value = value
    return input_value

def delete_untracked_nights(df):
    """
    Delete the untracked nights by using the restlessMomentsCount.
    This is because restless moments are only registered in the watch when sleep is detected.
    The subset was -restlessMomentsCount-.
    """
    return df.dropna(subset=["sleepMovement"]).reset_index(drop=True) # reset the index

def featurevector_timeseries(df):
    """
    Function that returns the time-sensitive readings from the watch as feature vectors in form of pandas dataframe and numpy array.
    The columns that are processed together are done so due to their common format.
    -    startGMT: sleepRestlessMoments, hrvData, sleepBodyBattery, sleepStress, sleepHeartRate
    -    startTimeGMT: welnessEpochRespirationDataDTOList
    -    startGMT - endGMT: sleepLevels
    """

    #preliminary_featurevector_allnights = pd.DataFrame()
    preliminary_featurevector_allnights = pd.DataFrame(index=["sleepRestlessMoments", "hrvData", "sleepStress", "sleepBodyBattery", "sleepHeartRate", "wellnessEpochRespirationDataDTOList", "sleepLevels"])
    # iterate over indeces of df to only process one day at a time

    for i in df.index:
        # initialize empty Dataframe. This will be filled line by line with each iteration, adding a different feature in each row
        preliminary_featurevector = pd.DataFrame()
        sleep_one_day = df.iloc[i] ### may use loc instead of iloc
        # columns that use "startGMT" as name for their timecolumn
        columns_startGMT = ["sleepRestlessMoments", "hrvData", "sleepStress", "sleepBodyBattery", "sleepHeartRate"]

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
        

    # replace all NaNs with 0
    preliminary_featurevector_allnights = preliminary_featurevector_allnights.fillna(0)
    # convert to numpy for usage as feature vector
    preliminary_featurevector_allnights_np = preliminary_featurevector_allnights.to_numpy()

    return preliminary_featurevector_allnights, preliminary_featurevector_allnights_np