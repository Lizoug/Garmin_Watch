from datetime import date, timedelta
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

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
    Delete the untracked nights by using the restlessMomentsCount & sleepMovement. 
    This is because restless moments are only registered in the watch when sleep is detected.
    """
    # remove empty list from this column
    mask = df['sleepMovement'].apply(lambda x: x == [])
    df = df[~mask]

    # remove NaN in these 2 columns
    df = df.dropna(subset=["sleepMovement", "restlessMomentsCount"]).reset_index(drop=True) # reset the index

    return df.dropna(axis='rows').reset_index(drop=True)

def convert_timestamps(df, timestamp_column, time_offset_hours=0):
    """
    Function that converts timestamps in a dataframe to a timezone-aware datetime format.
    """ 
    # Convert timestamp column to datetime
    df[timestamp_column] = pd.to_datetime(df[timestamp_column], unit='ms')

    # Convert GMT to local time by adding the specified number of hours
    #local_time_column = "startLocal"
    df[timestamp_column] = df[timestamp_column] + timedelta(hours=time_offset_hours)

    return df

def extract_value(df):
    interested_columns = ["sleepRestlessMoments", "hrvData", "sleepStress", "sleepBodyBattery", 
                      "sleepHeartRate", "wellnessEpochRespirationDataDTOList", "sleepLevels"]
    dfs= []
    for column in interested_columns:
        if column == 'sleepLevels':
            # column with startGMT, endGMT and activityLevel
            # 3 columns in total
            df1 = pd.concat([pd.json_normalize(item) for item in df[column]])
            # change the date from string to datetime
            df1['startGMT'] = pd.to_datetime(df1['startGMT'])
            # add 1 hour timedelta, to get local time
            df1['startGMT'] += timedelta(hours=1)
            df1.drop("endGMT", axis='columns', inplace=True)
            # we need to rename the column activityLevel to sleepLevel_value
            df1.rename(columns={'activityLevel': 'sleepLevel_value'}, inplace=True)
        elif column == 'wellnessEpochRespirationDataDTOList':
            # 2 columns startTimeGMT and value
            # column start with startTimeGMT
            df2 = pd.concat([pd.json_normalize(item) for item in df[column]])
            df2.rename(columns={'startTimeGMT': 'startGMT', 
                                'value': f'{column}_value'}, inplace=True)
            convert_timestamps(df2, 'startGMT', 1)
            ### we need to rename the column value to f"{column_name}_value"
        else:
            # 2 columns, startGMT and value
            df3 = pd.concat([pd.json_normalize(item) for item in df[column]])
            convert_timestamps(df3, 'startGMT', 1)
            ### we need to rename the column value to f"{column_name}_value"
            df3.rename(columns={'value': f'{column}_value'}, inplace=True)
            dfs.append(df3)

    return df1, df2, dfs

def merge_extracted_dataframes(main_df):
    # Extract the first two DataFrames and the list of DataFrames using the extract_value function
    df1 = extract_value(main_df)[0].set_index('startGMT')
    df2 = extract_value(main_df)[1].set_index('startGMT')
    df_list = extract_value(main_df)[2]
 
    # Merge df1 and df2 first. They are merged on their indices.
    merged_df = df1.merge(df2, left_index=True, right_index=True, how='outer')
 
    # Iteratively merge each DataFrame in df_list with merged_df
    for df in df_list:
        df.set_index('startGMT', inplace=True)  # Set 'startGMT' as the index for each DataFrame in df_list
        merged_df = merged_df.merge(df, left_index=True, right_index=True, how='outer')  # Adjust the merge type as necessary
 
    return merged_df

def generate_columns_to_interpolate(columns_to_rename):
    columns_to_interpolate = []
    for string in columns_to_rename:
        columns_to_interpolate.append(f"{string}_value")
    columns_to_interpolate.append('respirationValue')
    return columns_to_interpolate
 
def interpolate_dataframe(merged_df, columns_to_interpolate):
    interpolated_df = merged_df.copy()
    for column in columns_to_interpolate:
        if column in ['sleepLevel_value', 'sleepRestlessMoments_value']:
            # Use forward fill for these columns
            interpolated_df[column] = interpolated_df[column].ffill() # same as interpolation (pad method)
            interpolated_df[column] = interpolated_df[column].bfill() # backward fill the NaN values
        else:
            # Use time interpolation for other columns
            interpolated_df[column] = interpolated_df[column].interpolate(method='time') # interpolation (time)
            interpolated_df[column] = interpolated_df[column].bfill() # backward fill the NaN values
    return interpolated_df

def normalize_data(df):
    # Initialize the Min-Max scaler
    scaler = MinMaxScaler()

    # Normalize the DataFrame using the scaler
    normalized_array = scaler.fit_transform(df)

    # Convert the normalized NumPy array back into a DataFrame
    normalized_df = pd.DataFrame(normalized_array, columns=df.columns, index=df.index)

    return normalized_df

def main_interpolation(original_df):
    # Step 1: Delete untracked nights
    cleaned_df = delete_untracked_nights(original_df)

    # Extrahieren der Schlafstart- und -endzeiten in einem neuen DataFrame
    df_temp = pd.json_normalize(cleaned_df['dailySleepDTO'])[['sleepStartTimestampLocal', 'sleepEndTimestampLocal']]
    df_temp['sleepStartTimestampLocal'] = pd.to_datetime(df_temp['sleepStartTimestampLocal'], unit='ms')
    df_temp['sleepEndTimestampLocal'] = pd.to_datetime(df_temp['sleepEndTimestampLocal'], unit='ms')

    # Step 2: Merge the extracted DataFrames
    merged_df = merge_extracted_dataframes(cleaned_df)

    # Step 3: Generate column names for interpolation
    columns_to_rename = ["sleepRestlessMoments", "hrvData", "sleepStress", "sleepBodyBattery",
                         "sleepHeartRate", "sleepLevel"]
    columns_to_interpolate = generate_columns_to_interpolate(columns_to_rename)

    # Step 4: Resample and interpolate
    resampled_df = merged_df.resample('T').mean()  # Diskutieren, ob eine oder zwei Minuten
    final_processed_df = interpolate_dataframe(resampled_df, columns_to_interpolate)

    # Normalize with MinMaxScaler()
    df_normalized = normalize_data(final_processed_df)

    # Step 5: Split the night separately
    night_dfs = {}
    for index, row in df_temp.iterrows():
        start = row['sleepStartTimestampLocal']
        end = row['sleepEndTimestampLocal']
        filtered_df = df_normalized[start:end]
        night_dfs[index] = filtered_df

    return night_dfs  # Return a dictionary of night DataFrames