import streamlit as st
import pandas as pd
import garminconnect
from datetime import date, timedelta

def login(email, password):
    try:
        # Create a Garmin Connect instance and log in
        garmin = garminconnect.Garmin(email, password)
        garmin.login()

        # Retrieve the display name
        display_name = garmin.display_name
        st.write(f"Logged in as: {display_name}")
        return garmin
    
    except Exception as e:
        # Handle exceptions (e.g., login failure, network issues)
        st.write(f"An error occurred: {e}")

def get_sleep_data(garmin, end_date, duration):
    # Calculate the end date minus duration to get start date
    start_date = end_date - timedelta(days=duration)

    datelist = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]
    datelist_new = [x.isoformat() for x in datelist]

    dataframes_list = []  # List to store each day's DataFrame

    for day in datelist_new:
        try:
            current_entry = garmin.get_sleep_data(day)
            current_entry_df = pd.DataFrame([current_entry])  # Convert the dictionary to a DataFrame
            dataframes_list.append(current_entry_df)
        except Exception as e:
            st.write(f"Error retrieving data for {day}: {e}")

    # Concatenate all the DataFrames in the list
    return pd.concat(dataframes_list, ignore_index=True)