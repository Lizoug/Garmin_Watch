import streamlit as st
import numpy as np
import os
import garminconnect
import streamlit_backend as sb
import json

st.title("Sleep Data")

st.subheader("Login")
# LOGIN container
col1, col2 = st.columns(2)
with col1:
    email = st.text_input("Enter email address")
with col2:
    password = st.text_input("Enter password:", type="password")

login_verify = st.button("Login", type="primary")

if login_verify:
    garmin = sb.login(email, password)
    # Remove password to prevent continuously API request
    password = ""

st.subheader("Duration of study")
# Create weekly df
col1, col2 = st.columns(2)
with col1:
    END_DATE = st.date_input("Select last desired date", value="today")
with col2:
    DURATION = st.number_input("Duration", value=14)

if login_verify:
    dataframe = sb.get_sleep_data(garmin, end_date=END_DATE, duration=DURATION)
    st.write(dataframe)

    st.subheader("Save the file")
    # Option to save the dataframe as JSON
    json_string = dataframe.to_json(orient='records', lines=True)

    # Download button to download JSON
    st.download_button(
        label="Download data as JSON",
        file_name='sleep_data.json',
        mime='application/json',
        data=json_string
    )