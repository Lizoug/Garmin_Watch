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
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

#email = input("Enter email address: ")
#password = getpass("Enter password: ")

email = "miriam.agrawala@study.hs-duesseldorf.de"
password = "Garmin1Garmin"

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

pca = PCA().fit(featurevector_timeseries_numpy)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')


# create a PCA object
pca = PCA(n_components=3)  # Reduce to 3 features
reduced_data_pca = pca.fit_transform(featurevector_timeseries_numpy.T)  # Transpose to get correct shape

print(reduced_data_pca.shape)  # Should print (7007, 3)

# Create the model
iso_forest = IsolationForest(
    n_estimators=100,
    contamination=0.1,
    n_jobs=-1,
    random_state=42,
    verbose=0)

# Fit the model
iso_forest.fit(reduced_data_pca)

# Predict degree of anomaly
scores_pred = iso_forest.decision_function(reduced_data_pca)

# Set the threshold
threshold = -0.1  # Adjust this value to suit your needs

# Classify data points as outliers if their anomaly score is less than the threshold
outlier_classification = np.where(scores_pred < threshold, -1, 1)

fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
scatter = ax.scatter(reduced_data_pca[:, 0], reduced_data_pca[:, 1],
           c=scores_pred, cmap="plasma",
           alpha=0.5)

# Adding a colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Anomaly Degree')
plt.grid(True)
plt.show()


# find out which samples are the outliers
outlier_indices = np.where(outlier_classification == -1)[0]
print("Outlier indices:", outlier_indices)
print("Number of outliers:", len(outlier_indices))

# Transpose the DataFrame
featurevector_timeseries_pandas_transposed = featurevector_timeseries_pandas.transpose()

# Get the outlier rows from the transposed DataFrame
outlier_rows = featurevector_timeseries_pandas_transposed.iloc[outlier_indices]
print("Outlier rows:", outlier_rows)