# import the module needed
import os
import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import Input, MaxPooling1D, UpSampling1D
from sklearn.preprocessing import MinMaxScaler
from keras.utils import pad_sequences
import tensorflow as tf
import keras
from keras import layers
import numpy as np
from sklearn.model_selection import train_test_split
import draft.data_cleaning as dc

# Load the datasets
person_0 = pd.read_json(r'datasets/december/liza.json', orient='records', lines=True) # liza
person_1 = pd.read_json(r'datasets/december/sleep_data_Adham.json', lines=True) # adham
person_2 = pd.read_json(r'datasets/december/sleep_data_Miriam.json', lines=True) # miriam
person_3 = pd.read_json(r'datasets/december/sleep_data_Syahid.json', lines=True) # syahid
person_4 = pd.read_json(r'datasets/december/sleep_data_florian.json', lines=True)  #  florian
person_5 = pd.read_json(r'datasets/december/sleep_data_Shado.json', lines=True) # shado
person_6 = pd.read_json(r'datasets/december/sleep_data_Alina.json', lines=True) # alina

# Load labels dataframe from excel
labels_df = pd.read_excel(r'datasets\sleep_data.xlsx', sheet_name=None) # dict of all label

# Create a temp_id for every person
people_df = [person_0, person_1, person_2, person_3, person_4, person_5, person_6]

for num, df in enumerate(people_df):
    df.insert(0, "temp_id", num)

# Interpolate time series data
time_series_list = []

for num, df in enumerate(people_df):
    # Extrahieren der Zeitreihendaten (Annahme: gibt einen DataFrame zurück)
    temp_df = dc.main_interpolation(df)

    # Weisen Sie die eindeutige ID dem DataFrame zu
    temp_df['temp_id'] = num

    # Fügen Sie den aktualisierten DataFrame der neuen Liste hinzu
    time_series_list.append(temp_df)

# Find total nights
total_nights = 0
# Loop through every person in list
for person_index, person_data in enumerate(time_series_list):
    print(f"Person {person_index + 1}:")

    # Überprüfen, ob person_data ein Dictionary ist
    if isinstance(person_data, dict):
        # Durchlaufen jedes Nacht-DataFrames der Person
        for night, df in person_data.items():
            # Überprüfen, ob der Wert ein DataFrame ist
            if isinstance(df, pd.DataFrame):
                print(f"  Nacht {night}: {len(df)} Zeilen")
                total_nights +=1

print("Total nights: ", total_nights)

# Get maximum length of the interpolated data
def calculate_maximum_length(list_):
    max_length = 0  # Initialize max_length to store the maximum number of rows

    # Iterate through each person in the list
    for person_index, person_data in enumerate(time_series_list):
        # Iterate through each night DataFrame of the person
        for night, df in person_data.items():
            # Check if the value is a DataFrame
            if isinstance(df, pd.DataFrame):
                # Update max_length if this night has more rows
                if len(df) > max_length:
                    max_length = len(df)

    # After completing the iteration, max_length will hold the number of rows of the longest night
    return max_length

max_length = calculate_maximum_length(time_series_list)

"""This step is necessary because we are using convolutional layers that halve the dimensions after every layer. 
If any halve produces a .5, the ceiling will be taken. This causes the Upsampling to reproduce different dimensions that our original input. """

# Determine how many layers are in our Conv1D network. (how many times the data will be halved)
n_layers = 6

while True:
    if max_length % n_layers != 0:
        max_length += 1
    else:
        break

print("Max length: ", max_length)

# List to store normalized and padded dataframes
normalized_padded_dfs = []

for person_data in time_series_list:
    for night,df in person_data.items():
        if isinstance(df, pd.DataFrame):
            # Pad them to ensure  they have the same length (needed for training/testing split)
            df_padded = pad_sequences([df.values],maxlen=max_length,dtype='float32', padding='post')
            # Add the padded dataframe to the list of normalized & padded dataframes
            normalized_padded_dfs.append((df_padded[0]))

# Convert the list to an array
data = np.array(normalized_padded_dfs)

# Split into training and test
X_train, X_test = train_test_split(data, test_size=0.2,random_state=42)

# Define input dimensions
n_features = 7

# Define autoencoder architecture
model = keras.Sequential(
    [
        layers.Input(shape=(max_length,n_features)),
        layers.Conv1D(
            filters=64,
            kernel_size=7,
            padding="same",
            strides=2,
            activation="relu",
        ),
        layers.MaxPooling1D(
            pool_size=2, 
            strides=2, 
            padding="valid",
        ),
        layers.Conv1D(
            filters=32,
            kernel_size=7,
            padding="same",
            strides=2,
            activation="relu",
        ),
        layers.MaxPooling1D(
            pool_size=2, 
            strides=2, 
            padding="valid",
        ),
        layers.Flatten(),  # Flatten the output
        layers.Dense(500, activation='relu'),
        layers.Dense(100, activation='relu'),
        layers.Dense(500, activation='relu'),
        layers.Dense(1984, activation='relu'),
        layers.Reshape((62, 32)),  # Reshape to match the original shape
        layers.Conv1DTranspose(
            filters=32,
            kernel_size=7,
            padding="same",
            strides=2,
            activation="relu",
        ),
        # # # upsampling layer
        layers.UpSampling1D(size=2),
        layers.Conv1DTranspose(
            filters=64,
            kernel_size=7,
            padding="same",
            strides=2,
            activation="relu",
        ),
        layers.UpSampling1D(size=2),
        layers.Conv1DTranspose(filters=n_features, kernel_size=7, padding="same"),
        layers.ZeroPadding1D(padding=2)
    ]
)

# Define function to save embeddings
def save_embeddings(model, x_data, save_path='embeddings.npy'):
    embeddings = model.predict(x_data)
    np.save(save_path, embeddings)
    print(f'Embeddings saved to {save_path}')

 # Compile model and visualize layers
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
# Print model architecture
print(model.summary())

# Define checkpoints and early stopping
checkpoint = keras.callbacks.ModelCheckpoint('autoencoder_Conv1D.h5', save_best_only=True)
early_stopping = keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

# Train the autoencoder
history = model.fit(
    X_train,
    X_train,
    epochs=250,
    batch_size=32,
    validation_split=0.15,
    callbacks=[
        checkpoint,
        early_stopping
    ],
)

# Create the embeddings

# Recreate the exact same model, including its weights and the optimizer
autoencoder= tf.keras.models.load_model('autoencoder_Conv1D.h5')

# get only the encoder layer
encoder = keras.models.Model(inputs= autoencoder.input, outputs=autoencoder.get_layer(index=6).output)

# Show the model architecture
print(encoder.summary())

# Create embeddings folder if it doesn't exist
if not os.path.exists(f"datasets/embeddings"):
    os.makedirs("datasets/embeddings")

# Assuming encoder and max_length are defined elsewhere
nights_embeddings = {}
for person_data in time_series_list:
    for key, value in person_data.items():
        if isinstance(value, pd.DataFrame):
            # Pad the DataFrame to ensure consistent shape
            df_padded = pad_sequences([value.values], maxlen=max_length, dtype='float32', padding='post')
            # Predict embeddings
            embeddings = encoder.predict(df_padded)
            # Store embeddings
            nights_embeddings[key] = embeddings.flatten()  # Flatten embeddings if necessary

        if key == 'temp_id':
            # Convert the embeddings dictionary to a DataFrame
            embeddings_df = pd.DataFrame.from_dict(nights_embeddings, orient='index')  # Consider 'orient' based on your data structure
            
            # Generate new column names based on the number of features in each embedding
            new_column_names = [f'embedding_{i}' for i in range(embeddings_df.shape[1])]

            # Rename the DataFrame columns
            embeddings_df.columns = new_column_names

            # Save the DataFrame as a pickle file
            file_name = f"datasets/embeddings_100/embeddings_{value}.pkl"
            embeddings_df.to_pickle(file_name)

            # Reset the embeddings dictionary for the next person
            nights_embeddings = {}