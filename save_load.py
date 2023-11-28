import numpy as np
import pickle
import os

def save_pkl(arr_vector, name):
    # Create embedding folder if it doesnt exist
    if not os.path.exists("embeddings"):
        os.makedirs("embeddings")
    
    # Save the embeddings array to a file using pickle
    file_path = os.path.join("embeddings", name)
    with open(file_path, 'wb') as file:
        pickle.dump(arr_vector, file)

def load_pkl(name):
    # Load the embeddings from a file using pickle
    file_path = os.path.join("embeddings", name)
    with open(file_path, 'rb') as file:
        array = pickle.load(file)
        return array