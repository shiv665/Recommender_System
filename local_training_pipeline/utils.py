import pickle
import os

def print_header(title):
    print("=" * 60)
    print(f" {title}")
    print("=" * 60)

def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
