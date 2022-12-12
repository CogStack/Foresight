import pickle
import os

def load(path):
    return pickle.load(open(path, 'rb'))

def dump(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pickle.dump(data, open(path, 'wb'))
