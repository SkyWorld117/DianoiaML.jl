import pickle

def save_model(variable, path):
    with open(path, 'wb') as f:
        pickle.dump(variable, f)

def load_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)