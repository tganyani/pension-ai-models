import numpy as np

def add_noise(X):
    noise = np.random.normal(0, 0.01, X.shape)
    return X + noise