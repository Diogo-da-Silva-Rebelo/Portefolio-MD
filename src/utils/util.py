from data.dataset import Dataset
import numpy as np
import pandas as pd

def train_test_split(dataset: Dataset, test_size=0.3):
    nrows = dataset.get_X().shape[0]
    test_size = int(test_size * nrows)
    train_size = nrows - test_size
    idx = np.arange(nrows)
    np.random.shuffle(idx)
    train_idx = idx[:train_size]
    test_idx = idx[train_size:]

    # Get the X and Y attributes of the dataset object
    X = dataset.get_X()
    y = dataset.get_y()
    features = dataset.get_features()
    label = dataset.get_label()

    # Split the X and Y data into training and test sets
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Define the discrete and numeric features
    data = pd.DataFrame(X)
    discretes = [str(col) for col in data.columns if data[col].dtype in ['object', 'category', 'bool']]

    # Create the training and test datasets
    train = Dataset(X=X_train, y=y_train, features=features, discrete_features = discretes, label=label)
    test = Dataset(X=X_test, y=y_test, features=features, discrete_features = discretes, label=label)
 
    return train, test
