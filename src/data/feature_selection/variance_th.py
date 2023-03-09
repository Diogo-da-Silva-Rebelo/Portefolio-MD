import numpy as np
from copy import copy

from transformer import Transformer
from data.dataset import Dataset

class VarianceThreshold(Transformer):

    def __init__(self, threshold = 0) -> None:
        if threshold < 0:
            raise ValueError("The thershold must be a non-negative value.")
        self.threshold = threshold

    def fit(self, dataset: Dataset) -> None:

        if not dataset.all_numeric:
            raise ValueError("Theres is not encoded data. Consider using an encoder.")
        else:
            X = dataset.X
            self._var = np.var(X, axis=0)

    def transform(self, dataset, inline=False) -> Dataset:

        if not dataset.all_numeric:
            raise ValueError("Theres is not encoded data. Consider using an encoder.")
        else:
            X = dataset.X
            features_mask = np.where(self._var > self.threshold)[0]
            X_trans = X[:, features_mask]
            features_names = np.array(dataset.features)[features_mask]
            numeric_features = list(features_names)

            if inline:
                dataset.X = X_trans
                dataset.features = features_names
                dataset.numeric_features = numeric_features
                return dataset
            else:
                return Dataset(X=copy(X_trans),
                               y=copy(dataset.y),
                               features=list(features_names),
                               numeric_features = numeric_features, 
                               label=copy(dataset.label))
