from typing import Tuple, Sequence

import numpy as np
import pandas as pd
from scipy.stats import mode


class Dataset:
    def __init__(self, X: np.ndarray, 
                 y: np.ndarray = None, 
                 features: Sequence[str] = None,
                 discrete_features: Sequence[str] = None,
                 numeric_features: Sequence[str] = None, 
                 label: str = None):
        """
        Dataset represents a machine learning tabular dataset.

        Parameters
        ----------
        X: numpy.ndarray (n_samples, n_features)
            The feature matrix
        y: numpy.ndarray (n_samples, 1)
            The label vector
        features: list of str (n_features)
            The feature names
        discrete_features : list of str (n_features)
            The features names of discrete features
        numeric_features : list of str (n_features)
            The features names of numeric features
        label: str (1)
            The label name
        """
        if X is None:
            raise ValueError("X cannot be None")

        if features is None:
            features = [str(i) for i in range(X.shape[1])]
        else:
            features = list(features)

        if discrete_features is None and numeric_features is None:
            raise ValueError("At least one of discrete_features or numeric_features must be provided")
        elif discrete_features is None:
            self.discrete_mask = np.zeros(X.shape[1], dtype=bool)
            self.discrete_mask[numeric_features] = False
        elif numeric_features is None:
            self.discrete_mask = np.zeros(X.shape[1], dtype=bool)
            self.discrete_mask[discrete_features] = True
        else:
            self.discrete_mask = np.zeros(X.shape[1], dtype=bool)
            self.discrete_mask[discrete_features] = True
            self.discrete_mask[numeric_features] = False

        if y is not None and label is None:
            label = "y"

        self.X = X
        self.y = y
        self.features = features
        self.label = label

    def get_discrete_mask(self) -> np.ndarray:
        """
        Returns the boolean mask indicating which columns in X correspond to discrete features.

        Returns
        -------
        numpy.ndarray (n_features,)
            Boolean mask indicating which columns in X correspond to discrete features
        """
        return self.discrete_mask

    def get_numeric_mask(self) -> np.ndarray:
        """
        Returns the boolean mask indicating which columns in X correspond to numeric features.

        Returns
        -------
        numpy.ndarray (n_features,)
            Boolean mask indicating which columns in X correspond to numeric features
        """
        return ~self.discrete_mask

    def get_discrete_X(self) -> np.ndarray:
        """
        Returns the subset of X corresponding to the discrete features.

        Returns
        -------
        numpy.ndarray (n_samples, n_discrete_features)
            Subset of X corresponding to the discrete features
        """
        return self.X[:, self.discrete_mask]

    def get_numeric_X(self) -> np.ndarray:
        """
        Returns the subset of X corresponding to the numeric features.

        Returns
        -------
        numpy.ndarray (n_samples, n_numeric_features)
            Subset of X corresponding to the numeric features
        """
        return self.X[:, ~self.discrete_mask]

    def shape(self) -> Tuple[int, int]:
        """
        Returns the shape of the dataset.

        Returns
        -------
        tuple (n_samples, n_features)
        """
        return self.X.shape

    def has_label(self) -> bool:
        """
        Returns True if the dataset has a label.

        Returns
        -------
        bool
        """
        return self.y is not None

    def get_classes(self) -> np.ndarray:
        """
        Returns the unique classes in the dataset.

        Returns
        -------
        numpy.ndarray (n_classes)
        """
        if self.y is None:
            raise ValueError("Dataset does not have a label")
        return np.unique(self.y)

    def get_mean(self) -> np.ndarray:
        """
        Returns the mean of each numeric feature.

        Returns
        -------
        numpy.ndarray (n_numeric_features,)
            Array containing the mean of each numeric feature. If a feature is discrete or if its mean cannot be
            computed, its corresponding value in the array is NaN.
        """
        numeric_mask = self.get_numeric_mask()
        means = np.full(numeric_mask.sum(), np.nan)
        numeric_indices = np.where(numeric_mask)[0]

        for i in enumerate(self.features[numeric_mask]):
            try:
                means[i] = np.nanmean(self.X[:, numeric_indices[i]])
            except TypeError:
                # This feature is discrete or contains non-numeric values
                pass

        return means
    
    def get_variance(self) -> np.ndarray:
        """
        Returns the variance of each numeric feature.

        Returns
        -------
        numpy.ndarray (n_features)
            Array containing the variance of each numeric feature. If a feature is discrete or if its variance cannot be
            computed, its corresponding value in the array is NaN.
        """
        numeric_mask = self.get_numeric_mask()
        vars = np.full(numeric_mask.sum(), np.nan)
        numeric_indices = np.where(numeric_mask)[0]

        for i in enumerate(self.features[numeric_mask]):
            try:
                vars[i] = np.nanvar(self.X[:, numeric_indices[i]])
            except TypeError:
                # This feature is discrete or contains non-numeric values
                pass

        return vars

    def get_median(self) -> np.ndarray:
        """
        Returns the median of each numeric feature.

        Returns
        -------
        numpy.ndarray (n_features)
            Array containing the median of each numeric feature. If a feature is discrete or if its median cannot be
            computed, its corresponding value in the array is NaN.
        """
        numeric_mask = self.get_numeric_mask()
        median = np.full(numeric_mask.sum(), np.nan)
        numeric_indices = np.where(numeric_mask)[0]

        for i in enumerate(self.features[numeric_mask]):
            try:
                median[i] = np.nanmedian(self.X[:, numeric_indices[i]])
            except TypeError:
                # This feature is discrete or contains non-numeric values
                pass

        return vars
    
    def get_max(self) -> np.ndarray:
        """
        Returns the maximum value of each numeric feature.

        Returns
        -------
        numpy.ndarray (n_features,)
            Array containing the maximum value of each numeric feature. If a feature is not numeric or if its maximum
            cannot be computed, its corresponding value in the array is NaN.
        """
        numeric_mask = self.get_numeric_mask()
        max_val = np.full(numeric_mask.sum(), np.nan)
        numeric_indices = np.where(numeric_mask)[0]

        for i, idx in enumerate(numeric_indices):
            try:
                max_val[i] = np.nanmax(self.X[:, idx])
            except ValueError:
                # This feature is not numeric or contains non-numeric values
                pass

        return max_val
    
    def get_min(self) -> np.ndarray:
        """
        Returns the minimum value of each numeric feature.

        Returns
        -------
        numpy.ndarray (n_features,)
            Array containing the minimum value of each numeric feature. If a feature is not numeric or if its minimum
            cannot be computed, its corresponding value in the array is NaN.
        """
        numeric_mask = self.get_numeric_mask()
        min_val = np.full(numeric_mask.sum(), np.nan)
        numeric_indices = np.where(numeric_mask)[0]

        for i, idx in enumerate(numeric_indices):
            try:
                min_val[i] = np.nanmin(self.X[:, idx])
            except ValueError:
                # This feature is not numeric or contains non-numeric values
                pass
            
        return min_val


    def summary(self) -> pd.DataFrame:
        """
        Returns a summary of the dataset

        Returns
        -------
        pandas.DataFrame (n_features, 5)
        """
        data = {
            "mean": self.get_mean(),
            "median": self.get_median(),
            "var": self.get_variance(),
            "min": self.get_min(),
            "max": self.get_max()
        }
        return pd.DataFrame.from_dict(data, orient="index", columns=self.features)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, label: str = None):
        """
        Creates a Dataset object from a pandas DataFrame.

        Parameters
        ----------
        df: pandas.DataFrame
            The DataFrame
        label: str
            The label name

        Returns
        -------
        Dataset
        """
        if label:
            X = df.drop(label, axis=1).to_numpy()
            y = df[label].to_numpy()
        else:
            X = df.to_numpy()
            y = None

        features = df.columns.tolist()
        return cls(X, y, features=features, label=label)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Converts the dataset to a pandas DataFrame.

        Returns
        -------
        pandas.DataFrame
        """
        if self.y is None:
            return pd.DataFrame(self.X, columns=self.features)
        else:
            df = pd.DataFrame(self.X, columns=self.features)
            df[self.label] = self.y
            return df

    @classmethod
    def from_random(cls,
                    n_samples: int,
                    n_features: int,
                    n_classes: int = 2,
                    features: Sequence[str] = None,
                    label: str = None):
        """
        Creates a Dataset object from random data.

        Parameters
        ----------
        n_samples: int
            The number of samples
        n_features: int
            The number of features
        n_classes: int
            The number of classes
        features: list of str
            The feature names
        label: str
            The label name
        Returns
        -------
        Dataset
        """
        X = np.random.rand(n_samples, n_features)
        y = np.random.randint(0, n_classes, n_samples)
        return cls(X, y, features=features, label=label)
    
    def replace_nulls(self, method='mean'):
        """
        Replace all NaN values of each numeric feature using the specified method.

        Parameters
        ----------
        method : str or callable, optional (default='mean')
            Method of replacing
        """
        numeric_mask = self.get_numeric_mask()

        if method == 'mean':
            means = np.nanmean(self.get_numeric_X(), axis=0)
            self.X[:, numeric_mask] = np.where(np.isnan(self.X[:, numeric_mask]), means, self.X[:, numeric_mask])
        elif method == 'median':
            medians = np.nanmedian(self.get_numeric_X(), axis=0)
            self.X[:, numeric_mask] = np.where(np.isnan(self.X[:, numeric_mask]), medians, self.X[:, numeric_mask])
        else:
            raise ValueError("Invalid method: {}".format(method))

    def count_nulls(self) -> np.ndarray:
        """
        Counts the number of null values in each feature of X.

        Returns
        -------
        numpy.ndarray (n_features,)
            Array containing the number of null values in each feature.
        """
        return np.sum(np.isnan(self.X), axis=0)
    