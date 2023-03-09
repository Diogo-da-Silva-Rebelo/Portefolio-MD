import numpy as np
from data.dataset import Dataset
from typing import Tuple, Union
from scipy import stats


def f_regress(dataset: Dataset) -> Union[Tuple[np.ndarray, np.ndarray],
                                                Tuple[float, float]]:
    """Scoring function for regressions
    
    F-test for regression
    The null hypotesis, H0: all coefficientes are zero, in other words,
    the model does not have predictive capabilities.

    Parameters
    ----------
    daatset: Dataset
        A labeled dataset

    Returns
    -------
    (F, p): scores and p-values (numpy.ndarray tuple)
        Tupple of numpy.ndarrays
    """

    X = dataset.X
    y = dataset.y
    correlation_coefficient = np.array([stats.pearsonr(X[:, i], y)[0] for i in range(X.shape[1])])
    deg_of_freedom = y.size - 2
    corr_coef_squared = correlation_coefficient ** 2
    F = corr_coef_squared / (1 - corr_coef_squared) * deg_of_freedom
    p = stats.f.sf(F, 1, deg_of_freedom)
    return F, p