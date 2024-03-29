import numpy as np


def accuracy_score(y_true, y_pred):
    """
    Classification performance metric that computes the accuracy of y_true
    and y_pred.

    Parameters
    ----------
    y_true: numpy.ndarray (n_samples,)
        Ground truth correct labels.
    y_pred: numpy.ndarray  (n_samples,)
        Estimated target values.

    Returns
    -------
    accuracy (float) 
        Accuracy score.
    """
    accuracy = (y_true==y_pred).sum() / len(y_true)
    return accuracy


def mse(y_true, y_pred):
    """
    Mean squared error regression loss function.
    Parameters

    :param numpy.array y_true: array-like of shape (n_samples,)
        Ground truth (correct) target values.
    :param numpy.array y_pred: array-like of shape (n_samples,)
        Estimated target values.
    :returns: loss (float) A non-negative floating point value (the best value is 0.0).
    
    Note: some implementations of the MSE consider additionaly a division by 2
          to obtain a `cleaner` derivative allowing to cancel the factor '2' 
          (see mse_prime). 
          Computationally, they are equivalent as both require a bit shift.
    """
    return np.mean(np.power(y_true-y_pred, 2))


def mse_prime(y_true, y_pred):
    """ The derivative of the MSE.
     
    :param numpy.array y_true: array-like of shape (n_samples,)
        Ground truth (correct) target values.
    :param numpy.array y_pred: array-like of shape (n_samples,)
        Estimated target values.
    :returns: the derivative of the MSE irt the prediction
    
    Note: To avoid the additional multiplication by -1 just swap
          the y_pred and y_true.
    """
    return 2*(y_pred-y_true)/y_true.size


def rmse(y_true, y_pred):
    """Rooted MSE

    :param numpy.array y_true: array-like of shape (n_samples,)
        Ground truth (correct) target values.
    :param numpy.array y_pred: array-like of shape (n_samples,)
        Estimated target values.
    :returns: RMSE
    """
    return np.sqrt(mse(y_true,y_pred))


def rmse_prime(y_true, y_pred):
    """Derivative of RMSE

    :param numpy.array y_true: array-like of shape (n_samples,)
        Ground truth (correct) target values.
    :param numpy.array y_pred: array-like of shape (n_samples,)
        Estimated target values.
    :returns: the derivative of the RMSE irt the prediction
    """
    return (y_pred-y_true)/(rmse(y_true,y_pred)*y_true.size) 


def cross_entropy(y_true, y_pred):
    """Cross entropy

    :param numpy.array y_true: array-like of shape (n_samples,)
        Ground truth (correct) target values.
    :param numpy.array y_pred: array-like of shape (n_samples,)
        Estimated target values.
    :returns: cross entropy score
    """
    m = y_pred.shape[0]
    return -(y_true * np.log(y_pred)).sum()/m


def cross_entropy_prime(y_true, y_pred):
    """Cross entropy derivative

    :param numpy.array y_true: array-like of shape (n_samples,)
        Ground truth (correct) target values.
    :param numpy.array y_pred: array-like of shape (n_samples,)
        Estimated target values.
    :returns: cross entropy derivative
    """
    m = y_pred.shape[0]
    return (y_pred - y_true)/m


def r2_score(y_true, y_pred):
    """
    R^2 regression score function.
        R^2 = 1 - SS_res / SS_tot
    where SS_res is the residual sum of squares and SS_tot is the total
    sum of squares.

    :param numpy.array y_true : array-like of shape (n_samples,) Ground truth (correct) target values.
    :param numpy.array y_pred : array-like of shape (n_samples,) Estimated target values.
    :returns: score (float) R^2 score.
    """
    # Residual sum of squares.
    numerator = ((y_true - y_pred) ** 2).sum(axis=0)
    # Total sum of squares.
    denominator = ((y_true - np.average(y_true, axis=0)) ** 2).sum(axis=0)
    # R^2.
    score = 1 - numerator / denominator
    return score
