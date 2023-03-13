
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
