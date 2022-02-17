"""
Module that wraps around several standard metrics
"""
import numpy as np

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef as mcc, f1_score


def _numpyfy(x):
    """
    Internal helper function to convert standard lists into numpy arrays.

    Parameters
    ----------
    x : list or numpy.ndarray
        A list of numbers

    Returns
    -------
    x : numpy.ndarray
        Numpy array with the original numbers

    """
    if isinstance(x, np.ndarray):
        return x
    return np.array(x)


def accuracy(preds, labels):
    """
    Accuracy.

    Parameters
    ----------
    preds : list or numpy.ndarray
        A list of predictions from a model
    labels : list or numpy.ndarray
        A list of ground truth labels with the same number of elements as
        ``preds``

    Returns
    -------
    accuracy : float
        Accuracy of the model
    """
    preds = _numpyfy(preds)
    labels = _numpyfy(labels)
    return (preds == labels).mean()

import json
def f1(preds, labels):
    """
    F-Score or F1 score.

    .. note::
        The implementation from ``sklearn.metrics`` is used to compute the score.

    Parameters
    ----------
    preds : list or numpy.ndarray
        A list of predictions from a model
    labels : list or numpy.ndarray
        A list of ground truth labels with the same number of elements as
        ``preds``

    Returns
    -------
    f1_score : float
        F-Score of the model
    """
    preds = _numpyfy(preds)
    labels = _numpyfy(labels)
    return f1_score(y_true=labels, y_pred=preds)


def accuracy_and_f1(preds, labels):
    """
    Mean of Accuracy and F-Score.

    .. note::
        The implementation from ``sklearn.metrics`` is used to compute the
        F-Score.

    Parameters
    ----------
    preds : list or numpy.ndarray
        A list of predictions from a model
    labels : list or numpy.ndarray
        A list of ground truth labels with the same number of elements as
        ``preds``

    Returns
    -------
    acc_f1_mean : float
        Mean of Accuracy and F-Score of the model
    """
    preds = _numpyfy(preds)
    labels = _numpyfy(labels)
    acc = accuracy(preds, labels)
    f1_s = f1_score(y_true=labels, y_pred=preds)
    return (acc + f1_s) / 2


def pearson(preds, labels):
    """
    Pearson's correlation coefficient

    .. note::
        The implementation from ``scipy.stats`` is used to compute the score.

    Parameters
    ----------
    preds : list or numpy.ndarray
        A list of predictions from a model
    labels : list or numpy.ndarray
        A list of ground truth labels with the same number of elements as
        ``preds``

    Returns
    -------
    pearson_score : float
        Pearson's correlation coefficient of the model
    """
    preds = _numpyfy(preds)
    labels = _numpyfy(labels)
    return pearsonr(preds, labels)[0]


def spearman(preds, labels):
    """
    Spearman correlation coefficient

    .. note::
        The implementation from ``scipy.stats`` is used to compute the score.

    Parameters
    ----------
    preds : list or numpy.ndarray
        A list of predictions from a model
    labels : list or numpy.ndarray
        A list of ground truth labels with the same number of elements as
        ``preds``

    Returns
    -------
    spearman_score : float
        Spearman correlation coefficient of the model
    """
    preds = _numpyfy(preds)
    labels = _numpyfy(labels)
    return spearmanr(preds, labels)[0]


def pearson_and_spearman(preds, labels):
    """
    Mean of Pearson and Spearman correlation coefficients.

    .. note::
        The implementation from ``scipy.stats`` is used to compute the scores.

    Parameters
    ----------
    preds : list or numpy.ndarray
        A list of predictions from a model
    labels : list or numpy.ndarray
        A list of ground truth labels with the same number of elements as
        ``preds``

    Returns
    -------
    pearson_spearman_mean : float
        Mean of Pearson and Spearman correlation coefficients of the model
    """
    preds = _numpyfy(preds)
    labels = _numpyfy(labels)
    return (pearson(preds, labels) + spearman(preds, labels)) / 2


def matthews_corrcoef(preds, labels):
    """
    Matthew's correlation coefficient

    .. note::
        The implementation from ``sklearn.metrics`` is used to compute the score.

    Parameters
    ----------
    preds : list or numpy.ndarray
        A list of predictions from a model
    labels : list or numpy.ndarray
        A list of ground truth labels with the same number of elements as
        ``preds``

    Returns
    -------
    mcc_score : float
        Matthew's correlation coefficient of the model
    """
    preds = _numpyfy(preds)
    labels = _numpyfy(labels)
    return mcc(preds, labels)


def compute_score(preds, labels, metric):
    """
    Utility function to compute scores using several metrics.

    Parameters
    ----------
    preds : list or numpy.ndarray
        A list of predictions from a model
    labels : list or numpy.ndarray
        A list of ground truth labels with the same number of elements as
        ``preds``
    metric : str
        One of ``accuracy``, ``f1``, ``accuracy_and_f1``, ``pearson``,
        ``spearman``, ``pearson_and_spearman`` or ``matthews_corrcoef``.

    Returns
    -------
    score : float
        Score of the model with the chosen metric
    """
    if metric == 'accuracy':
        return accuracy(preds, labels)
    elif metric == 'f1':
        return f1(preds, labels)
    elif metric == 'accuracy_and_f1':
        return accuracy_and_f1(preds, labels)
    elif metric == 'pearson':
        return pearson(preds, labels)
    elif metric == 'spearman':
        return spearman(preds, labels)
    elif metric == 'pearson_and_spearman':
        return pearson_and_spearman(preds, labels)
    elif metric == 'matthews_corrcoef':
        return matthews_corrcoef(preds, labels)