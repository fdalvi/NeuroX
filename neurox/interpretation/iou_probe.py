"""Module for IoU method to rank neurons

This module implements the Intersection over Union method to rank neurons based on their activation values and the true class.

.. seealso::
        Mu, J., & Andreas, J. (2020). Compositional explanations of neurons. Advances in Neural Information Processing Systems, 33, 17153-17163.

"""
import numpy as np
from sklearn.metrics import average_precision_score


def get_neuron_ordering(X_train, y_train, threshold=0.05):
    """
    Returns a list of top neurons w.r.t a tag e.g. noun

    Parameters
    ----------
    X_train : numpy.ndarray
        Numpy Matrix of size [``NUM_TOKENS`` x ``NUM_NEURONS``]. Usually the
        output of ``interpretation.utils.create_tensors``
    y_train : numpy.ndarray
        Numpy Vector of size [``NUM_TOKENS``] with class labels for each input
        token. Usually the output of ``interpretation.utils.create_tensors``.
    threshold : float
        The minimum absolute activation value below which the neuron is ignored for ranking purposes

    Returns
    -------
    ranking : list
        list of ``NUM_NEURONS`` neuron indices, in decreasing order of importance.

    """
    X_train[np.abs(X_train) < threshold] = 0
    scores = []
    for i in range(X_train.shape[1]):
        scores.append(average_precision_score(y_train, X_train[:, i]))
    scores = np.array(scores)
    ranking = np.argsort(scores)[::-1]
    return ranking
