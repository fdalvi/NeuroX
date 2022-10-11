"""Module for IoU method to rank neurons

This module implements the Intersection over Union method to rank neurons based on their activation values and the true class.

.. seealso::
        Seil Na and Yo Joong Choe and Dong-Hyun Lee and Gunhee Kim. Discovery of Natural Language Concepts in Individual Units of CNNs. International Conference on Learning Representations, 2019.

"""
import numpy as np


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
    mu_plus = np.mean(X_train[y_train == 1], axis=0)
    mu_minus = np.mean(X_train[y_train == 0], axis=0)
    max_activations = np.max(X_train, axis=0)
    min_activations = np.min(X_train, axis=0)
    selectivity = (mu_plus - mu_minus) / (max_activations - min_activations + 1e-7)
    ranking = np.argsort(np.abs(selectivity))[::-1]
    return ranking
