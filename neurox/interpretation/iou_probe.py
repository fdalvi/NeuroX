from sklearn.metrics import average_precision_score
import numpy as np
def get_neuron_ordering(X_train, y_train, threshold):
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
        the maximal ignored values in X_train

    Returns
    -------
    ranking : list
        list of ``NUM_NEURONS`` neuron indices, in decreasing order of importance.

    """
    X_train[np.abs(X_train)< threshold] = 0
    score = []
    for i in range(X_train.shape[1]):
        score.append(average_precision_score(y_train,X_train[:,i]))
    score = np.array(score)
    ranking = np.argsort(score)[::-1]
    return ranking

