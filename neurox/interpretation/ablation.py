""" Module for ablating neurons using various techniques.

This module provides a set of methods to ablate both layers and individual
neurons from a given set.
"""

def filter_activations_by_layers(
    train_activations, test_activations, filter_layers, rnn_size, num_layers, is_brnn
):
    """
    Filter activations so that they only contain specific layers.

    Useful for performing layer-wise analysis.

    .. warning::
        This function is deprecated and will be removed in future versions.

    Parameters
    ----------
    train_activations : list of numpy.ndarray
        List of *sentence representations* from the train set, where each
        *sentence representation* is a numpy matrix of shape
        ``[NUM_TOKENS x NUM_NEURONS]``. The method assumes that neurons from
        all layers are present, with the number of neurons in every layer given
        by ``rnn_size``
    test_activations : list of numpy.ndarray
        Similar to ``train_activations`` but with sentences from a test set.
    filter_layers : str
        A comma-separated string of the form "f1,f2,f10". "f" indicates a "forward"
        layer while "b" indicates a backword layer in a Bidirectional RNN. If the
        activations are from different kind of model, set ``is_brnn`` to ``False``
        and provide only "f" entries. The number next to "f" is the layer number,
        1-indexed. So "f1" corresponds to the embedding layer and so on.
    rnn_size : int
        Number of neurons in every layer.
    num_layers : int
        Total number of layers in the original model.
    is_brnn : bool
        Boolean indicating if the neuron activations are from a bidirectional model.

    Returns
    -------
    filtered_train_activations : list of numpy.ndarray
        Filtered train activations
    filtered_test_activations : list of numpy.ndarray
        Filtered test activations

    Notes
    -----
    For bidirectional models, the method assumes that the internal structure is
    as follows: forward layer 1 neurons, backward layer 1 neurons, forward layer
    2 neurons ...

    """
    _layers = filter_layers.split(",")

    layer_prefixes = ["f"]
    if is_brnn:
        layer_prefixes = ["f", "b"]

    # FILTER settings
    layers = list(
        range(1, num_layers + 1)
    )  # choose which layers you need the activations
    filtered_train_activations = None
    filtered_test_activations = None

    layers_idx = []
    for brnn_idx, b in enumerate(layer_prefixes):
        for l in layers:
            if "%s%d" % (b, l) in _layers:
                start_idx = brnn_idx * (num_layers * rnn_size) + (l - 1) * rnn_size
                end_idx = brnn_idx * (num_layers * rnn_size) + (l) * rnn_size

                print(
                    "Including neurons from %s%d(#%d to #%d)"
                    % (b, l, start_idx, end_idx)
                )
                layers_idx.append(np.arange(start_idx, end_idx))
    layers_idx = np.concatenate(layers_idx)

    filtered_train_activations = [a[:, layers_idx] for a in train_activations]
    filtered_test_activations = [a[:, layers_idx] for a in test_activations]

    return filtered_train_activations, filtered_test_activations

def keep_specific_neurons(X, neuron_list):
    """
    Filter activations so that they only contain specific neurons.

    .. warning::
        This function is deprecated and will be removed in future versions. Use
        ``interpretation.ablation.filter_activations_keep_neurons`` instead.

    Parameters
    ----------
    X : numpy.ndarray
        Numpy Matrix of size [``NUM_TOKENS`` x ``NUM_NEURONS``]. Usually the
        output of ``interpretation.utils.create_tensors``
    neuron_list : list or numpy.ndarray
        List of neurons to keep

    Returns
    -------
    filtered_X : numpy.ndarray view
        Numpy Matrix of size [``NUM_TOKENS`` x ``len(neuron_list)``]

    """
    return X[:, neuron_list]

def filter_activations_keep_neurons(X, neurons_to_keep):
    """
    Filter activations so that they only contain specific neurons.

    .. note::
        The returned value is a view, so modifying it will modify the original
        matrix.

    Parameters
    ----------
    X : numpy.ndarray
        Numpy Matrix of size [``NUM_TOKENS`` x ``NUM_NEURONS``]. Usually the
        output of ``interpretation.utils.create_tensors``
    neurons_to_keep : list or numpy.ndarray
        List of neurons to keep

    Returns
    -------
    filtered_X : numpy.ndarray view
        Numpy Matrix of size [``NUM_TOKENS`` x ``len(neurons_to_keep)``]

    """
    return X[:, neurons_to_keep]


def filter_activations_remove_neurons(X, neurons_to_remove):
    """
    Filter activations so that they do not contain specific neurons.

    .. note::
        The returned value is a view, so modifying it will modify the original
        matrix.

    Parameters
    ----------
    X : numpy.ndarray
        Numpy Matrix of size [``NUM_TOKENS`` x ``NUM_NEURONS``]. Usually the
        output of ``interpretation.utils.create_tensors``
    neurons_to_remove : list or numpy.ndarray
        List of neurons to remove

    Returns
    -------
    filtered_X : numpy.ndarray view
        Numpy Matrix of size [``NUM_TOKENS`` x ``NUM_NEURONS - len(neurons_to_remove)``]

    """
    neurons_to_keep = np.arange(X.shape[1])
    neurons_to_keep[neurons_to_remove] = -1
    neurons_to_keep = np.where(neurons_to_keep != -1)[0]
    return X[:, neurons_to_keep]


def zero_out_activations_keep_neurons(X, neurons_to_keep):
    """
    Mask all neurons activations with zero other than specified neurons.

    Parameters
    ----------
    X : numpy.ndarray
        Numpy Matrix of size [``NUM_TOKENS`` x ``NUM_NEURONS``]. Usually the
        output of ``interpretation.utils.create_tensors``
    neurons_to_keep : list or numpy.ndarray
        List of neurons to not mask

    Returns
    -------
    filtered_X : numpy.ndarray
        Numpy Matrix of size [``NUM_TOKENS`` x ``NUM_NEURONS``]

    """
    _X = np.zeros_like(X)

    _X[:, neurons_to_keep] = X[:, neurons_to_keep]

    return _X


def zero_out_activations_remove_neurons(X, neurons_to_remove):
    """
    Mask specific neuron activations with zero.

    Parameters
    ----------
    X : numpy.ndarray
        Numpy Matrix of size [``NUM_TOKENS`` x ``NUM_NEURONS``]. Usually the
        output of ``interpretation.utils.create_tensors``
    neurons_to_remove : list or numpy.ndarray
        List of neurons to mask

    Returns
    -------
    filtered_X : numpy.ndarray
        Numpy Matrix of size [``NUM_TOKENS`` x ``NUM_NEURONS``]

    """
    _X = np.copy(X)

    _X[:, neurons_to_remove] = 0

    return _X