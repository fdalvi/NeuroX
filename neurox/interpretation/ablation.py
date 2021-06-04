""" Module for ablating neurons using various techniques.

This module provides a set of methods to ablate both layers and individual
neurons from a given set.
"""
import numpy as np

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

def filter_activations_by_layers(
    X, layers_to_keep, num_layers, bidirectional_filtering="none"
):
    """
    Filter activations so that they only contain specific layers.

    Useful for performing layer-wise analysis.

    Parameters
    ----------
    X : numpy.ndarray
        Numpy Matrix of size [``NUM_TOKENS`` x ``NUM_NEURONS``]. Usually the
        output of ``interpretation.utils.create_tensors``
    layers_to_keep : list or numpy.ndarray
        List of layers to keep. Layers are 0-indexed
    num_layers : int
        Total number of layers in the original model.
    bidirectional_filtering : str
        Can be either "none" (Default), "forward" or "backward". Useful if the
        model being analyzed is bi-directional and only layers in a certain
        direction need to be analyzed.
    Returns
    -------
    filtered_X : numpy.ndarray
        Numpy Matrix of size [``NUM_TOKENS`` x ``NUM_NEURONS_PER_LAYER * NUM_LAYERS``]
        The second dimension is doubled if the original model is bidirectional
        and no filtering is done.

    Notes
    -----
    For bidirectional models, the method assumes that the internal structure is
    as follows: forward layer 0 neurons, backward layer 0 neurons, forward layer
    0 neurons ...

    """
    bidirectional_filtering = bidirectional_filtering.lower()
    assert bidirectional_filtering in ["none", "forward", "backward"]

    neurons_to_keep = []
    for layer in layers_to_keep:
        if bidirectional_filtering == "none":
            num_neurons_per_layer = X.shape[1] // num_layers
            start = layer * num_neurons_per_layer
            end = start + num_neurons_per_layer
        elif bidirectional_filtering == "forward":
            num_neurons_per_layer = X.shape[1] // (num_layers * 2)
            start = layer * (num_neurons_per_layer * 2)
            end = start + num_neurons_per_layer
        elif bidirectional_filtering == "backward":
            num_neurons_per_layer = X.shape[1] // (num_layers * 2)
            start = layer * num_neurons_per_layer * 2 + num_neurons_per_layer
            end = start + num_neurons_per_layer

        neurons_to_keep.append(list(range(start, end)))
    neurons_to_keep = np.concatenate(neurons_to_keep)

    return filter_activations_keep_neurons(X, neurons_to_keep)