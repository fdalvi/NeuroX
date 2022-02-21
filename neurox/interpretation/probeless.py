"""Module for Probeless method

This module extracts neuron ranking for a label/tag (e.g Verbs) or for the entire property set (e.g Part of speech).

.. seealso::
        `Antverg, Omer and Belinkov, Yonatan "On The Pitfalls Of Analyzing Idividual Neurons in Language Models." In Proceedings of the 10th International Conference on Learning Representations (ICLR).`_
"""
import numpy as np
import torch
import itertools
import sys

from . import ablation


def _get_mean_vectors(X_t, y_t):
    
    embeddings_by_labels = {}
    average_embeddings_by_label = {}
    rankings_per_label = {}

    for words_idx in range(y_t.size):
        label = y_t[words_idx]
        if (label in embeddings_by_labels):
            embeddings_by_labels[label].append(X_t[words_idx])
        else:
            vector = [] 
            vector.append(X_t[words_idx])
            embeddings_by_labels[label] = vector
        
    for k, v in embeddings_by_labels.items():
        average_embeddings_by_label[k] = np.mean(v, axis=0)

    avg_embeddings = list(average_embeddings_by_label.values())
    
    return avg_embeddings, average_embeddings_by_label

def _get_overall_ranking(mean_vectors):
    
    overall = np.zeros_like(mean_vectors[0])
    
    for couple in itertools.combinations(mean_vectors, 2):
        overall += np.abs(couple[0] - couple[1])

    ranking = np.argsort(overall)[::-1].tolist()
    return ranking

def _get_tag_wise_ranking(mean_vectors_by_label, tag):

    qz = mean_vectors_by_label[tag]
    summation = np.abs(np.subtract(qz, qz))

    for c, qzz in mean_vectors_by_label.items():
        
        if (tag != c):
            summation = np.add(summation, np.abs(np.subtract(qz,qzz)))
    
    ranking = np.argsort(summation)[::-1].tolist()
    return summation, ranking 


def _get_layer_vector(X_t, y_t, NUM_LAYERS, layer=-1):

    if (layer == -1):
        layer_X_t = X_t

    elif (layer <= NUM_LAYERS):
        layer_X_t = ablation.filter_activations_by_layers(X_t, [layer], NUM_LAYERS)
        
    else:
        raise ValueError(f"Invalid layer number {layer}")

    return layer_X_t

def get_top_overall_neurons(X_t, y_t, NUM_LAYERS, layer=-1):

    """
    Returns a list of top neurons w.r.t the overall task e.g. POS

    Parameters
    ----------
    X_t : numpy.ndarray
    Numpy Matrix of size [``NUM_TOKENS`` x ``NUM_NEURONS``]. Usually the 
    output of ``interpretation.utils.create_tensors``
    y_t : numpy.ndarray
    Numpy Vector of size [``NUM_TOKENS``] with class labels for each input
    token. Usually the output of ``interpretation.utils.create_tensors``.
    NUM_LAYERS : int
    Total number of layers in the network
    layer : int, optional
    layer number for which ranking needs to be extracted, -1 for entire network


    Returns
    -------
    ranking : list

    """
    
    layer_X_t = _get_layer_vector(X_t, y_t, NUM_LAYERS, layer)
    avg_embeddings, average_embeddings_by_label = _get_mean_vectors(layer_X_t, y_t)
    ranking = _get_overall_ranking(avg_embeddings)
    
    return ranking

def get_top_neurons_for_tag(X_t, y_t , tag, NUM_LAYERS, layer=-1):

    """
    Returns a list of top neurons w.r.t a tag e.g. noun

    Parameters
    ----------
    X_t : numpy.ndarray
    Numpy Matrix of size [``NUM_TOKENS`` x ``NUM_NEURONS``]. Usually the 
    output of ``interpretation.utils.create_tensors``
    y_t : numpy.ndarray
    Numpy Vector of size [``NUM_TOKENS``] with class labels for each input
    token. Usually the output of ``interpretation.utils.create_tensors``.
    tag : int 
    tag for which rankings are extracted
    NUM_LAYERS : int
    Total number of layers in the network
    layer : int, optional
    layer number for which ranking needs to be extracted, -1 for entire network

    Returns
    -------
    ranking : list

    """

    layer_X_t = _get_layer_vector(X_t, y_t, NUM_LAYERS, layer)
    avg_embeddings, average_embeddings_by_label = _get_mean_vectors(layer_X_t, y_t)
    summation, ranking = _get_tag_wise_ranking(average_embeddings_by_label, tag)

    return ranking

def get_top_neurons_for_all_tags(X_t, y_t, NUM_LAYERS, layer=-1):


    """
    Returns a dictionary of tags along with top neurons for each tag
    Returns a list of overall ranking

    Parameters
    ----------
    X_t : numpy.ndarray
    Numpy Matrix of size [``NUM_TOKENS`` x ``NUM_NEURONS``]. Usually the 
    output of ``interpretation.utils.create_tensors``
    y_t : numpy.ndarray
    Numpy Vector of size [``NUM_TOKENS``] with class labels for each input
    token. Usually the output of ``interpretation.utils.create_tensors``.
    NUM_LAYERS : int
    Total number of layers in the network
    layer : int, optional
    layer number for which ranking needs to be extracted, -1 for entire network


    Returns
    -------
    ranking : list
    overall_ranking : dict

    """
    
    ranking_per_tag = {}
    layer_X_t = _get_layer_vector(X_t, y_t, NUM_LAYERS, layer)
    avg_embeddings, average_embeddings_by_label = _get_mean_vectors(layer_X_t, y_t)

    overall = np.zeros_like(layer_X_t[0])

    for c, qz in average_embeddings_by_label.items():

        summation, ranking = _get_tag_wise_ranking(average_embeddings_by_label, c)
        overall = np.add(overall, summation)
        ranking_per_tag [c] = ranking

    overall_ranking = np.argsort(overall)[::-1].tolist()

    return overall_ranking, ranking_per_tag








