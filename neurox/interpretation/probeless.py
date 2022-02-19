"""Module for Probeless method
This module extracts neuron ranking for an attribute or for the entire property set .
.. see also::
        `Antverg, Omer and Belinkov, Yonatan "On The Pitfalls Of Analyzing Idividual Neurons in Language Models." In Proceedings of the 10th International Conference on Learning Representations (ICLR).
"""
import numpy as np
import torch
import itertools
import sys
import torch.nn as nn
from torch.autograd import Variable

from . import metrics
from . import utils
from . import ablation


def get_mean_vectors(X_t, Y_t):
    
    embeddings_by_labels = {}
    average_embeddings_by_label = {}
    rankings_per_label = {}

    for words in range(Y_t.size):
        label = Y_t[words]
        if (label in embeddings_by_labels):
            embeddings_by_labels[label].append(X_t[words])
        else:
            vector = [] 
            vector.append(X_t[words])
            embeddings_by_labels[label] = vector
        
    for k, v in embeddings_by_labels.items():
        average_embeddings_by_label[k] = np.mean(v, axis=0)

    avg_embeddings = list(average_embeddings_by_label.values())
    
    return avg_embeddings, average_embeddings_by_label

def get_overall_ranking(mean_vectors):
    
    overall = np.zeros_like(mean_vectors[0])
    
    for couple in itertools.combinations(mean_vectors, 2):
        overall += np.abs(couple[0] - couple[1])

    ranking = np.argsort(overall)[::-1].tolist()
    return ranking

def get_tag_wise_ranking(mean_vectors_by_label, tag):

    qz = mean_vectors_by_label[tag]
    summation = np.abs(np.subtract(qz, qz))

    for c, qzz in mean_vectors_by_label.items():
        
        if (tag != c):
            summation = np.add(summation, np.abs(np.subtract(qz,qzz)))
    
    ranking = np.argsort(summation)[::-1].tolist()
    return summation, ranking 


def get_layer_vector(X_a, y_t, NUM_LAYERS, layer=-1):

    if (layer == -1):
        layer_X_train = X_a

    elif (layer <= NUM_LAYERS):
        layer_X_train = ablation.filter_activations_by_layers(X_a, [layer], NUM_LAYERS)
        
    else:
        print ("Invalid layer number",layer)
        sys.exit(1)

    return layer_X_train

def overall_ranking(X_a, y_t, NUM_LAYERS, layer=-1):
    
    layer_X_train = get_layer_vector(X_a, y_t, NUM_LAYERS, layer)
    avg_embeddings, average_embeddings_by_label = get_mean_vectors(layer_X_train, y_t)
    ranking = get_overall_ranking(avg_embeddings)
    
    return ranking

def tag_wise_ranking(X_a, y_t , tag, NUM_LAYERS, layer=-1):

    layer_X_train = get_layer_vector(X_a, y_t, NUM_LAYERS, layer)
    avg_embeddings, average_embeddings_by_label = get_mean_vectors(layer_X_train, y_t)
    summation, ranking = get_tag_wise_ranking(average_embeddings_by_label, tag)

    return ranking

def ranking_for_all_tags(X_a, y_t, NUM_LAYERS, layer=-1):
    
    ranking_per_tag = {}
    layer_X_train = get_layer_vector(X_a, y_t, NUM_LAYERS, layer)
    avg_embeddings, average_embeddings_by_label = get_mean_vectors(layer_X_train, y_t)

    overall = np.zeros_like(layer_X_train[0])

    for c, qz in average_embeddings_by_label.items():

        summation, ranking = get_tag_wise_ranking(average_embeddings_by_label, c)
        overall = np.add(overall, summation)
        ranking_per_tag [c] = ranking

    overall_ranking = np.argsort(overall)[::-1].tolist()

    return overall_ranking, ranking_per_tag








