import random 
import unittest 

import neurox.interpretation.probeless as probeless 
import neurox.interpretation.iou_probe as iou_probe
import neurox.interpretation.gaussian_probe as guassian_probe
import neurox.interpretation.linear_probe as linear_probe
import numpy as np
import torch 

seed_value = 99
np.random.seed(seed_value)


def test_probeless_ranking(snapshot):
    X = np.random.rand(100, 768)
    y = np.random.randint(2, size = 100)
    probeless_ordering = probeless.get_neuron_ordering(X,y)
    assert probeless_ordering == snapshot

def test_iou_ranking(snapshot): 
    X = np.random.rand(100, 768)
    y = np.random.randint(2, size = 100)
    iou_ordering = iou_probe.get_neuron_ordering(X, y)
    iou_ordering = list(iou_ordering)
    assert iou_ordering == snapshot


def test_guassian_ranking(snapshot):
    X = np.random.rand(10, 5)
    y = np.random.randint(2, size = 10)
    probe = guassian_probe.train_probe(X, y)
    guassian_ordering =  guassian_probe.get_neuron_ordering(probe, 768)
    assert guassian_ordering == snapshot

def test_logistic_regression_probe(snapshot): 

    X = np.random.rand(100, 768)
    y = np.random.randint(2, size = 100)
    probe = linear_probe.train_logistic_regression_probe(
            X, y, lambda_l2=0.1, lambda_l1=0.1
        )
    top_neurons, _ = linear_probe.get_top_neurons(probe, 1, {"class0": 0, "class1": 1})
    top_neurons = list(top_neurons)
    print(top_neurons)
    assert top_neurons == snapshot