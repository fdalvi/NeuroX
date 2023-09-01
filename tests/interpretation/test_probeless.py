import random
import unittest

import neurox.interpretation.probeless as  probeless

import numpy as np
import torch


# add tests to all functions in the file; not just only one test !!
class TestGetNeuronOrdering(unittest.TestCase):
    def test_get_neuron_ordering(self):
        "Basic get neuron ordering test"

     
        X = np.array([[-1, -1, 0], [-1, 0, 0], [1, 1, 0], [1, 0, 0]])
        y = np.array([0, 0, 1, 1])
        expected_neuron_order = [0, 1, 2]
        ordering = probeless.get_neuron_ordering(X, y)

        self.assertListEqual(list(ordering), expected_neuron_order)

class TestGetNeuronOrderingForTag(unittest.TestCase): 
    def test_get_neuron_ordering_for_tag(self): 
        """test for getting the neuron ordering for a particular tag"""
        X = np.array([[-1, -1, 0], [-1, 0, 0], [1, 1, 0], [1, 0, 0]])
        y = np.array([0, 0, 1, 1])
        idx2label = {0: "class0", 1:"class1"}
        expected_overall_ranking = [0, 1, 2]
        expected_class0_ranking = [0, 1, 2] 
        overall_ranking, ranking_per_tag = probeless.get_neuron_ordering_for_all_tags(X, y, idx2label)
        self.assertListEqual(list(overall_ranking), expected_overall_ranking)
        self.assertListEqual(list(ranking_per_tag["class0"]), expected_class0_ranking)
        self.assertListEqual(list(ranking_per_tag["class1"]), expected_class0_ranking) # why is the same ranking returned for twp classes
