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