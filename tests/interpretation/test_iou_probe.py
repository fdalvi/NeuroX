import random
import unittest

import neurox.interpretation.iou_probe as iou_probe

import numpy as np
import torch


class TestGetNeuronOrdering(unittest.TestCase):
    def test_get_neuron_ordering(self):
        "Basic get neuron ordering test"

        # Create a weight matrix with 2 samples and 3 neurons
        # Neuron 3 is the most important neuron as it has the highest weight in
        # both samples, followed by Neuron 1 which has equal weight in sample 2
        # but higher weight in sample 1, ending with Neuron 2
        # mock_weight_matrix = [[4, 1, 5], [1, 1, 10]]
        # expected_neuron_order = [2, 0, 1]
        # probe_mock.parameters.return_value = [torch.Tensor(mock_weight_matrix)]

        X = np.array([[-1, -1, 0], [-1, 0, 0], [1, 1, 0], [1, 0, 0]])
        y = np.array([0, 0, 1, 1])
        expected_neuron_order = [0, 1, 2]
        ordering = iou_probe.get_neuron_ordering(X, y)

        self.assertListEqual(list(ordering), expected_neuron_order)
