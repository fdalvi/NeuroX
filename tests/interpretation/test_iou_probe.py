import random
import unittest

import neurox.interpretation.iou_probe as iou_probe

import numpy as np
import torch


class TestGetNeuronOrdering(unittest.TestCase):
    def test_get_neuron_ordering(self):
        "Basic get neuron ordering test"

        # The neuron 0 is perfectly correlated with label, the neuron 1 is partly correlated
        # The neuron 2 is not correlated

        X = np.array([[-1, -1, 0], [-1, 0, 0], [1, 1, 0], [1, 0, 0]])
        y = np.array([0, 0, 1, 1])
        expected_neuron_order = [0, 1, 2]
        ordering = iou_probe.get_neuron_ordering(X, y)

        self.assertListEqual(list(ordering), expected_neuron_order)
