import random
import unittest

import numpy as np
import torch

from neurox.interpretation.gaussian_probe import get_neuron_ordering, train_probe


class TestGetNeuronOrdering(unittest.TestCase):
    def test_get_neuron_ordering(self):
        "Basic get neuron ordering test"

        # The neuron 0 is perfectly correlated with label, the neuron 1 is partly correlated
        # The neuron 2 is not correlated

        X = np.array(
            [
                [-1.05, -1.1, 0.01],
                [0, -1.05, 0.005],
                [1.05, 1.1, 0.003],
                [0, 1.03, 0.1],
            ],
            dtype=float,
        )
        y = np.array([0, 0, 1, 1])
        expected_neuron_order = [1, 0, 2]
        probe = train_probe(X, y)
        selected_neurons = get_neuron_ordering(probe, 3)
        self.assertListEqual(list(selected_neurons), expected_neuron_order)
