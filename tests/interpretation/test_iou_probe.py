import random
import unittest

from unittest.mock import ANY, MagicMock, patch

import neurox.interpretation.iou_probe as linear_probe

import numpy as np
import torch

from torch.autograd import Variable


class TestGetNeuronOrdering(unittest.TestCase):
    @patch("neurox.interpretation.linear_probe.LinearProbe")
    def test_get_neuron_ordering(self, probe_mock):
        "Basic get neuron ordering test"

        # Create a weight matrix with 2 samples and 3 neurons
        # Neuron 3 is the most important neuron as it has the highest weight in
        # both samples, followed by Neuron 1 which has equal weight in sample 2
        # but higher weight in sample 1, ending with Neuron 2
        mock_weight_matrix = [[4, 1, 5], [1, 1, 10]]
        expected_neuron_order = [2, 0, 1]
        probe_mock.parameters.return_value = [torch.Tensor(mock_weight_matrix)]

        ordering, cutoffs = iou_probe.get_neuron_ordering(
            probe_mock, {"class0": 0, "class1": 1}
        )

        self.assertListEqual(ordering, expected_neuron_order)
