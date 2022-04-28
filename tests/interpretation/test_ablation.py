import unittest

from unittest.mock import MagicMock, patch

import neurox.interpretation.ablation as ablation

import numpy as np


class TestFilterActivationsKeepNeurons(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.total_examples = 10
        cls.total_neurons = 100
        cls.neurons_to_keep = 3

    def setUp(self):
        activations = np.random.random((self.total_examples, self.total_neurons))
        useful_activations = activations[:, : self.neurons_to_keep].copy()
        shuffled_idx = np.random.permutation(np.arange(self.total_neurons))
        activations = activations[:, shuffled_idx]

        neuron_idx_to_keep = np.concatenate(
            [np.where(shuffled_idx == i)[0] for i in range(self.neurons_to_keep)]
        )

        self.activations = activations
        self.useful_activations = useful_activations
        self.neuron_idx_to_keep = neuron_idx_to_keep

    def test_filter_activations_keep_neurons(self):
        "Filter activations (keep neurons)"
        # Test if the correct activations are returned
        filtered_activations = ablation.filter_activations_keep_neurons(
            self.activations, self.neuron_idx_to_keep
        )
        np.testing.assert_array_almost_equal(
            filtered_activations, self.useful_activations
        )

    def test_filter_activations_keep_neurons_view(self):
        "Filter activations (keep neurons) view"
        # Test if changing the returned view changes the original matrix
        filtered_activations = ablation.filter_activations_keep_neurons(
            self.activations, self.neuron_idx_to_keep
        )
        filtered_activations[:, :] = 0
        np.testing.assert_array_almost_equal(
            filtered_activations, np.zeros((self.total_examples, self.neurons_to_keep))
        )

    def test_keep_specific_neurons(self):
        "Filter activations (keep neurons) - alternative function"
        # Test if the correct activations are returned
        filtered_activations = ablation.keep_specific_neurons(
            self.activations, self.neuron_idx_to_keep
        )
        np.testing.assert_array_almost_equal(
            filtered_activations, self.useful_activations
        )

    def test_keep_specific_neurons_view(self):
        "Filter activations (keep neurons) view - alternative function"
        # Test if changing the returned view changes the original matrix
        filtered_activations = ablation.keep_specific_neurons(
            self.activations, self.neuron_idx_to_keep
        )
        filtered_activations[:, :] = 0
        np.testing.assert_array_almost_equal(
            filtered_activations, np.zeros((self.total_examples, self.neurons_to_keep))
        )


class TestFilterActivationsRemoveNeurons(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.total_examples = 10
        cls.total_neurons = 100
        cls.neurons_to_remove = 3

    def setUp(self):
        activations = np.random.random((self.total_examples, self.total_neurons))
        useful_activations = np.sort(activations[:, self.neurons_to_remove :].copy())
        shuffled_idx = np.random.permutation(np.arange(self.total_neurons))
        activations = activations[:, shuffled_idx]

        neuron_idx_to_remove = np.concatenate(
            [np.where(shuffled_idx == i)[0] for i in range(self.neurons_to_remove)]
        )

        self.activations = activations
        self.useful_activations = useful_activations
        self.neuron_idx_to_remove = neuron_idx_to_remove

    def test_filter_activations_remove_neurons(self):
        "Filter activations (remove neurons)"
        # Test if the correct activations are returned
        filtered_activations = ablation.filter_activations_remove_neurons(
            self.activations, self.neuron_idx_to_remove
        )
        np.testing.assert_array_almost_equal(
            np.sort(filtered_activations), self.useful_activations
        )

    def test_filter_activations_remove_neurons_view(self):
        "Filter activations (remove neurons) view"
        # Test if changing the returned view changes the original matrix
        filtered_activations = ablation.filter_activations_remove_neurons(
            self.activations, self.neuron_idx_to_remove
        )
        filtered_activations[:, :] = 0
        np.testing.assert_array_almost_equal(
            filtered_activations,
            np.zeros(
                (self.total_examples, self.total_neurons - self.neurons_to_remove)
            ),
        )


class TestZeroOutActivationsKeepNeurons(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.total_examples = 10
        cls.total_neurons = 100
        cls.neurons_to_keep = 3

    def test_zero_out_activations_keep_neurons(self):
        "Zero out activations (keep neurons)"

        # Test if the correct activations are returned
        activations = np.random.random((self.total_examples, self.total_neurons))
        expected_activations = activations.copy()
        expected_activations[:, self.neurons_to_keep :] = 0
        shuffled_idx = np.random.permutation(np.arange(self.total_neurons))
        activations = activations[:, shuffled_idx]
        expected_activations = expected_activations[:, shuffled_idx]

        neuron_idx_to_keep = np.concatenate(
            [np.where(shuffled_idx == i)[0] for i in range(self.neurons_to_keep)]
        )

        filtered_activations = ablation.zero_out_activations_keep_neurons(
            activations, neuron_idx_to_keep
        )
        np.testing.assert_array_almost_equal(filtered_activations, expected_activations)


class TestZeroOutActivationsRemoveNeurons(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.total_examples = 10
        cls.total_neurons = 100
        cls.neurons_to_remove = 3

    def test_zero_out_activations_remove_neurons(self):
        "Zero out activations (remove neurons)"

        # Test if the correct activations are returned
        activations = np.random.random((self.total_examples, self.total_neurons))
        expected_activations = activations.copy()
        expected_activations[:, : self.neurons_to_remove] = 0
        shuffled_idx = np.random.permutation(np.arange(self.total_neurons))
        activations = activations[:, shuffled_idx]
        expected_activations = expected_activations[:, shuffled_idx]

        neuron_idx_to_remove = np.concatenate(
            [np.where(shuffled_idx == i)[0] for i in range(self.neurons_to_remove)]
        )

        filtered_activations = ablation.zero_out_activations_remove_neurons(
            activations, neuron_idx_to_remove
        )
        np.testing.assert_array_almost_equal(filtered_activations, expected_activations)


class TestFilterActivationsByLayers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.num_examples = 10
        cls.num_neurons_per_layer = 72
        cls.num_layers = 4

    def setUp(self):
        layer_activations = []
        for l in range(self.num_layers):
            layer_activations.append(
                np.random.random((self.num_examples, self.num_neurons_per_layer))
            )

        self.layer_activations = layer_activations
        self.activations = np.concatenate(layer_activations, axis=1)

    def test_filter_activations_by_layers_start_layer(self):
        "Filter activations by layer (Start layer)"

        selected_layer = 0
        filtered_activations = ablation.filter_activations_by_layers(
            self.activations, [selected_layer], self.num_layers
        )
        np.testing.assert_array_almost_equal(
            filtered_activations, self.layer_activations[selected_layer]
        )

    def test_filter_activations_by_layers_middle_layer(self):
        "Filter activations by layer (Middle layer)"

        selected_layer = 1
        filtered_activations = ablation.filter_activations_by_layers(
            self.activations, [selected_layer], self.num_layers
        )
        np.testing.assert_array_almost_equal(
            filtered_activations, self.layer_activations[selected_layer]
        )

    def test_filter_activations_by_layers_last_layer(self):
        "Filter activations by layer (Last layer)"

        selected_layer = self.num_layers - 1
        filtered_activations = ablation.filter_activations_by_layers(
            self.activations, [selected_layer], self.num_layers
        )
        np.testing.assert_array_almost_equal(
            filtered_activations, self.layer_activations[selected_layer]
        )

    def test_filter_activations_by_layers_multiple(self):
        "Filter activations by layers (Multiple layers)"

        selected_layers = [1, 3]
        filtered_activations = ablation.filter_activations_by_layers(
            self.activations, selected_layers, self.num_layers
        )
        expected_output = np.concatenate(
            [self.layer_activations[s_l] for s_l in selected_layers], axis=1
        )
        np.testing.assert_array_almost_equal(filtered_activations, expected_output)

    def test_filter_activations_by_layers_bidi_forward(self):
        "Filter activations by layer (Bi-directional forward)"

        selected_layer = 2
        filtered_activations = ablation.filter_activations_by_layers(
            self.activations,
            [selected_layer],
            self.num_layers,
            bidirectional_filtering="forward",
        )
        np.testing.assert_array_almost_equal(
            filtered_activations,
            self.layer_activations[selected_layer][
                :, : self.num_neurons_per_layer // 2
            ],
        )

    def test_filter_activations_by_layers_bidi_backward(self):
        "Filter activations by layer (Bi-directional backward)"

        selected_layer = 2
        filtered_activations = ablation.filter_activations_by_layers(
            self.activations,
            [selected_layer],
            self.num_layers,
            bidirectional_filtering="backward",
        )
        np.testing.assert_array_almost_equal(
            filtered_activations,
            self.layer_activations[selected_layer][
                :, self.num_neurons_per_layer // 2 :
            ],
        )

    def test_filter_activations_by_layers_bidi_forward_multiple(self):
        "Filter activations by layers (Bi-directional forward)"

        selected_layers = [1, 3]
        filtered_activations = ablation.filter_activations_by_layers(
            self.activations,
            selected_layers,
            self.num_layers,
            bidirectional_filtering="forward",
        )
        expected_output = np.concatenate(
            [
                self.layer_activations[s_l][:, : self.num_neurons_per_layer // 2]
                for s_l in selected_layers
            ],
            axis=1,
        )
        np.testing.assert_array_almost_equal(filtered_activations, expected_output)

    def test_filter_activations_by_layers_bidi_backward_multiple(self):
        "Filter activations by layers (Bi-directional backward)"

        selected_layers = [1, 3]
        filtered_activations = ablation.filter_activations_by_layers(
            self.activations,
            selected_layers,
            self.num_layers,
            bidirectional_filtering="backward",
        )
        expected_output = np.concatenate(
            [
                self.layer_activations[s_l][:, self.num_neurons_per_layer // 2 :]
                for s_l in selected_layers
            ],
            axis=1,
        )
        np.testing.assert_array_almost_equal(filtered_activations, expected_output)
