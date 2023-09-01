import random
import unittest
from typing import Dict

import neurox.data.loader as data_loader
import neurox.interpretation.ablation as ablation

import neurox.interpretation.probeless as probeless
import neurox.interpretation.utils as utils
import numpy as np
import torch
from neurox.data.extraction.transformers_extractor import get_model_and_tokenizer
from neurox.data.writer import ActivationsWriter

LOW_ACTIVATION_MEAN = 0.1
LOW_ACTIVATION_STD = 0.01
HIGH_ACTIVATION_MEAN = 0.9
HIGH_ACTIVATION_STD = 0.1

SEED = 99  # cannot set it as a fixture as we can only use fixtures from arguments of test functions
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)


################################ UTILITY FUNCTIONS ################################
def _construct_ground_truth_neurons(num_layers: int, total_neurons: int, concepts):
    neuron_options = list(range(total_neurons))
    result = {}
    for l in range(num_layers):
        neurons = {}
        for concept in concepts:
            neurons[concept] = random.sample(neuron_options, 1)
        result[l] = neurons
    return result


def _extract_synthetic_sentence_representations(
    sentence: str,
    num_layers: int,
    num_neurons: int,
    ground_truth_neurons: Dict,
    concepts,
):
    tokens = sentence.split()
    X = np.random.normal(
        LOW_ACTIVATION_MEAN,
        LOW_ACTIVATION_STD,
        size=(num_layers, len(tokens), num_neurons),
    )
    for idx, token in enumerate(tokens):
        if token in concepts:
            for layer in range(num_layers):
                neurons = ground_truth_neurons[layer][
                    token
                ]  # ground_truth_neurons is a dictionary in the form of {l: {c: n},l: {c: n} etc. }
                for neuron in neurons:
                    X[layer, idx, neuron] = np.random.normal(
                        HIGH_ACTIVATION_MEAN, HIGH_ACTIVATION_STD
                    )
    return X, tokens


def _extract_synthetic_representations(
    input_corpus,
    output_file,
    num_layers,
    num_neurons,
    ground_truth_neurons,
    concepts,
    output_type="json",
    decompose_layers=False,
    filter_layers=None,
):
    """
    Function to extract the representations for all sentences in the corpus
    """

    print("Reading input corpus")

    def corpus_generator(input_corpus_path):
        with open(input_corpus_path, "r") as fp:
            for line in fp:
                yield line.strip()
            return

    print("Preparing output file")
    writer = ActivationsWriter.get_writer(
        output_file,
        filetype=output_type,
        decompose_layers=decompose_layers,
        filter_layers=filter_layers,
    )
    print("Extracting representations from model")
    # I need to know the number of sentences before hand
    for sentence_idx, sentence in enumerate(corpus_generator(input_corpus)):
        hidden_states, extracted_words = _extract_synthetic_sentence_representations(
            sentence, num_layers, num_neurons, ground_truth_neurons, concepts
        )
        writer.write_activations(sentence_idx, extracted_words, hidden_states)
    writer.close()


################################ TEST CASES ################################
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
        X = np.array(
            [[1, -1, -1, -1], [-1, 1, -1, -1], [-1, -1, 1, -1], [-1, -1, -1, 1]]
        )
        y = np.array([0, 1, 2, 3])
        idx2label = {0: "class0", 1: "class1", 2: "class2", 3: "class3"}
        (
            top_neuron_for_class_0,
            top_neuron_for_class_1,
            top_neuron_for_class_2,
            top_neuron_for_class_3,
        ) = (0, 1, 2, 3)

        _, ranking_per_tag = probeless.get_neuron_ordering_for_all_tags(X, y, idx2label)
        self.assertEqual(ranking_per_tag["class0"][0], top_neuron_for_class_0)
        self.assertEqual(ranking_per_tag["class1"][0], top_neuron_for_class_1)
        self.assertEqual(ranking_per_tag["class2"][0], top_neuron_for_class_2)
        self.assertEqual(ranking_per_tag["class3"][0], top_neuron_for_class_3)


class TestGetNeuronOrderingForSyntheticData(unittest.TestCase):
    def test_get_neuron_ordering_for_synthetic_data(self):
        num_layers = 13
        num_neurons = 768
        concepts = ["NNP"]
        input_corpus = "tests/interpretation/test-data/pos_test.label"
        output_file = "tests/interpretation/test-data/activations.json"
        ground_truth_neurons = _construct_ground_truth_neurons(
            num_layers, num_neurons, concepts
        )
        _extract_synthetic_representations(
            input_corpus,
            output_file,
            num_layers,
            num_neurons,
            ground_truth_neurons,
            concepts,
        )
        activations, num_layers = data_loader.load_activations(output_file, 768)
        tokens = data_loader.load_data(input_corpus, input_corpus, activations, 512)
        X, y, mapping = utils.create_tensors(tokens, activations, "NA")
        _, idx2label, _, _ = mapping
        layer = random.choice(range(num_layers))
        layer_X = ablation.filter_activations_by_layers(X, [layer], num_layers)
        _, ranking_per_tag = probeless.get_neuron_ordering_for_all_tags(
            layer_X, y, idx2label
        )
        self.assertEqual(
            ranking_per_tag["NNP"][0], ground_truth_neurons[layer]["NNP"][0]
        )
