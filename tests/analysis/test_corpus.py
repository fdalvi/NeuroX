import random
import unittest

import neurox.analysis.corpus as corpus

import numpy as np


class TestGetTopWords(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.num_tokens = 999
        cls.num_neurons = 3

        all_idx = np.random.permutation(np.arange(cls.num_tokens))
        neuron_1_idx = all_idx[:333]
        neuron_2_idx = all_idx[333:666]
        neuron_3_idx = all_idx[666:]

        # Create activations such that for all tokens associated with a particular
        # neuron, that neuron's activation is > 0.5, and all others are close
        # to zero
        cls.activations = [
            np.random.random((cls.num_tokens, cls.num_neurons)).astype(np.float32) * 2
            - 1
        ]

        # Drive irrelevant token/neuron pair activations to zero
        cls.activations[0][np.concatenate((neuron_2_idx, neuron_3_idx)), 0] /= 100000.0
        cls.activations[0][np.concatenate((neuron_1_idx, neuron_3_idx)), 1] /= 100000.0
        cls.activations[0][np.concatenate((neuron_1_idx, neuron_2_idx)), 2] /= 100000.0

        cls.tokens = {"source": [[None] * cls.num_tokens]}
        for idx in neuron_1_idx:
            cls.tokens["source"][0][idx] = f"neuron_1_token_{random.randint(0, 50)}"
        for idx in neuron_2_idx:
            cls.tokens["source"][0][idx] = f"neuron_2_token_{random.randint(0, 50)}"
        for idx in neuron_3_idx:
            cls.tokens["source"][0][idx] = f"neuron_3_token_{random.randint(0, 50)}"

    def test_get_top_words(self):
        "Basic top words evaluation"

        top_words = corpus.get_top_words(self.tokens, self.activations, 0)

        tokens = map(lambda x: x[0], top_words)
        tokens = [x[:14] for x in tokens]
        self.assertNotIn("neuron_2_token", tokens)
        self.assertNotIn("neuron_3_token", tokens)

    def test_get_top_words_2(self):
        "Basic top words evaluation"

        top_words = corpus.get_top_words(self.tokens, self.activations, 1)

        tokens = map(lambda x: x[0], top_words)
        tokens = [x[:14] for x in tokens]
        self.assertNotIn("neuron_1_token", tokens)
        self.assertNotIn("neuron_3_token", tokens)

    def test_get_top_words_3(self):
        "Basic top words evaluation"

        top_words = corpus.get_top_words(self.tokens, self.activations, 2)

        tokens = map(lambda x: x[0], top_words)
        tokens = [x[:14] for x in tokens]
        self.assertNotIn("neuron_1_token", tokens)
        self.assertNotIn("neuron_2_token", tokens)

    def test_get_top_words_limit_tokens(self):
        "Top words with limit on the number of tokens"

        top_words = corpus.get_top_words(
            self.tokens, self.activations, 0, num_tokens=10
        )
        self.assertEqual(len(top_words), 10)

    def test_min_words_and_min_threshold_exception(self):
        "Throw exception if both num_tokens and min_threshold is set"

        self.assertRaises(
            ValueError,
            corpus.get_top_words,
            self.tokens,
            self.activations,
            1,
            num_tokens=10,
            min_threshold=0.5,
        )
