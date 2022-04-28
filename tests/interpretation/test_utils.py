import unittest

from unittest.mock import MagicMock, patch

import neurox.interpretation.linear_probe as linear_probe
import neurox.interpretation.utils as utils

import numpy as np


class TestIsNotebook(unittest.TestCase):
    def test_isnotebook(self):
        "Is Notebook"
        self.assertFalse(utils.isnotebook())

    # @patch("IPython.get_ipython")
    # def test_isnotebook_fake_jupyter(self, get_python_mock):
    #     "Is Notebook in fake jupyter environment"
    #     get_python_mock.return_value.__class__.__name__ = "ZMQInteractiveShell"
    #     self.assertTrue(utils.isnotebook())


class TestGetProgressBar(unittest.TestCase):
    def test_get_progress_bar(self):
        "Get Progress bar"
        from tqdm import tqdm as expected_progress_bar

        self.assertEqual(utils.get_progress_bar(), expected_progress_bar)

    @patch("neurox.interpretation.utils.isnotebook")
    def test_get_progress_bar_fake_jupyter(self, isnotebook_mock):
        "Get Progress bar in fake jupyter environment"
        isnotebook_mock.return_value = True
        from tqdm import tqdm_notebook as expected_progress_bar

        self.assertEqual(utils.get_progress_bar(), expected_progress_bar)


class TestBatchGenerator(unittest.TestCase):
    def test_batch_generator(self):
        "Batch Generator basic test"
        num_examples_per_batch = 10
        num_neurons = 72
        num_batches = 10

        batch_data = []
        for batch_idx in range(num_batches):
            batch_data.append(
                (
                    np.random.random((num_examples_per_batch, num_neurons)),
                    (np.random.random((num_examples_per_batch,)) * 10).astype(np.int),
                )
            )

        X = np.concatenate([X for X, _ in batch_data])
        y = np.concatenate([y for _, y in batch_data])

        for batch_idx, (X_batch, y_batch) in enumerate(
            utils.batch_generator(X, y, batch_size=10)
        ):
            np.testing.assert_array_equal(X_batch, batch_data[batch_idx][0])
            np.testing.assert_array_equal(y_batch, batch_data[batch_idx][1])

    def test_batch_generator_uneven_batch(self):
        "Batch Generator with uneven last batch"
        num_examples_per_batch = 10
        num_neurons = 72
        num_batches = 10

        batch_data = []
        for batch_idx in range(num_batches):
            batch_data.append(
                (
                    np.random.random((num_examples_per_batch, num_neurons)),
                    (np.random.random((num_examples_per_batch,)) * 10).astype(np.int),
                )
            )
        batch_data.append(
            (
                np.random.random((5, num_neurons)),
                (np.random.random((5,)) * 10).astype(np.int),
            )
        )

        X = np.concatenate([X for X, _ in batch_data])
        y = np.concatenate([y for _, y in batch_data])

        for batch_idx, (X_batch, y_batch) in enumerate(
            utils.batch_generator(X, y, batch_size=10)
        ):
            np.testing.assert_array_equal(X_batch, batch_data[batch_idx][0])
            np.testing.assert_array_equal(y_batch, batch_data[batch_idx][1])


class TestTok2Idx(unittest.TestCase):
    def test_tok2idx_unique_tokens(self):
        "tok2idx with unique tokens"
        tokens = [
            ["This", "is", "a", "sentence", "."],
            ["Over", "there", "!"],
        ]
        tok2idx = utils.tok2idx(tokens)

        # Ensure all tokens are in returned dict
        for sentence in tokens:
            for token in sentence:
                self.assertIn(token, tok2idx)

        # Ensure indices are contiguous
        idx = set(tok2idx.values())
        for i in range(len(tokens[0]) + len(tokens[1])):
            self.assertIn(i, idx)

    def test_tok2idx_repeated_tokens(self):
        "tok2idx with repeated tokens"
        tokens = [
            ["This", "is", "a", "sentence", "."],
            ["Over", "there", "!"],
            ["This", "is", "another", "sentence", "."],
        ]
        tok2idx = utils.tok2idx(tokens)

        # Ensure all tokens are in returned dict
        for sentence in tokens:
            for token in sentence:
                self.assertIn(token, tok2idx)

        # Ensure indices are contiguous
        idx = set(tok2idx.values())
        # One new unique token in last sentence
        for i in range(len(tokens[0]) + len(tokens[1]) + 1):
            self.assertIn(i, idx)


class TestIdx2Tok(unittest.TestCase):
    def test_idx2tok(self):
        "idx2tok"
        tokens = [
            ["This", "is", "a", "sentence", "."],
            ["Over", "there", "!"],
        ]
        tok2idx = utils.tok2idx(tokens)
        idx2tok = utils.idx2tok(tok2idx)

        # Ensure indices are contiguous
        idx = set(idx2tok.keys())
        for i in range(len(tokens[0]) + len(tokens[1])):
            self.assertIn(i, idx)

        all_tokens = set(idx2tok.values())
        for sentence in tokens:
            for token in sentence:
                self.assertIn(token, all_tokens)


class TestCountTargetWords(unittest.TestCase):
    def test_count_target_words_unique(self):
        "Count target words"
        tokens = {}
        tokens["target"] = [
            ["This", "is", "a", "sentence", "."],
            ["Over", "there", "!"],
        ]

        count = utils.count_target_words(tokens)
        expected_count = len(tokens["target"][0]) + len(tokens["target"][1])

        self.assertEqual(count, expected_count)

    def test_count_target_words_repeated(self):
        "Count target words with repeated tokens"
        tokens = {}
        tokens["target"] = [
            ["This", "is", "a", "sentence", "."],
            ["Over", "there", "!"],
            ["This", "is", "another", "sentence", "."],
        ]

        count = utils.count_target_words(tokens)
        expected_count = (
            len(tokens["target"][0])
            + len(tokens["target"][1])
            + len(tokens["target"][2])
        )

        self.assertEqual(count, expected_count)


class TestCreateTensors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.num_neurons = 72
        cls.tokens = {
            "source": [
                ["This", "is", "a", "sentence", "."],
                ["Over", "there", "!"],
            ],
            "target": [
                ["class0", "class1", "class0", "class0", "class1"],
                ["class2", "class0", "class1"],
            ],
        }

    def test_create_tensors_float32(self):
        "Create tensors basic test. float32 in, float32 out (default)"

        activations = [
            np.random.random((len(self.tokens["source"][0]), self.num_neurons)).astype(
                "float32"
            ),
            np.random.random((len(self.tokens["source"][1]), self.num_neurons)).astype(
                "float32"
            ),
        ]

        X, y, mapping = utils.create_tensors(self.tokens, activations, "class2")

        global_token_count = 0
        for activation in activations:
            for local_token_count in range(activation.shape[0]):
                np.testing.assert_array_almost_equal(
                    activation[local_token_count, :],
                    X[global_token_count, :],
                    decimal=3,
                )
                global_token_count += 1
                self.assertEqual(X.dtype, "float32")

    def test_create_tensors_float32_to_float16(self):
        "Create tensors basic test. float32 in, float16 out"

        activations = [
            np.random.random((len(self.tokens["source"][0]), self.num_neurons)).astype(
                "float32"
            ),
            np.random.random((len(self.tokens["source"][1]), self.num_neurons)).astype(
                "float32"
            ),
        ]

        X, y, mapping = utils.create_tensors(
            self.tokens, activations, "class2", dtype="float16"
        )

        global_token_count = 0
        for activation in activations:
            for local_token_count in range(activation.shape[0]):
                np.testing.assert_array_almost_equal(
                    activation[local_token_count, :],
                    X[global_token_count, :],
                    decimal=3,
                )
                global_token_count += 1
                self.assertEqual(X.dtype, "float16")


class TestPrintHelpers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mock_results = {
            "probe": linear_probe.LinearProbe(100, 5),
            "original_accs": {"__OVERALL__": 1.0},
            "global_results": {
                "10%": {
                    "keep_top_accs": {"__OVERALL__": 1.0},
                    "keep_random_accs": {"__OVERALL__": 1.0},
                    "keep_bottom_accs": {"__OVERALL__": 1.0},
                    "zero_out_top_accs": {"__OVERALL__": 1.0},
                    "zero_out_random_accs": {"__OVERALL__": 1.0},
                    "zero_out_bottom_accs": {"__OVERALL__": 1.0},
                },
                "15%": {
                    "keep_top_accs": {"__OVERALL__": 1.0},
                    "keep_random_accs": {"__OVERALL__": 1.0},
                    "keep_bottom_accs": {"__OVERALL__": 1.0},
                    "zero_out_top_accs": {"__OVERALL__": 1.0},
                    "zero_out_random_accs": {"__OVERALL__": 1.0},
                    "zero_out_bottom_accs": {"__OVERALL__": 1.0},
                },
                "20%": {
                    "keep_top_accs": {"__OVERALL__": 1.0},
                    "keep_random_accs": {"__OVERALL__": 1.0},
                    "keep_bottom_accs": {"__OVERALL__": 1.0},
                    "zero_out_top_accs": {"__OVERALL__": 1.0},
                    "zero_out_random_accs": {"__OVERALL__": 1.0},
                    "zero_out_bottom_accs": {"__OVERALL__": 1.0},
                },
                "ordering": [1, 2, 3],
            },
            "local_results": {"percentages": []},
        }

    @patch("builtins.print")
    def test_print_overall_stats(self, print_mock):
        "Print overall stats"
        utils.print_overall_stats(self.mock_results)
        print_mock.assert_called()

    @patch("builtins.print")
    def test_print_machine_stats(self, print_mock):
        "Print machine stats"
        utils.print_machine_stats(self.mock_results)
        print_mock.assert_called()


class TestBalanceBinaryClassData(unittest.TestCase):
    def test_balance_binary_class_data(self):
        "Balance binary class data"
        num_samples = 100
        num_neurons = 72

        X = np.random.random((num_samples, num_neurons))
        y = np.concatenate((np.zeros((10,)), np.ones((num_samples - 10))))

        balanced_X, balanced_y = utils.balance_binary_class_data(X, y)

        # Make sure both classes are balanced
        self.assertEqual(np.where(balanced_y == 0)[0].shape[0], 10)
        self.assertEqual(np.where(balanced_y == 1)[0].shape[0], 10)

        class0_X = balanced_X[np.where(balanced_y == 0)[0], :]
        class1_X = balanced_X[np.where(balanced_y == 1)[0], :]

        # Make sure all class activations were assigned correctly
        np.testing.assert_array_almost_equal(np.sort(class0_X), np.sort(X[:10, :]))
        for activation in class1_X:
            found = False
            for class1_neuron in range(10, num_samples):
                found = np.allclose(activation, X[class1_neuron, :])
                if found:
                    break
            self.assertTrue(found)


class TestBalanceMultiClassData(unittest.TestCase):
    def test_balance_multi_class_data(self):
        "Balance multi class data"
        num_samples = 100
        num_neurons = 72

        X = np.random.random((num_samples, num_neurons))
        y = np.concatenate(
            (np.zeros((10,)), np.ones((20,)), np.ones((num_samples - 10 - 20,)) * 2)
        )

        balanced_X, balanced_y = utils.balance_multi_class_data(X, y)

        # Make sure all three classes are balanced
        self.assertEqual(np.where(balanced_y == 0)[0].shape[0], 10)
        self.assertEqual(np.where(balanced_y == 1)[0].shape[0], 10)
        self.assertEqual(np.where(balanced_y == 2)[0].shape[0], 10)

        class0_X = balanced_X[np.where(balanced_y == 0)[0], :]
        class1_X = balanced_X[np.where(balanced_y == 1)[0], :]
        class2_X = balanced_X[np.where(balanced_y == 2)[0], :]

        # Make sure all class activations were assigned correctly
        np.testing.assert_array_almost_equal(np.sort(class0_X), np.sort(X[:10, :]))
        for activation in class1_X:
            found = False
            for class1_neuron in range(10, 30):
                found = np.allclose(activation, X[class1_neuron, :])
                if found:
                    break
            self.assertTrue(found)
        for activation in class2_X:
            found = False
            for class2_neuron in range(30, num_samples):
                found = np.allclose(activation, X[class2_neuron, :])
                if found:
                    break
            self.assertTrue(found)
