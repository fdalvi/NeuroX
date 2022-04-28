import random
import unittest

from unittest.mock import ANY, MagicMock, patch

import neurox.interpretation.linear_probe as linear_probe

import numpy as np
import torch

from torch.autograd import Variable


class TestL1Regularization(unittest.TestCase):
    def test_l1_penalty(self):
        "L1 Regularization"
        tmp = np.random.random((5, 5))
        expected_penalty = np.sum(np.abs(tmp))
        penalty = linear_probe.l1_penalty(Variable(torch.Tensor(tmp)))
        self.assertIsInstance(penalty, Variable)
        self.assertAlmostEqual(expected_penalty, penalty.data.item(), places=3)


class TestL2Regularization(unittest.TestCase):
    def test_l2_penalty(self):
        "L2 Regularization"
        tmp = np.random.random((5, 5))
        expected_penalty = np.sqrt(np.sum(np.power(tmp, 2)))
        penalty = linear_probe.l2_penalty(Variable(torch.Tensor(tmp)))
        self.assertIsInstance(penalty, Variable)
        self.assertAlmostEqual(expected_penalty, penalty.data.item(), places=3)


class TestLinearProbeClass(unittest.TestCase):
    def test_linear_probe_init(self):
        "Linear Probe Initialization"
        probe = linear_probe.LinearProbe(50, 5)
        self.assertEqual(probe.linear.in_features, 50)
        self.assertEqual(probe.linear.out_features, 5)

    @patch("torch.nn.Linear")
    def test_linear_probe_forward(self, linear_mock):
        "Linear Probe Forward"
        probe = linear_probe.LinearProbe(50, 5)
        probe.forward(torch.rand((50, 1)))
        linear_mock.assert_called_once()


class TestTrainProbe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.num_examples = 10
        cls.num_features = 100
        cls.num_classes = 3

        cls.X = np.random.random((cls.num_examples, cls.num_features)).astype(
            np.float32
        )

        # Ensure y has all class labels atleast once for classification
        cls.y_classification = np.concatenate(
            (
                np.arange(cls.num_classes),
                np.random.randint(
                    0, cls.num_classes, size=cls.num_examples - cls.num_classes
                ),
            )
        )
        cls.y_regression = np.random.random((cls.num_examples)).astype(np.float32)

    @patch("torch.optim.Adam.step")
    def test_train_classification_probe(self, optimizer_step_fn):
        "Basic classification probe training test"
        num_epochs = 5

        linear_probe._train_probe(
            self.X, self.y_classification, "classification", num_epochs=num_epochs
        )

        self.assertEqual(optimizer_step_fn.call_count, num_epochs)

    def test_train_classification_probe_one_class(self):
        "Classification probe with one class test"

        y = np.zeros((self.num_examples,))
        self.assertRaises(
            ValueError, linear_probe._train_probe, self.X, y, "classification"
        )

    def test_train_probe_invalid_type(self):
        "Train probe of invalid type"
        self.assertRaises(
            ValueError,
            linear_probe._train_probe,
            self.X,
            self.y_classification,
            "invalid-type",
        )

    @patch("torch.optim.Adam.step")
    def test_train_regression_probe(self, optimizer_step_fn):
        "Basic regression probe training test"
        num_epochs = 12

        linear_probe._train_probe(
            self.X, self.y_regression, "regression", num_epochs=num_epochs
        )

        self.assertEqual(optimizer_step_fn.call_count, num_epochs)

    def test_train_probe_no_regularization(self):
        "Probe training with wrong regularization test"

        self.assertRaises(
            ValueError,
            linear_probe._train_probe,
            self.X,
            self.y_classification,
            "classification",
            lambda_l1=None,
        )

    @patch("torch.optim.Adam.step")
    def test_train_probe_float16(self, optimizer_step_fn):
        "Basic probe training test. Same test as before but different data dtype"
        X = np.random.random((self.num_examples, self.num_features)).astype(np.float16)

        # Ensure y has all three class labels atleast once
        y = np.concatenate(
            (
                np.arange(self.num_classes),
                np.random.randint(
                    0, self.num_classes, size=self.num_examples - self.num_classes
                ),
            )
        )
        linear_probe._train_probe(X, y, "classification")

        self.assertEqual(optimizer_step_fn.call_count, 10)


class TestEvaluateProbe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.num_examples = 10
        cls.num_features = 100
        cls.num_classes = 3

        X = np.random.random((cls.num_examples, cls.num_features)).astype(np.float32)
        # Ensure y has all three class labels atleast once
        y_classification = np.concatenate(
            (
                np.arange(cls.num_classes),
                np.random.randint(
                    0, cls.num_classes, size=cls.num_examples - cls.num_classes
                ),
            )
        )

        y_regression = np.random.random((cls.num_examples,)).astype(np.float32)

        cls.trained_probe = linear_probe._train_probe(
            X, y_classification, "classification"
        )
        cls.trained_regression_probe = linear_probe._train_probe(
            X, y_regression, "regression"
        )

    def test_evaluate_classification_probe(self):
        "Basic classification probe evaluation"

        scores = linear_probe.evaluate_probe(
            self.trained_probe,
            np.random.random((self.num_examples, self.num_features)).astype(np.float32),
            np.random.randint(0, self.num_classes, size=self.num_examples),
        )
        self.assertIn("__OVERALL__", scores)

    def test_evaluate_regression_probe(self):
        "Basic regresson probe evaluation"

        scores = linear_probe.evaluate_probe(
            self.trained_regression_probe,
            np.random.random((self.num_examples, self.num_features)).astype(np.float32),
            np.random.random((self.num_examples,)),
        )
        self.assertIn("__OVERALL__", scores)

    def test_evaluate_probe_with_class_labels(self):
        "Evaluation with class labels"

        scores = linear_probe.evaluate_probe(
            self.trained_probe,
            np.random.random((self.num_examples, self.num_features)).astype(np.float32),
            np.random.randint(0, self.num_classes, size=self.num_examples),
            idx_to_class={0: "class0", 1: "class1", 2: "class2"},
        )
        self.assertIn("__OVERALL__", scores)
        self.assertIn("class0", scores)
        self.assertIn("class1", scores)

    def test_evaluate_probe_with_class_labels_float16(self):
        "Evaluation with class labels. Same test as before but different data dtype"

        scores = linear_probe.evaluate_probe(
            self.trained_probe,
            np.random.random((self.num_examples, self.num_features)).astype(np.float16),
            np.random.randint(0, self.num_classes, size=self.num_examples),
            idx_to_class={0: "class0", 1: "class1", 2: "class2"},
        )
        self.assertIn("__OVERALL__", scores)
        self.assertIn("class0", scores)
        self.assertIn("class1", scores)

    def test_evaluate_probe_with_return_predictions(self):
        "Probe evaluation with returned predictions"

        y_true = np.random.randint(0, self.num_classes, size=self.num_examples)
        scores, predictions = linear_probe.evaluate_probe(
            self.trained_probe,
            np.random.random((self.num_examples, self.num_features)).astype(np.float32),
            y_true,
            return_predictions=True,
        )
        self.assertIn("__OVERALL__", scores)
        self.assertIsInstance(predictions, list)
        self.assertEqual(len(predictions), self.num_examples)
        self.assertIsInstance(predictions[0], tuple)

        # Source words should be from 0 to num_examples since no source_tokens
        # were given
        self.assertListEqual(
            [p[0] for p in predictions], list(range(self.num_examples))
        )
        self.assertNotEqual([p[1] for p in predictions], list(y_true))


class TestGetTopNeurons(unittest.TestCase):
    @patch("neurox.interpretation.linear_probe.LinearProbe")
    def test_get_top_neurons(self, probe_mock):
        "Basic get top neurons test"

        # Create a weight matrix with 2 samples and 3 neurons
        # In the first sample, more than 50% of the weight mass is covered by
        # the first neuron.
        # In the second sample, more than 50% of the weight mass is covered by
        # the second neuron
        mock_weight_matrix = [[5, 1, 1], [1, 10, 1]]
        probe_mock.parameters.return_value = [torch.Tensor(mock_weight_matrix)]

        top_neurons, classwise_top_neurons = linear_probe.get_top_neurons(
            probe_mock, 0.5, {"class0": 0, "class1": 1}
        )
        np.testing.assert_array_equal(top_neurons, [0, 1])
        np.testing.assert_array_equal(classwise_top_neurons["class0"], [0])
        np.testing.assert_array_equal(classwise_top_neurons["class1"], [1])

    @patch("neurox.interpretation.linear_probe.LinearProbe")
    def test_get_top_neurons_all_selection(self, probe_mock):
        "Get top neurons with all selection test"

        # Create a weight matrix with 2 samples and 3 neurons
        mock_weight_matrix = [[10, 9, 8], [10, 2, 1]]
        probe_mock.parameters.return_value = [torch.Tensor(mock_weight_matrix)]

        top_neurons, classwise_top_neurons = linear_probe.get_top_neurons(
            probe_mock,
            1.1,  # Percentage is higher than total mass, all neurons will be top neurons
            {"class0": 0, "class1": 1},
        )

        np.testing.assert_array_equal(top_neurons, [0, 1, 2])
        np.testing.assert_array_equal(classwise_top_neurons["class0"], [0, 1, 2])
        np.testing.assert_array_equal(classwise_top_neurons["class1"], [0, 1, 2])


class TestGetTopNeuronsHardThreshold(unittest.TestCase):
    @patch("neurox.interpretation.linear_probe.LinearProbe")
    def test_get_top_neurons_hard_threshold(self, probe_mock):
        "Basic get top neurons with hard threshold test"

        # Create a weight matrix with 2 samples and 4 neurons
        # In the first sample, only the first neuron is higher than
        # max_weight (5) / threshold (2) = 2.5
        # In the second sample, the second and fourth neuron are higher than
        # max_weight(10) / threshold (2) = 5
        mock_weight_matrix = [[5, 1, 2, 1], [1, 10, 1, 6]]
        probe_mock.parameters.return_value = [torch.Tensor(mock_weight_matrix)]

        (
            top_neurons,
            classwise_top_neurons,
        ) = linear_probe.get_top_neurons_hard_threshold(
            probe_mock, 2, {"class0": 0, "class1": 1}
        )
        np.testing.assert_array_equal(top_neurons, [0, 1, 3])
        np.testing.assert_array_equal(classwise_top_neurons["class0"], [0])
        np.testing.assert_array_equal(classwise_top_neurons["class1"], [1, 3])


class TestGetBottomNeurons(unittest.TestCase):
    @patch("neurox.interpretation.linear_probe.LinearProbe")
    def test_get_bottom_neurons(self, probe_mock):
        "Basic get bottom neurons test"

        # Create a weight matrix with 2 samples and 3 neurons
        # In the first sample, the third neuron alone covers the bottom 10% of
        # the total weight mass (5+4+1=10)
        # In the second sample, the second and third neuron cover the bottom 10%
        # of the total weight mass (10+1+1=12)
        mock_weight_matrix = [[5, 4, 1], [10, 1, 1]]
        probe_mock.parameters.return_value = [torch.Tensor(mock_weight_matrix)]

        bottom_neurons, classwise_bottom_neurons = linear_probe.get_bottom_neurons(
            probe_mock, 0.1, {"class0": 0, "class1": 1}
        )

        np.testing.assert_array_equal(bottom_neurons, [1, 2])
        np.testing.assert_array_equal(classwise_bottom_neurons["class0"], [2])
        np.testing.assert_array_equal(classwise_bottom_neurons["class1"], [1, 2])

    @patch("neurox.interpretation.linear_probe.LinearProbe")
    def test_get_bottom_neurons_all_selection(self, probe_mock):
        "Get bottom neurons with all selection test"

        # Create a weight matrix with 2 samples and 3 neurons
        mock_weight_matrix = [[8, 9, 10], [1, 2, 10]]
        probe_mock.parameters.return_value = [torch.Tensor(mock_weight_matrix)]

        bottom_neurons, classwise_bottom_neurons = linear_probe.get_bottom_neurons(
            probe_mock,
            1.1,  # Percentage is higher than total mass, all neurons will be bottom neurons
            {"class0": 0, "class1": 1},
        )

        # All neurons must be bottom neurons
        np.testing.assert_array_equal(bottom_neurons, [0, 1, 2])
        np.testing.assert_array_equal(classwise_bottom_neurons["class0"], [0, 1, 2])
        np.testing.assert_array_equal(classwise_bottom_neurons["class1"], [0, 1, 2])


class TestGetRandomNeurons(unittest.TestCase):
    @patch("numpy.random.random")
    @patch("neurox.interpretation.linear_probe.LinearProbe")
    def test_get_random_neurons(self, probe_mock, numpy_random):
        "Basic get random neurons test"
        # Mock the randomize weights against which the probability (0.35 here)
        # is compared.
        # Exactly expected_random_neurons will have random values below 0.35 and
        # the rest will have a higher random values
        expected_random_neurons = 3413
        probe_mock.parameters.return_value = [torch.rand((2, 10000))]
        mock_weights = np.array(
            [random.random() / 3 for _ in range(expected_random_neurons)]
            + [
                0.4 + random.random() / 6
                for _ in range(10000 - expected_random_neurons)
            ]
        )
        numpy_random.return_value = mock_weights

        random_neurons = linear_probe.get_random_neurons(probe_mock, 0.35)

        self.assertEqual(len(random_neurons), expected_random_neurons)


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

        ordering, cutoffs = linear_probe.get_neuron_ordering(
            probe_mock, {"class0": 0, "class1": 1}
        )

        self.assertListEqual(ordering, expected_neuron_order)


class TestGetNeuronOrderingGranular(unittest.TestCase):
    @patch("neurox.interpretation.linear_probe.LinearProbe")
    def test_get_neuron_ordering_granular(self, probe_mock):
        "Basic get neuron ordering (granular) test"

        # Create a weight matrix with 2 samples and 3 neurons
        # Neuron 3 is the most important neuron as it has the highest weight in
        # both samples, followed by Neuron 1 which has equal weight in sample 2
        # but higher weight in sample 1, ending with Neuron 2
        mock_weight_matrix = [[4, 1, 5], [1, 1, 10]]
        expected_neuron_order = [2, 0, 1]
        probe_mock.parameters.return_value = [torch.Tensor(mock_weight_matrix)]

        ordering, cutoffs = linear_probe.get_neuron_ordering_granular(
            probe_mock,
            {"class0": 0, "class1": 1},
            granularity=1,  # Basic test looks at 1 neuron in every chunk
        )

        self.assertListEqual(ordering, expected_neuron_order)


class TestGetFixedBottomNeurons(unittest.TestCase):
    @patch("neurox.interpretation.linear_probe.LinearProbe")
    def test_get_fixed_number_of_bottom_neurons(self, probe_mock):
        "Basic get fixed global bottom neurons test"

        # Create a weight matrix with 2 samples and 3 neurons
        # Neuron 2 is the least important globally as it has the smallest weight
        # in both samples
        mock_weight_matrix = [[4, 1, 5], [1, 1, 10]]
        probe_mock.parameters.return_value = [torch.Tensor(mock_weight_matrix)]

        bottom_neurons = linear_probe.get_fixed_number_of_bottom_neurons(
            probe_mock, 1, {"class0": 0, "class1": 1}
        )

        self.assertListEqual(bottom_neurons, [1])


class TestTrainClassificationProbe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.num_examples = 10
        cls.num_features = 100
        cls.num_classes = 3

    @patch("neurox.interpretation.linear_probe._train_probe")
    def test_train_logistic_regression_probe(self, train_probe_mock):
        "Logistic Regression probe test"

        linear_probe.train_logistic_regression_probe("X_data", "y_data")

        train_probe_mock.assert_called_with(
            ANY,
            ANY,
            task_type="classification",
            batch_size=ANY,
            lambda_l1=ANY,
            lambda_l2=ANY,
            learning_rate=ANY,
            num_epochs=ANY,
        )


class TestTrainRegressionProbe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.num_examples = 10
        cls.num_features = 100
        cls.num_classes = 3

    @patch("neurox.interpretation.linear_probe._train_probe")
    def test_train_linear_regression_probe(self, train_probe_mock):
        "Linear Regression probe test"

        linear_probe.train_linear_regression_probe("X_data", "y_data")

        train_probe_mock.assert_called_with(
            ANY,
            ANY,
            task_type="regression",
            batch_size=ANY,
            lambda_l1=ANY,
            lambda_l2=ANY,
            learning_rate=ANY,
            num_epochs=ANY,
        )
