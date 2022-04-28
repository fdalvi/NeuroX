import unittest

from unittest.mock import MagicMock, patch

import neurox.interpretation.metrics as metrics

import numpy as np


class TestAccuracy(unittest.TestCase):
    def test_accuracy(self):
        "Accuracy"

        y_true = [1 for _ in range(50)] + [0 for _ in range(50)]
        y_pred = (
            [1 for _ in range(30)] + [0 for _ in range(20)] + [0 for _ in range(50)]
        )
        score = metrics.accuracy(y_pred, y_true)

        self.assertAlmostEqual(score, 0.8)

    def test_accuracy_numpy(self):
        "Accuracy with numpy arrays"

        y_true = np.array([1 for _ in range(50)] + [0 for _ in range(50)])
        y_pred = (
            [1 for _ in range(30)] + [0 for _ in range(20)] + [0 for _ in range(50)]
        )
        score = metrics.accuracy(y_pred, y_true)

        self.assertAlmostEqual(score, 0.8)

    def test_accuracy_no_correct(self):
        "Accuracy with no correct predictions"

        y_true = [1 for _ in range(50)] + [0 for _ in range(50)]
        y_pred = [0 for _ in range(50)] + [1 for _ in range(50)]
        score = metrics.accuracy(y_pred, y_true)

        self.assertAlmostEqual(score, 0.0)

    def test_accuracy_all_correct(self):
        "Accuracy with all correct predictions"

        y_true = [1 for _ in range(50)] + [0 for _ in range(50)]
        y_pred = [1 for _ in range(50)] + [0 for _ in range(50)]
        score = metrics.accuracy(y_pred, y_true)

        self.assertAlmostEqual(score, 1.0)


class TestF1(unittest.TestCase):
    @patch("neurox.interpretation.metrics.f1_score")
    def test_f1(self, sklearn_f1_mock):
        "F1"
        sklearn_f1_mock.return_value = 0
        y = [1 for _ in range(50)]
        score = metrics.f1(y, y)
        self.assertAlmostEqual(score, 0)

        sklearn_f1_mock.assert_called_once()

    @patch("neurox.interpretation.metrics.f1_score")
    def test_f1_numpy(self, sklearn_f1_mock):
        "F1 with numpy arrays"
        y = np.array([1 for _ in range(50)])
        score = metrics.f1(y, y)

        sklearn_f1_mock.assert_called_once()


class TestAccuracyAndF1(unittest.TestCase):
    @patch("neurox.interpretation.metrics.f1_score")
    def test_accuracy_and_f1(self, sklearn_f1_mock):
        "Accuracy and F1"
        sklearn_f1_mock.return_value = 0
        y_true = [1 for _ in range(50)] + [0 for _ in range(50)]
        y_pred = (
            [1 for _ in range(30)] + [0 for _ in range(20)] + [0 for _ in range(50)]
        )
        score = metrics.accuracy_and_f1(y_pred, y_true)

        self.assertAlmostEqual(score, 0.4)
        sklearn_f1_mock.assert_called_once()

    @patch("neurox.interpretation.metrics.f1_score")
    def test_accuracy_and_f1_numpy(self, sklearn_f1_mock):
        "Accuracy and F1 with numpy arrays"
        sklearn_f1_mock.return_value = 0
        y_true = [1 for _ in range(50)] + [0 for _ in range(50)]
        y_pred = (
            [1 for _ in range(30)] + [0 for _ in range(20)] + [0 for _ in range(50)]
        )
        score = metrics.accuracy_and_f1(y_pred, y_true)

        self.assertAlmostEqual(score, 0.4)
        sklearn_f1_mock.assert_called_once()


class TestPearson(unittest.TestCase):
    @patch("neurox.interpretation.metrics.pearsonr")
    def test_pearson(self, scipy_pearson_mock):
        "Pearson"
        scipy_pearson_mock.return_value = [0]
        y = [1 for _ in range(50)]
        score = metrics.pearson(y, y)

        scipy_pearson_mock.assert_called_once()

    @patch("neurox.interpretation.metrics.pearsonr")
    def test_pearson_numpy(self, scipy_pearson_mock):
        "Pearson with numpy arrays"
        y = np.array([1 for _ in range(50)])
        score = metrics.pearson(y, y)

        scipy_pearson_mock.assert_called_once()


class TestSpearman(unittest.TestCase):
    @patch("neurox.interpretation.metrics.spearmanr")
    def test_spearman(self, scipy_spearman_mock):
        "Pearson"
        scipy_spearman_mock.return_value = [0]
        y = [1 for _ in range(50)]
        score = metrics.spearman(y, y)

        scipy_spearman_mock.assert_called_once()

    @patch("neurox.interpretation.metrics.spearmanr")
    def test_spearman_numpy(self, scipy_spearman_mock):
        "Pearson with numpy arrays"
        y = np.array([1 for _ in range(50)])
        score = metrics.spearman(y, y)

        scipy_spearman_mock.assert_called_once()


class TestPearsonAndSpearman(unittest.TestCase):
    @patch("neurox.interpretation.metrics.spearmanr")
    @patch("neurox.interpretation.metrics.pearsonr")
    def test_pearson_and_spearman(self, scipy_pearson_mock, scipy_spearman_mock):
        "Pearson and Spearman"
        scipy_pearson_mock.return_value = [0.8]
        scipy_spearman_mock.return_value = [0.4]
        y = [1 for _ in range(50)]
        score = metrics.pearson_and_spearman(y, y)

        self.assertAlmostEqual(score, 0.6)
        scipy_pearson_mock.assert_called_once()
        scipy_spearman_mock.assert_called_once()

    @patch("neurox.interpretation.metrics.spearmanr")
    @patch("neurox.interpretation.metrics.pearsonr")
    def test_pearson_and_spearman_numpy(self, scipy_pearson_mock, scipy_spearman_mock):
        "Pearson and Spearman with numpy arrays"
        scipy_pearson_mock.return_value = [0.8]
        scipy_spearman_mock.return_value = [0.4]
        y = np.array([1 for _ in range(50)])
        score = metrics.pearson_and_spearman(y, y)

        self.assertAlmostEqual(score, 0.6)
        scipy_pearson_mock.assert_called_once()
        scipy_spearman_mock.assert_called_once()


class TestMatthewsCorrcoef(unittest.TestCase):
    @patch("neurox.interpretation.metrics.mcc")
    def test_matthews_corrcoef(self, sklearn_mcc_mock):
        "Matthew's Correlation Coefficient"
        y = [1 for _ in range(50)]
        score = metrics.matthews_corrcoef(y, y)

        sklearn_mcc_mock.assert_called_once()

    @patch("neurox.interpretation.metrics.mcc")
    def test_matthews_corrcoef_numpy(self, sklearn_mcc_mock):
        "Matthew's Correlation Coefficient with numpy arrays"
        y = np.array([1 for _ in range(50)])
        score = metrics.matthews_corrcoef(y, y)

        sklearn_mcc_mock.assert_called_once()


class TestComputeScore(unittest.TestCase):
    @patch("neurox.interpretation.metrics.accuracy")
    def test_compute_score_accuracy(self, accuracy_mock):
        "Accuracy"
        y = [1 for _ in range(50)]
        score = metrics.compute_score(y, y, "accuracy")

        accuracy_mock.assert_called_once()

    @patch("neurox.interpretation.metrics.f1")
    def test_compute_score_f1(self, f1_mock):
        "F1"
        y = [1 for _ in range(50)]
        score = metrics.compute_score(y, y, "f1")

        f1_mock.assert_called_once()

    @patch("neurox.interpretation.metrics.accuracy_and_f1")
    def test_compute_score_accuracy_and_f1(self, accuracy_and_f1_mock):
        "Mean of Accuracy and F1"
        y = [1 for _ in range(50)]
        score = metrics.compute_score(y, y, "accuracy_and_f1")

        accuracy_and_f1_mock.assert_called_once()

    @patch("neurox.interpretation.metrics.pearson")
    def test_compute_score_pearson(self, pearson_mock):
        "Pearson's Correlation Coefficient"
        y = [1 for _ in range(50)]
        score = metrics.compute_score(y, y, "pearson")

        pearson_mock.assert_called_once()

    @patch("neurox.interpretation.metrics.spearman")
    def test_compute_score_spearman(self, spearman_mock):
        "Spearman's Correlation Coefficient"
        y = [1 for _ in range(50)]
        score = metrics.compute_score(y, y, "spearman")

        spearman_mock.assert_called_once()

    @patch("neurox.interpretation.metrics.pearson_and_spearman")
    def test_compute_score_pearson_and_spearman(self, pearson_and_spearman_mock):
        "Mean of Pearson's and Spearman's Correlation Coefficient"
        y = [1 for _ in range(50)]
        score = metrics.compute_score(y, y, "pearson_and_spearman")

        pearson_and_spearman_mock.assert_called_once()

    @patch("neurox.interpretation.metrics.matthews_corrcoef")
    def test_compute_score_matthews_corrcoef(self, matthews_corrcoef_mock):
        "Matthew's Correlation Coefficient"
        y = [1 for _ in range(50)]
        score = metrics.compute_score(y, y, "matthews_corrcoef")

        matthews_corrcoef_mock.assert_called_once()
