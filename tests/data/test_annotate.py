import re
import unittest

from io import StringIO
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import neurox.data.annotate as annotate
import neurox.data.loader as data_loader
import neurox.data.utils as data_utils
import torch


class TestCreateBinaryData(unittest.TestCase):
    def setUp(self):
        self.num_layers = 13
        test_sentences = [
            "Hello , this test is test 1 .",
            "Hello , this is another test number 2 .",
            "This is test # 3 .",
            "This book is a good book",
            "And finally , this is the last test !",
        ]
        self.tokens = {
            "source": [s.split(" ") for s in test_sentences],
        }
        self.activations = [
            torch.rand((len(sentence), self.num_layers * 768))
            for sentence in self.tokens["source"]
        ]

    def tearDown(self):
        pass

    def test_basic_binary_annotation(self):
        "Basic binary annotation"

        words, labels, activations = annotate._create_binary_data(
            self.tokens, self.activations, {"this"}
        )

        self.assertEqual(len(words), len(labels))
        self.assertEqual(len(words), len(activations))

    def test_basic_binary_annotation_word(self):
        "Basic binary annotation with single word"

        test_word = "Hello"
        words, labels, activations = annotate._create_binary_data(
            self.tokens, self.activations, {test_word}
        )

        self.assertIn(test_word, words)

    def test_basic_binary_annotation_activation(self):
        "Basic binary annotation with single word and its activation"

        test_word = "another"
        words, labels, activations = annotate._create_binary_data(
            self.tokens, self.activations, {test_word}
        )

        self.assertIn(test_word, words)

        idx = words.index(test_word)
        self.assertTrue(torch.equal(activations[idx], self.activations[1][4, :]))

    def test_duplicate_words_in_a_sentence(self):
        "Check multiple occurrences of a word in a sentence"

        test_word = "book"
        words, labels, activations = annotate._create_binary_data(
            self.tokens, self.activations, {test_word}
        )

        self.assertIn(test_word, words)
        self.assertEqual(words.count(test_word), 2)

        idx = [i for i, w in enumerate(words) if w == test_word]

        self.assertTrue(torch.equal(activations[idx[0]], self.activations[3][1, :]))
        self.assertTrue(torch.equal(activations[idx[1]], self.activations[3][5, :]))

    def test_duplicate_words_across_a_sentence(self):
        "Check multiple occurrences of a word in a sentence"

        test_word = "test"
        words, labels, activations = annotate._create_binary_data(
            self.tokens, self.activations, {test_word}
        )

        self.assertIn(test_word, words)
        self.assertEqual(words.count(test_word), 5)

        idx = [i for i, w in enumerate(words) if w == test_word]

        self.assertTrue(torch.equal(activations[idx[0]], self.activations[0][3, :]))
        self.assertTrue(torch.equal(activations[idx[1]], self.activations[0][5, :]))
        self.assertTrue(torch.equal(activations[idx[2]], self.activations[1][5, :]))
        self.assertTrue(torch.equal(activations[idx[3]], self.activations[2][2, :]))
        self.assertTrue(torch.equal(activations[idx[4]], self.activations[4][7, :]))

    def test_number_of_positive_negative_examples(self):
        "Check that positive and negative examples are equal if balance_data is True"

        test_word = "test"
        words, labels, activations = annotate._create_binary_data(
            self.tokens, self.activations, {test_word}, balance_data=True
        )

        self.assertEqual(words.count(test_word), 5)
        self.assertEqual(labels.count("positive"), labels.count("negative"))

    def test_if_positive_class_is_zero(self):
        "Check if specific pattern or word does not exist in the list of sentences. Positive class will be zero"

        test_word = "abc"
        self.assertRaises(
            ValueError,
            annotate._create_binary_data,
            self.tokens,
            self.activations,
            {test_word},
        )

    def test_if_negative_class_is_zero(self):
        "Check if negative class is zero since all words are associated with a positive class"

        test_regex = re.compile(r"^.+$")
        self.assertRaises(
            ValueError,
            annotate._create_binary_data,
            self.tokens,
            self.activations,
            test_regex,
        )

    def test_regex_based_annotation(self):
        "Basic regex based binary filter test"
        test_regex = re.compile(r"^\d$")
        words, labels, activations = annotate._create_binary_data(
            self.tokens, self.activations, test_regex
        )

        self.assertIn("1", words)
        self.assertIn("2", words)
        self.assertIn("3", words)

        self.assertTrue(
            torch.equal(activations[words.index("1")], self.activations[0][6, :])
        )
        self.assertTrue(
            torch.equal(activations[words.index("2")], self.activations[1][7, :])
        )
        self.assertTrue(
            torch.equal(activations[words.index("3")], self.activations[2][4, :])
        )

    def test_binary_filter_invalid(self):
        "Test if binary filter is wrongly specified"
        self.assertRaises(
            NotImplementedError,
            annotate._create_binary_data,
            self.tokens,
            self.activations,
            "abc",
        )

    @patch("sys.stdout", new_callable=StringIO)
    def test_warning_negative_class_is_smaller(self, mock_stdout):
        "print a warning if negative class is smaller than the positive class"

        test_regex = re.compile(r"^\w+$")
        words, labels, activations = annotate._create_binary_data(
            self.tokens, self.activations, test_regex
        )

        self.assertIn(
            "WARNING: the negative class examples are less than the positive class examples",
            mock_stdout.getvalue(),
        )

    def test_function_based_annotation(self):
        "Basic function based binary filter test"
        test_fn = lambda x: "t" in x
        words, labels, activations = annotate._create_binary_data(
            self.tokens, self.activations, test_fn
        )

        # 11 words contain "t"
        self.assertEqual(labels.count("positive"), 11)


# TODO tests
# "Check if positive examples are getting label 'positive' and vice verse"
# Check function filter type


class TestAnnotateData(unittest.TestCase):
    def setUp(self):
        self.num_layers = 13
        self.test_sentences = [
            "Hello , this test is test 1 .",
            "Hello , this is another test number 2 .",
            "This is test # 3 .",
            "This book is a good book",
            "And finally , this is the last test !",
        ]
        self.tokens = {
            "source": [s.split(" ") for s in self.test_sentences],
        }
        self.activations = [
            torch.rand((self.num_layers, len(sentence), 768))
            for sentence in self.tokens["source"]
        ]
        self.tmpdir = TemporaryDirectory()
        data_utils.save_files(
            self.test_sentences,
            self.test_sentences,
            self.activations,
            f"{self.tmpdir.name}/gold",
        )

    @patch("neurox.data.annotate._create_binary_data")
    def test_binary_data_wrapper(self, mock_create_binary_data):
        mock_create_binary_data.return_value = (
            self.test_sentences,
            self.test_sentences,
            self.activations,
        )

        annotate.annotate_data(
            f"{self.tmpdir.name}/gold.word",
            f"{self.tmpdir.name}/gold.hdf5",
            {"test"},
            f"{self.tmpdir.name}/test",
        )

        with open(f"{self.tmpdir.name}/test.word") as fp:
            for line_idx, line in enumerate(fp):
                self.assertEqual(self.test_sentences[line_idx], line.strip())

        # Load and check activations as well
        test_activations, test_num_layers = data_loader.load_activations(
            f"{self.tmpdir.name}/test.hdf5"
        )
        self.assertEqual(self.num_layers, test_num_layers)

        gold_activations = [a.reshape((a.shape[1], -1)) for a in self.activations]
        for act_idx, act in enumerate(test_activations):
            self.assertTrue(
                torch.allclose(gold_activations[act_idx], torch.FloatTensor(act))
            )

        # # Check hdf5 structure
        # self.assertEqual(len(saved_activations.keys()), len(self.test_sentences) + 1)
        # self.assertTrue("sentence_to_index" in saved_activations)
        # self.assertTrue(torch.equal(torch.FloatTensor(saved_activations[idx]), self.expected_activations[int(idx)]))
