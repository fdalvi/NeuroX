import json
import os
import unittest

from io import StringIO
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import h5py

import neurox.data.extraction.transformers_extractor as transformers_extractor
import numpy as np
import torch


class MockTokenizer(object):
    all_special_tokens = ["<s>", "</s>"]
    vocab = {"a", "b", "c", *all_special_tokens}
    token_to_idx = {t: idx for idx, t in enumerate(vocab)}
    idx_to_tokens = {idx: t for t, idx in token_to_idx.items()}

    def convert_tokens_to_ids(self, tokens):
        ids = []
        for tok in tokens:
            if tok not in self.token_to_idx:
                self.vocab.add(tok)
                idx = len(self.token_to_idx)
                self.token_to_idx[tok] = idx
                self.idx_to_tokens[idx] = tok
            ids.append(self.token_to_idx[tok])

        return ids

    def convert_ids_to_tokens(self, ids):
        return [self.idx_to_tokens.get(idx, "UNSEEN") for idx in ids]

    def __call__(self, text):
        return {
            "input_ids": self.convert_tokens_to_ids(
                f"<s> {text.strip()} </s>".split(" ")
            )
        }


class TestAggregation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # State is 3 layers, 4 tokens, 2 features per token
        cls.state = np.random.random((3, 4, 2))

    @classmethod
    def tearDownClass(cls):
        cls.state = None

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_aggregate_repr_first(self):
        "First subword aggregation"
        self.assertTrue(
            np.array_equal(
                self.state[:, 0, :].squeeze(),
                transformers_extractor.aggregate_repr(self.state, 0, 2, "first"),
            )
        )
        self.assertTrue(
            np.array_equal(
                self.state[:, 2, :].squeeze(),
                transformers_extractor.aggregate_repr(self.state, 2, 2, "first"),
            )
        )
        self.assertTrue(
            np.array_equal(
                self.state[:, 1, :].squeeze(),
                transformers_extractor.aggregate_repr(self.state, 1, 3, "first"),
            )
        )

    def test_aggregate_repr_last(self):
        "Last subword aggregation"
        self.assertTrue(
            np.array_equal(
                self.state[:, 2, :].squeeze(),
                transformers_extractor.aggregate_repr(self.state, 0, 2, "last"),
            )
        )
        self.assertTrue(
            np.array_equal(
                self.state[:, 2, :].squeeze(),
                transformers_extractor.aggregate_repr(self.state, 2, 2, "last"),
            )
        )
        self.assertTrue(
            np.array_equal(
                self.state[:, 3, :].squeeze(),
                transformers_extractor.aggregate_repr(self.state, 1, 3, "last"),
            )
        )

    def test_aggregate_repr_average(self):
        "Average subword aggregation"
        self.assertTrue(
            np.array_equal(
                np.average(self.state[:, 0:3, :], axis=1),
                transformers_extractor.aggregate_repr(self.state, 0, 2, "average"),
            )
        )
        self.assertTrue(
            np.array_equal(
                np.average(self.state[:, 2:3, :], axis=1),
                transformers_extractor.aggregate_repr(self.state, 2, 2, "average"),
            )
        )
        self.assertTrue(
            np.array_equal(
                np.average(self.state[:, 1:4, :], axis=1),
                transformers_extractor.aggregate_repr(self.state, 1, 3, "average"),
            )
        )


class TestExtraction(unittest.TestCase):
    @classmethod
    @patch("transformers.BertTokenizer")
    @patch("transformers.BertModel")
    def setUpClass(cls, model_mock, tokenizer_mock):
        cls.num_layers = 13
        cls.num_neurons_per_layer = 768

        # Input format is "TOKEN_{num_subwords}"
        # UNK is used when num_subwords == 0
        # Token is dropped completely when num_subwords < 0
        sentences = [
            ("Single token sentence without any subwords", ["TOKEN_1"]),
            ("Multi token sentence without any subwords", ["TOKEN_1", "TOKEN_1"]),
            ("Subword token in the beginning", ["SUBTOKEN_2", "TOKEN_1"]),
            ("Subword token in the middle", ["STOKEN_1", "SUBTOKEN_2", "ETOKEN_1"]),
            ("Subword token in the end", ["TOKEN_1", "SUBTOKEN_2"]),
            ("Multiple subword tokens", ["SOMETHING_2", "TOKEN_4"]),
            (
                "Multiple subword tokens with unknown token",
                ["TOKEN_2", "TOKEN_2", "TOKEN_0"],
            ),
            ("All unknown tokens", ["SOMETHING_0", "SOMETHING2_0", "SOMETHING3_0"]),
            ("Token that is dropped by tokenizer", ["DISAPPEAR_-1"]),
            (
                "Token in the beginning that is dropped by tokenizer in context",
                ["DISAPPEAR_-1", "SOMETHING_2"],
            ),
            (
                "Token in the middle that is dropped by tokenizer in context",
                ["SOMETHING_2", "DISAPPEAR_-1", "ANOTHER_4"],
            ),
            (
                "Token in the end that is dropped by tokenizer in context",
                ["SOMETHING_3", "DISAPPEAR_-1"],
            ),
            (
                "Input longer than tokenizer's limit",
                ["SOMETHING_100", "ANOTHER_300", "MIDDLE_110", "FINAL_100"],
            ),
            (
                "Input exactly equal to tokenizer's limit",
                ["SOMETHING_100", "ANOTHER_300", "FINAL_110"],
            ),
            (
                "Input longer than tokenizer's limit with break in the middle of tokenization",
                ["SOMETHING_100", "ANOTHER_300", "FINAL_200"],
            ),
            (
                "Input exactly equal to tokenizer's limit with dropped token",
                ["SOMETHING_100", "ANOTHER_300", "FINAL_110", "DISAPPEAR_-1"],
            ),
            (
                "Input longer than tokenizer's limit with break at dropped token",
                [
                    "SOMETHING_100",
                    "ANOTHER_300",
                    "MIDDLE_110",
                    "DISAPPEAR_-1",
                    "FINAL_100",
                ],
            ),
            (
                "Input longer than tokenizer's limit with dropped token",
                [
                    "SOMETHING_100",
                    "DISAPPEAR_-1",
                    "ANOTHER_300",
                    "MIDDLE_110",
                    "FINAL_100",
                ],
            ),
            (
                "Varying tokenization for same token",
                [
                    "VARYING_1",
                    "ANOTHER_1",
                    "VARYING_1",
                    "ANOTHER_1",
                    "TOKEN_4",
                    "VARYING_1",
                ],
            ),
            (
                "Varying tokenization for same token with unknown tokens",
                [
                    "VARYING_1",
                    "ANOTHER_1",
                    "VARYING_1",
                    "TOKEN_0",
                    "TOKEN_4",
                    "STOKEN_0",
                    "VARYING_1",
                ],
            ),
        ]
        cls.tests_data = []

        # Create mock model and tokenizer
        model_mock = MagicMock()
        tokenizer_mock = MagicMock()
        tokenizer_mock.all_special_tokens = ["[CLS]", "[UNK]", "[SEP]"]
        tokenizer_mock.unk_token = "[UNK]"
        tokenizer_mock.model_max_length = 512

        tokenization_mapping = {
            k: i for i, k in enumerate(tokenizer_mock.all_special_tokens)
        }
        tokenization_mapping["a"] = len(tokenization_mapping)

        def word_to_subwords(word, position=None):
            if "_" not in word:
                return [word]
            actual_word, occurrence = word.split("_")
            if position is not None and actual_word == "VARYING":
                if position == "start":
                    actual_word = "VARYINGSTART"
                    occurrence = (
                        5  # broken into 5 subwords if word occurs at the beginning
                    )
                elif position == "end":
                    actual_word = "VARYINGEND"
                    occurrence = 3  # broken into 3 subwords if word occurs at the end
                elif position == "middle":
                    occurrence = (
                        4  # broken into 4 subwords if word occurs at the middle
                    )
            else:
                occurrence = int(occurrence)
            if occurrence == 0:
                return [tokenizer_mock.unk_token]
            elif occurrence < 0:
                return []
            else:
                subwords = []
                for o_idx in range(occurrence):
                    if o_idx == 0:
                        subwords.append(f"{actual_word}_{o_idx+1}")
                    else:
                        subwords.append(f"@@{actual_word}_{o_idx+1}")
                return subwords

        for _, sentence in sentences:
            for word_idx, word in enumerate(sentence):
                if word_idx == 0:
                    position = "start"
                elif word_idx == len(sentence) - 1:
                    position = "end"
                else:
                    position = "middle"
                for subword in word_to_subwords(word, position):
                    if subword not in tokenization_mapping:
                        tokenization_mapping[subword] = len(tokenization_mapping)

        inverse_tokenization_mapping = {v: k for k, v in tokenization_mapping.items()}

        def tokenization_side_effect(arg):
            # Truncate input at 512 tokens
            arg = arg[:512]

            return [tokenization_mapping[w] for w in arg]

        tokenizer_mock.convert_tokens_to_ids.side_effect = tokenization_side_effect

        def inverse_tokenization_side_effect(arg):
            # Truncate input at 512 tokens
            arg = arg[:512]

            return [inverse_tokenization_mapping[i] for i in arg]

        tokenizer_mock.convert_ids_to_tokens.side_effect = (
            inverse_tokenization_side_effect
        )

        def encode_side_effect(arg, **kwargs):
            tokenized_sentence = ["[CLS]"]
            words = arg.split(" ")
            for w_idx, w in enumerate(words):
                if w_idx == 0:
                    position = "start"
                elif w_idx == len(words) - 1:
                    position = "end"
                else:
                    position = "middle"
                tokenized_sentence.extend(word_to_subwords(w, position))
            if (
                kwargs.get("truncation", False)
                and len(tokenized_sentence) >= tokenizer_mock.model_max_length
            ):
                tokenized_sentence = tokenized_sentence[:511]
            tokenized_sentence.append("[SEP]")

            return tokenization_side_effect(tokenized_sentence)

        tokenizer_mock.encode.side_effect = encode_side_effect

        # Build Expected outputs
        for test_description, sentence in sentences:
            counter = 0
            model_mock_output = {k: [] for k in range(cls.num_layers)}
            first_expected_output = {k: [] for k in range(cls.num_layers)}
            last_expected_output = {k: [] for k in range(cls.num_layers)}
            average_expected_output = {k: [] for k in range(cls.num_layers)}
            special_tokens_expected_output = {k: [] for k in range(cls.num_layers)}

            idx = [counter]
            tokenized_sentence = ["[CLS]"]
            for k in model_mock_output:
                tmp = torch.rand((1, cls.num_neurons_per_layer))
                model_mock_output[k].append(tmp)
                special_tokens_expected_output[k].append(tmp.squeeze())

            counter += 1

            for w_idx, w in enumerate(sentence):
                idx.append(counter)
                if w_idx == 0:
                    position = "start"
                elif w_idx == len(sentence) - 1:
                    position = "end"
                else:
                    position = "middle"
                subwords = word_to_subwords(w, position)
                counter += len(subwords)
                tokenized_sentence.extend(subwords)

                for k in model_mock_output:
                    num_subtokens = len(subwords)
                    # Account for truncation in the middle of tokenization
                    num_extra_tokens = len(tokenized_sentence) - (
                        tokenizer_mock.model_max_length - 1
                    )
                    if num_extra_tokens > 0:
                        num_subtokens -= num_extra_tokens

                    # Output is zero array if the number of subwords is 0, i.e.
                    # a token dropped by the tokenizer
                    if num_subtokens > 0:
                        tmp = torch.rand((num_subtokens, cls.num_neurons_per_layer))
                        model_mock_output[k].append(tmp)
                    else:
                        tmp = torch.zeros((1, cls.num_neurons_per_layer))

                    # Check if input is too long already (and account
                    # for [SEP] token)
                    if len(tokenized_sentence) > (tokenizer_mock.model_max_length - 1):
                        continue
                    first_expected_output[k].append(tmp[0, :])  # Pick first subword idx
                    last_expected_output[k].append(tmp[-1, :])  # Pick last subword idx
                    average_expected_output[k].append(tmp.mean(axis=0))
                    special_tokens_expected_output[k].append(tmp[-1, :])

                # Check if input is too long already (and account
                # for [SEP] token)
                if len(tokenized_sentence) > (tokenizer_mock.model_max_length - 1):
                    break

            idx.append(counter)
            tokenized_sentence.append("[SEP]")
            for k in model_mock_output:
                tmp = torch.rand((1, cls.num_neurons_per_layer))
                model_mock_output[k].append(tmp)
                special_tokens_expected_output[k].append(tmp.squeeze())

            counter += 1

            model_mock_output = tuple(
                [
                    torch.cat(model_mock_output[k]).unsqueeze(0)
                    for k in range(cls.num_layers)
                ]
            )
            first_expected_output = tuple(
                [torch.stack(first_expected_output[k]) for k in range(cls.num_layers)]
            )
            last_expected_output = tuple(
                [torch.stack(last_expected_output[k]) for k in range(cls.num_layers)]
            )
            average_expected_output = tuple(
                [torch.stack(average_expected_output[k]) for k in range(cls.num_layers)]
            )
            special_tokens_expected_output = tuple(
                [
                    torch.stack(special_tokens_expected_output[k])
                    for k in range(cls.num_layers)
                ]
            )

            cls.tests_data.append(
                (
                    test_description,
                    sentence,
                    model_mock_output,
                    first_expected_output,
                    last_expected_output,
                    average_expected_output,
                    special_tokens_expected_output,
                )
            )

        cls.model = model_mock
        cls.tokenizer = tokenizer_mock

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def run_test(self, testcase, dropped_tokens=0, **kwargs):
        (
            _,
            sentence,
            model_mock_output,
            first_expected_output,
            last_expected_output,
            average_expected_output,
            special_tokens_expected_output,
        ) = testcase
        if kwargs["aggregation"] == "first":
            expected_output = first_expected_output
            extra_tokens = 0
        if kwargs["aggregation"] == "last":
            expected_output = last_expected_output
            extra_tokens = 0
        if kwargs["aggregation"] == "average":
            expected_output = average_expected_output
            extra_tokens = 0
        if (
            "include_special_tokens" in kwargs
            and kwargs["include_special_tokens"] == True
        ):
            expected_output = special_tokens_expected_output

            # Account for [CLS] and [SEP]
            extra_tokens = 2

        extra_tokens -= dropped_tokens
        self.model.return_value = ("placeholder", model_mock_output)

        words = sentence
        (
            hidden_states,
            extracted_words,
        ) = transformers_extractor.extract_sentence_representations(
            " ".join(words), self.model, self.tokenizer, **kwargs
        )
        expected_length = min(512, len(words) + extra_tokens)
        self.assertEqual(len(extracted_words), expected_length)
        self.assertEqual(hidden_states.shape[1], expected_length)

        # Test output from all layers
        for l in range(self.num_layers):
            np.testing.assert_array_almost_equal(
                hidden_states[l, :, :], expected_output[l][:, :].numpy()
            )

    ########################### First tests ############################
    def test_extract_sentence_representations_first_aggregation_multiple_token(self):
        "First aggregation: Multi token sentence without any subwords"
        self.run_test(self.tests_data[1], aggregation="first")

    def test_extract_sentence_representations_first_aggregation_subword_begin(self):
        "First aggregation: Subword token in the beginning"
        self.run_test(self.tests_data[2], aggregation="first")

    def test_extract_sentence_representations_first_aggregation_subword_middle(self):
        "First aggregation: Subword token in the middle"
        self.run_test(self.tests_data[3], aggregation="first")

    def test_extract_sentence_representations_first_aggregation_subword_end(self):
        "First aggregation: Subword token in the end"
        self.run_test(self.tests_data[4], aggregation="first")

    def test_extract_sentence_representations_first_aggregation_mutliple_subwords(self):
        "First aggregation: Multiple subword tokens"
        self.run_test(self.tests_data[5], aggregation="first")

    def test_extract_sentence_representations_first_aggregation_mutliple_subwords_with_unk(
        self,
    ):
        "First aggregation: Multiple subword tokens with unknown token"
        self.run_test(self.tests_data[6], aggregation="first")

    def test_extract_sentence_representations_first_aggregation_all_unk(self):
        "First aggregation: All unknown tokens"
        self.run_test(self.tests_data[7], aggregation="first")

    def test_extract_sentence_representations_first_aggregation_dropped_token(
        self,
    ):
        "First aggregation: Token that is dropped by tokenizer"
        self.run_test(self.tests_data[8], aggregation="first")

    def test_extract_sentence_representations_first_aggregation_dropped_token_beginning(
        self,
    ):
        "First aggregation: Token in the beginning that is dropped by tokenizer in context"
        self.run_test(self.tests_data[9], aggregation="first")

    def test_extract_sentence_representations_first_aggregation_dropped_token_middle(
        self,
    ):
        "First aggregation: Token in the middle that is dropped by tokenizer in context"
        self.run_test(self.tests_data[10], aggregation="first")

    def test_extract_sentence_representations_first_aggregation_dropped_token_end(
        self,
    ):
        "First aggregation: Token in the end that is dropped by tokenizer in context"
        self.run_test(self.tests_data[11], aggregation="first")

    ############################ Last tests ############################
    def test_extract_sentence_representations_last_aggregation_multiple_token(self):
        "Last aggregation: Multi token sentence without any subwords"
        self.run_test(self.tests_data[1], aggregation="last")

    def test_extract_sentence_representations_last_aggregation_subword_begin(self):
        "Last aggregation: Subword token in the beginning"
        self.run_test(self.tests_data[2], aggregation="last")

    def test_extract_sentence_representations_last_aggregation_subword_middle(self):
        "Last aggregation: Subword token in the middle"
        self.run_test(self.tests_data[3], aggregation="last")

    def test_extract_sentence_representations_last_aggregation_subword_end(self):
        "Last aggregation: Subword token in the end"
        self.run_test(self.tests_data[4], aggregation="last")

    def test_extract_sentence_representations_last_aggregation_mutliple_subwords(self):
        "Last aggregation: Multiple subword tokens"
        self.run_test(self.tests_data[5], aggregation="last")

    def test_extract_sentence_representations_last_aggregation_mutliple_subwords_with_unk(
        self,
    ):
        "Last aggregation: Multiple subword tokens with unknown token"
        self.run_test(self.tests_data[6], aggregation="last")

    def test_extract_sentence_representations_last_aggregation_all_unk(self):
        "Last aggregation: All unknown tokens"
        self.run_test(self.tests_data[7], aggregation="last")

    def test_extract_sentence_representations_last_aggregation_dropped_token(
        self,
    ):
        "Last aggregation: Token that is dropped by tokenizer"
        self.run_test(self.tests_data[8], aggregation="last")

    def test_extract_sentence_representations_last_aggregation_dropped_token_beginning(
        self,
    ):
        "Last aggregation: Token in the beginning that is dropped by tokenizer in context"
        self.run_test(self.tests_data[9], aggregation="last")

    def test_extract_sentence_representations_last_aggregation_dropped_token_middle(
        self,
    ):
        "Last aggregation: Token in the middle that is dropped by tokenizer in context"
        self.run_test(self.tests_data[10], aggregation="last")

    def test_extract_sentence_representations_last_aggregation_dropped_token_end(
        self,
    ):
        "Last aggregation: Token in the end that is dropped by tokenizer in context"
        self.run_test(self.tests_data[11], aggregation="last")

    ########################## Average tests ###########################
    def test_extract_sentence_representations_average_aggregation_single_token(self):
        "Average aggregation: Single token sentence without any subwords"
        self.run_test(self.tests_data[0], aggregation="average")

    def test_extract_sentence_representations_average_aggregation_multiple_token(self):
        "Average aggregation: Multi token sentence without any subwords"
        self.run_test(self.tests_data[1], aggregation="average")

    def test_extract_sentence_representations_average_aggregation_subword_begin(self):
        "Average aggregation: Subword token in the beginning"
        self.run_test(self.tests_data[2], aggregation="average")

    def test_extract_sentence_representations_average_aggregation_subword_middle(self):
        "Average aggregation: Subword token in the middle"
        self.run_test(self.tests_data[3], aggregation="average")

    def test_extract_sentence_representations_average_aggregation_subword_end(self):
        "Average aggregation: Subword token in the end"
        self.run_test(self.tests_data[4], aggregation="average")

    def test_extract_sentence_representations_average_aggregation_mutliple_subwords(
        self,
    ):
        "Average aggregation: Multiple subword tokens"
        self.run_test(self.tests_data[5], aggregation="average")

    def test_extract_sentence_representations_average_aggregation_mutliple_subwords_with_unk(
        self,
    ):
        "Average aggregation: Multiple subword tokens with unknown token"
        self.run_test(self.tests_data[6], aggregation="average")

    def test_extract_sentence_representations_average_aggregation_all_unk(self):
        "Average aggregation: All unknown tokens"
        self.run_test(self.tests_data[7], aggregation="average")

    def test_extract_sentence_representations_average_aggregation_dropped_token(
        self,
    ):
        "Average aggregation: Token that is dropped by tokenizer"
        self.run_test(self.tests_data[8], aggregation="average")

    def test_extract_sentence_representations_average_aggregation_dropped_token_beginning(
        self,
    ):
        "Average aggregation: Token in the beginning that is dropped by tokenizer in context"
        self.run_test(self.tests_data[9], aggregation="average")

    def test_extract_sentence_representations_average_aggregation_dropped_token_middle(
        self,
    ):
        "Average aggregation: Token in the middle that is dropped by tokenizer in context"
        self.run_test(self.tests_data[10], aggregation="average")

    def test_extract_sentence_representations_average_aggregation_dropped_token_end(
        self,
    ):
        "Average aggregation: Token in the end that is dropped by tokenizer in context"
        self.run_test(self.tests_data[11], aggregation="average")

    ############################# Embedding tests ##############################
    def test_extract_sentence_representations_include_embeddings(self):
        "Extraction with embedding layer"
        _, sentence, model_mock_output, _, expected_output, _, _ = self.tests_data[1]
        self.model.return_value = ("placeholder", model_mock_output)

        (
            hidden_states,
            extracted_words,
        ) = transformers_extractor.extract_sentence_representations(
            " ".join(sentence), self.model, self.tokenizer, include_embeddings=True
        )

        self.assertEqual(hidden_states.shape[0], self.num_layers)

        for l in range(self.num_layers):
            np.testing.assert_array_almost_equal(
                hidden_states[l, :, :], expected_output[l][:, :].numpy()
            )

    def test_extract_sentence_representations_exclude_embeddings(self):
        "Extraction without embedding layer"
        _, sentence, model_mock_output, _, expected_output, _, _ = self.tests_data[1]
        self.model.return_value = ("placeholder", model_mock_output)

        (
            hidden_states,
            extracted_words,
        ) = transformers_extractor.extract_sentence_representations(
            " ".join(sentence), self.model, self.tokenizer, include_embeddings=False
        )

        self.assertEqual(hidden_states.shape[0], self.num_layers - 1)

        for l in range(1, self.num_layers):
            np.testing.assert_array_almost_equal(
                hidden_states[l - 1, :, :], expected_output[l][:, :].numpy()
            )

    ############################ Long Input tests #############################
    @patch("sys.stdout", new_callable=StringIO)
    def test_extract_sentence_representations_long_input(self, mock_stdout):
        "Input longer than tokenizer's limit"
        self.run_test(self.tests_data[12], dropped_tokens=1, aggregation="average")
        self.assertIn("Input truncated because of length", mock_stdout.getvalue())

    def test_extract_sentence_representations_long_input_exact_length(self):
        "Input exactly equal to tokenizer's limit"
        self.run_test(self.tests_data[13], aggregation="average")

    @patch("sys.stdout", new_callable=StringIO)
    def test_extract_sentence_representations_long_input_tokenization_break(
        self, mock_stdout
    ):
        "Input longer than tokenizer's limit with break in the middle of tokenization"
        self.run_test(self.tests_data[14], dropped_tokens=1, aggregation="average")
        self.assertIn("Input truncated because of length", mock_stdout.getvalue())

    def test_extract_sentence_representations_long_input_exact_length_dropped_token(
        self,
    ):
        "Input exactly equal to tokenizer's limit with dropped token"
        self.run_test(self.tests_data[15], aggregation="average")

    @patch("sys.stdout", new_callable=StringIO)
    def test_extract_sentence_representations_long_input_dropped_token_break(
        self, mock_stdout
    ):
        "Input longer than tokenizer's limit with break at dropped token"
        self.run_test(self.tests_data[16], dropped_tokens=1, aggregation="average")
        self.assertIn("Input truncated because of length", mock_stdout.getvalue())

    @patch("sys.stdout", new_callable=StringIO)
    def test_extract_sentence_representations_long_input_dropped_token(
        self, mock_stdout
    ):
        "Input longer than tokenizer's limit with dropped token"
        self.run_test(self.tests_data[17], dropped_tokens=1, aggregation="average")
        self.assertIn("Input truncated because of length", mock_stdout.getvalue())

    ####################### Varying tokenization tests ########################
    def test_extract_sentence_representations_varying_tokenization(self):
        "Same token with different in-context tokenizations"
        _, sentence, model_mock_output, _, expected_output, _, _ = self.tests_data[18]
        self.model.return_value = ("placeholder", model_mock_output)

        (
            hidden_states,
            extracted_words,
        ) = transformers_extractor.extract_sentence_representations(
            " ".join(sentence), self.model, self.tokenizer
        )

        for l in range(1, self.num_layers):
            np.testing.assert_array_almost_equal(
                hidden_states[l, :, :], expected_output[l][:, :].numpy()
            )

    def test_extract_sentence_representations_varying_tokenization_with_unk(self):
        "Same token with different in-context tokenizations with unknown tokens"
        _, sentence, model_mock_output, _, expected_output, _, _ = self.tests_data[19]
        self.model.return_value = ("placeholder", model_mock_output)

        (
            hidden_states,
            extracted_words,
        ) = transformers_extractor.extract_sentence_representations(
            " ".join(sentence), self.model, self.tokenizer
        )

        for l in range(1, self.num_layers):
            np.testing.assert_array_almost_equal(
                hidden_states[l, :, :], expected_output[l][:, :].numpy()
            )

    ############################# Special Tokens ##############################
    def test_extract_sentence_representations_special_tokens_multiple_token(self):
        "Special Tokens Extraction: Multi token sentence without any subwords"
        self.run_test(
            self.tests_data[1], aggregation="last", include_special_tokens=True
        )

    def test_extract_sentence_representations_special_tokens_subword_begin(self):
        "Special Tokens Extraction: Subword token in the beginning"
        self.run_test(
            self.tests_data[2], aggregation="last", include_special_tokens=True
        )

    def test_extract_sentence_representations_special_tokens_subword_middle(self):
        "Special Tokens Extraction: Subword token in the middle"
        self.run_test(
            self.tests_data[3], aggregation="last", include_special_tokens=True
        )

    def test_extract_sentence_representations_special_tokens_subword_end(self):
        "Special Tokens Extraction: Subword token in the end"
        self.run_test(
            self.tests_data[4], aggregation="last", include_special_tokens=True
        )

    def test_extract_sentence_representations_special_tokens_mutliple_subwords(self):
        "Special Tokens Extraction: Multiple subword tokens"
        self.run_test(
            self.tests_data[5], aggregation="last", include_special_tokens=True
        )

    def test_extract_sentence_representations_special_tokens_mutliple_subwords_with_unk(
        self,
    ):
        "Special Tokens Extraction: Multiple subword tokens with unknown token"
        self.run_test(
            self.tests_data[6], aggregation="last", include_special_tokens=True
        )

    def test_extract_sentence_representations_special_tokens_all_unk(self):
        "Special Tokens Extraction: All unknown tokens"
        self.run_test(
            self.tests_data[7], aggregation="last", include_special_tokens=True
        )

    def test_extract_sentence_representations_special_tokens_dropped_token(
        self,
    ):
        "Special Tokens Extraction: Dropped token between two special tokens"
        _, sentence, model_mock_output, _, _, _, expected_output = self.tests_data[8]
        self.model.return_value = ("placeholder", model_mock_output)

        with self.assertRaises(Exception) as error_context:
            transformers_extractor.extract_sentence_representations(
                " ".join(sentence),
                self.model,
                self.tokenizer,
                include_special_tokens=True,
            )

        self.assertIn(
            "token dropped by the tokenizer appeared next",
            error_context.exception.args[0],
        )

    def test_extract_sentence_representations_special_tokens_dropped_token_beginning(
        self,
    ):
        "Special Tokens Extraction: Dropped token after a Special token"
        _, sentence, model_mock_output, _, _, _, expected_output = self.tests_data[9]
        self.model.return_value = ("placeholder", model_mock_output)

        with self.assertRaises(Exception) as error_context:
            transformers_extractor.extract_sentence_representations(
                " ".join(sentence),
                self.model,
                self.tokenizer,
                include_special_tokens=True,
            )

        self.assertIn(
            "token dropped by the tokenizer appeared next",
            error_context.exception.args[0],
        )

    def test_extract_sentence_representations_special_tokens_dropped_token_middle(
        self,
    ):
        "Special Tokens Extraction: Token in the middle that is dropped by tokenizer in context"
        self.run_test(
            self.tests_data[10], aggregation="last", include_special_tokens=True
        )

    def test_extract_sentence_representations_special_tokens_dropped_token_end(
        self,
    ):
        "Special Tokens Extraction: Dropped token before a Special token"
        _, sentence, model_mock_output, _, _, _, expected_output = self.tests_data[11]
        self.model.return_value = ("placeholder", model_mock_output)

        with self.assertRaises(Exception) as error_context:
            transformers_extractor.extract_sentence_representations(
                " ".join(sentence),
                self.model,
                self.tokenizer,
                include_special_tokens=True,
            )

        self.assertIn(
            "token dropped by the tokenizer appeared next",
            error_context.exception.args[0],
        )

    @patch("sys.stdout", new_callable=StringIO)
    def test_extract_sentence_representations_special_tokens_long_input(
        self, mock_stdout
    ):
        "Special Tokens Extraction: Input longer than tokenizer's limit"
        self.run_test(
            self.tests_data[12],
            dropped_tokens=1,
            aggregation="last",
            include_special_tokens=True,
        )
        self.assertIn("Input truncated because of length", mock_stdout.getvalue())

    def test_extract_sentence_representations_special_tokens_long_input_exact_length(
        self,
    ):
        "Special Tokens Extraction: Input exactly equal to tokenizer's limit"
        self.run_test(
            self.tests_data[13], aggregation="last", include_special_tokens=True
        )

    @patch("sys.stdout", new_callable=StringIO)
    def test_extract_sentence_representations_special_tokens_long_input_tokenization_break(
        self, mock_stdout
    ):
        "Special Tokens Extraction: Input longer than tokenizer's limit with break in the middle of tokenization"
        self.run_test(
            self.tests_data[14],
            dropped_tokens=1,
            aggregation="last",
            include_special_tokens=True,
        )
        self.assertIn("Input truncated because of length", mock_stdout.getvalue())

    def test_extract_sentence_representations_special_tokens_long_input_exact_length_dropped_token(
        self,
    ):
        "Special Tokens Extraction: Input exactly equal to tokenizer's limit with dropped token"
        with self.assertRaises(Exception) as error_context:
            self.run_test(
                self.tests_data[15], aggregation="last", include_special_tokens=True
            )
        self.assertIn(
            "token dropped by the tokenizer appeared next",
            error_context.exception.args[0],
        )

    def test_extract_sentence_representations_special_tokens_long_input_dropped_token_break(
        self,
    ):
        "Special Tokens Extraction: Input longer than tokenizer's limit with break at dropped token"
        with self.assertRaises(Exception) as error_context:
            self.run_test(
                self.tests_data[16],
                dropped_tokens=1,
                aggregation="last",
                include_special_tokens=True,
            )
        self.assertIn(
            "token dropped by the tokenizer appeared next",
            error_context.exception.args[0],
        )

    @patch("sys.stdout", new_callable=StringIO)
    def test_extract_sentence_representations_special_tokens_long_input_dropped_token(
        self, mock_stdout
    ):
        "Special Tokens Extraction: Input longer than tokenizer's limit with dropped token"
        self.run_test(
            self.tests_data[17],
            dropped_tokens=1,
            aggregation="last",
            include_special_tokens=True,
        )
        self.assertIn("Input truncated because of length", mock_stdout.getvalue())


class TestModelAndTokenizerGetter(unittest.TestCase):
    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("transformers.AutoModel.from_pretrained")
    def test_get_model_and_tokenizer_normal(self, auto_model_mock, auto_tokenizer_mock):
        """Normal model and tokenizer loading test"""

        # Using torch matrix here to avoid mocking .to(device) in torch
        expected_model = torch.rand((5, 2))
        expected_tokenizer = torch.rand((5,))
        auto_model_mock.return_value = expected_model
        auto_tokenizer_mock.return_value = expected_tokenizer

        model, tokenizer = transformers_extractor.get_model_and_tokenizer(
            "non-existent model"
        )

        self.assertTrue(torch.equal(model, expected_model))
        self.assertTrue(torch.equal(tokenizer, expected_tokenizer))

    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("transformers.AutoModel.from_pretrained")
    def test_get_model_and_tokenizer_simple(self, auto_model_mock, auto_tokenizer_mock):
        """Randomized model and tokenizer loading test"""

        transformers_extractor.get_model_and_tokenizer(
            "non-existent model", random_weights=True
        )

        auto_model_mock.return_value.to.return_value.init_weights.assert_called_once()

    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("transformers.AutoModel.from_pretrained")
    def test_get_model_and_tokenizer_custom(self, auto_model_mock, auto_tokenizer_mock):
        """Tokenizer with name different than model loading test"""

        # Using torch matrix here to avoid mocking .to(device) in torch
        expected_model = torch.rand((5, 2))
        expected_tokenizer = torch.rand((5,))
        auto_model_mock.return_value = expected_model
        auto_tokenizer_mock.side_effect = (
            lambda *args, **kwargs: expected_tokenizer
            if args[0] == "custom-tokenizer"
            else torch.rand((5,))
        )

        model, tokenizer = transformers_extractor.get_model_and_tokenizer(
            "non-existent model,custom-tokenizer"
        )

        self.assertTrue(torch.equal(model, expected_model))
        self.assertTrue(torch.equal(tokenizer, expected_tokenizer))


class TestSaving(unittest.TestCase):
    def setUp(self):
        self.test_sentences = [
            "Hello , this is test 1 .",
            "Hello , this is another test number 2 .",
            "This is test # 3 .",
            "And finally , this is the last test !",
        ]
        self.expected_activations = [
            torch.rand((13, len(sentence.split(" ")), 768))
            for sentence in self.test_sentences
        ]
        self.tmpdir = TemporaryDirectory()

        self.input_file = os.path.join(self.tmpdir.name, "input_file.txt")
        with open(self.input_file, "w") as fp:
            for sentence in self.test_sentences:
                fp.write(sentence + "\n")

        self.call_counter = 0

        def mocked_model(*args, **kwargs):
            sentence = self.test_sentences[self.call_counter]
            activations = self.expected_activations[self.call_counter]

            self.call_counter += 1

            return activations, sentence.split(" ")

        self.mocked_model = mocked_model

    def tearDown(self):
        self.tmpdir.cleanup()

    @patch(
        "neurox.data.extraction.transformers_extractor.extract_sentence_representations"
    )
    @patch("neurox.data.extraction.transformers_extractor.get_model_and_tokenizer")
    def test_save_hdf5(self, get_model_mock, extraction_mock):
        "Saving activations in single hdf5 file"
        get_model_mock.return_value = (None, MockTokenizer())
        extraction_mock.side_effect = self.mocked_model

        output_file = os.path.join(self.tmpdir.name, "output.hdf5")

        transformers_extractor.extract_representations(
            "non-existant model", self.input_file, output_file, output_type="hdf5"
        )

        saved_activations = h5py.File(output_file, "r")

        # Check hdf5 structure
        self.assertEqual(len(saved_activations.keys()), len(self.test_sentences) + 1)
        self.assertTrue("sentence_to_index" in saved_activations)
        for idx in range(len(self.test_sentences)):
            self.assertTrue(str(idx) in saved_activations)

        # Check saved sentences
        self.assertEqual(len(saved_activations["sentence_to_index"]), 1)
        sentence_to_index = json.loads(saved_activations["sentence_to_index"][0])
        self.assertEqual(len(sentence_to_index), len(self.test_sentences))
        for sentence in sentence_to_index:
            self.assertEqual(
                sentence, self.test_sentences[int(sentence_to_index[sentence])]
            )

        # Check saved activations
        for sentence in sentence_to_index:
            idx = sentence_to_index[sentence]
            self.assertTrue(
                torch.equal(
                    torch.FloatTensor(saved_activations[idx]),
                    self.expected_activations[int(idx)],
                )
            )

    @patch(
        "neurox.data.extraction.transformers_extractor.extract_sentence_representations"
    )
    @patch("neurox.data.extraction.transformers_extractor.get_model_and_tokenizer")
    def test_save_json(self, get_model_mock, extraction_mock):
        "Saving activations in single json file"
        get_model_mock.return_value = (None, MockTokenizer())
        extraction_mock.side_effect = self.mocked_model

        output_file = os.path.join(self.tmpdir.name, "output.json")

        transformers_extractor.extract_representations(
            "non-existant model", self.input_file, output_file, output_type="json"
        )

        with open(output_file) as fp:
            saved_activations = []
            for line in fp:
                saved_activations.append(json.loads(line))

        # Check json structure
        self.assertEqual(len(saved_activations), len(self.test_sentences))

        for representation in saved_activations:
            self.assertIn("linex_index", representation)
            self.assertIn("features", representation)

        # Check sentences and activations
        for idx, representation in enumerate(saved_activations):
            tokens = self.test_sentences[idx].split(" ")
            self.assertEqual(len(representation["features"]), len(tokens))
            for token_idx, token_repr in enumerate(representation["features"]):
                self.assertEqual(token_repr["token"], tokens[token_idx])
                self.assertEqual(len(token_repr["layers"]), 13)
                for layer_idx in range(13):
                    # Using allclose instead of equals since json is a lossy format
                    self.assertTrue(
                        torch.allclose(
                            torch.Tensor(token_repr["layers"][layer_idx]["values"]),
                            self.expected_activations[idx][layer_idx, token_idx, :],
                        )
                    )

    @patch(
        "neurox.data.extraction.transformers_extractor.extract_sentence_representations"
    )
    @patch("neurox.data.extraction.transformers_extractor.get_model_and_tokenizer")
    def test_save_decomposed(self, get_model_mock, extraction_mock):
        "Saving activations in multiple files, one per layer"
        get_model_mock.return_value = (None, MockTokenizer())
        extraction_mock.side_effect = self.mocked_model

        base_output_file = os.path.join(self.tmpdir.name, "output.hdf5")
        output_files = [
            os.path.join(self.tmpdir.name, f"output-layer{layer_idx}.hdf5")
            for layer_idx in range(13)
        ]

        transformers_extractor.extract_representations(
            "non-existant model",
            self.input_file,
            base_output_file,
            decompose_layers=True,
        )

        for layer_idx, output_file in enumerate(output_files):
            saved_activations = h5py.File(output_file, "r")
            sentence_to_index = json.loads(saved_activations["sentence_to_index"][0])

            # Check saved activations
            for sentence in self.test_sentences:
                idx = sentence_to_index[sentence]
                self.assertTrue(
                    torch.equal(
                        torch.FloatTensor(saved_activations[idx]),
                        self.expected_activations[int(idx)][[layer_idx], :, :],
                    )
                )

    @patch(
        "neurox.data.extraction.transformers_extractor.extract_sentence_representations"
    )
    @patch("neurox.data.extraction.transformers_extractor.get_model_and_tokenizer")
    def test_save_filter_layers(self, get_model_mock, extraction_mock):
        "Saving activations from specific layers"
        get_model_mock.return_value = (None, MockTokenizer())
        extraction_mock.side_effect = self.mocked_model

        output_file = os.path.join(self.tmpdir.name, "output.hdf5")
        filter_layers = [1, 3, 5, 7, 12]

        transformers_extractor.extract_representations(
            "non-existant model",
            self.input_file,
            output_file,
            filter_layers=",".join(map(str, filter_layers)),
        )

        saved_activations = h5py.File(output_file, "r")
        sentence_to_index = json.loads(saved_activations["sentence_to_index"][0])

        # Check saved activations
        for sentence in self.test_sentences:
            idx = sentence_to_index[sentence]
            self.assertTrue(
                torch.equal(
                    torch.FloatTensor(saved_activations[idx]),
                    self.expected_activations[int(idx)][filter_layers, :, :],
                )
            )

    @patch(
        "neurox.data.extraction.transformers_extractor.extract_sentence_representations"
    )
    @patch("neurox.data.extraction.transformers_extractor.get_model_and_tokenizer")
    def test_save_decomposed_filter_layers(self, get_model_mock, extraction_mock):
        "Saving activations in multiple files, for specific layers"
        get_model_mock.return_value = (None, MockTokenizer())
        extraction_mock.side_effect = self.mocked_model

        base_output_file = os.path.join(self.tmpdir.name, "output.hdf5")
        filter_layers = [1, 3, 5, 7, 12]

        output_files = [
            os.path.join(self.tmpdir.name, f"output-layer{layer_idx}.hdf5")
            for layer_idx in filter_layers
        ]

        transformers_extractor.extract_representations(
            "non-existant model",
            self.input_file,
            base_output_file,
            decompose_layers=True,
            filter_layers=",".join(map(str, filter_layers)),
        )

        for layer_idx, output_file in zip(filter_layers, output_files):
            saved_activations = h5py.File(output_file, "r")
            sentence_to_index = json.loads(saved_activations["sentence_to_index"][0])

            # Check saved activations
            for sentence in self.test_sentences:
                idx = sentence_to_index[sentence]
                self.assertTrue(
                    torch.equal(
                        torch.FloatTensor(saved_activations[idx]),
                        self.expected_activations[int(idx)][[layer_idx], :, :],
                    )
                )


if __name__ == "__main__":
    unittest.main()
