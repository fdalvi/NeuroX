import unittest

from io import StringIO
from unittest.mock import MagicMock, patch

import numpy as np
import torch

import neurox.data.extraction.transformers_extractor as transformers_extractor


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
    @patch('transformers.BertTokenizer')
    @patch('transformers.BertModel')
    def setUpClass(cls, model_mock, tokenizer_mock):
        cls.num_layers = 13
        cls.num_neurons_per_layer = 768

        # Input format is "TOKEN_{num_subwords}"
        # UNK is used when num_subwords == 0
        # Token is dropped completely when num_subwords < 0
        sentences = [
            ( "Single token sentence without any subwords", ["TOKEN_1"] ),
            ( "Multi token sentence without any subwords", ["TOKEN_1", "TOKEN_1"] ),
            ( "Subword token in the beginning", ["SUBTOKEN_2", "TOKEN_1"] ),
            ( "Subword token in the middle", ["STOKEN_1", "SUBTOKEN_2", "ETOKEN_1"] ),
            ( "Subword token in the end", ["TOKEN_1", "SUBTOKEN_2"] ),
            ( "Multiple subword tokens", ["SOMETHING_2", "TOKEN_4"] ),
            ( "Multiple subword tokens with unknown token", ["TOKEN_2", "TOKEN_2", "TOKEN_0"] ),
            ( "All unknown tokens", ["SOMETHING_0", "SOMETHING2_0", "SOMETHING3_0"] ),
            ( "Special token that is dropped by tokenizer", ["DISAPPEAR_-1"] ),
            ( "Special token in the beginning that is dropped by tokenizer in context", ["DISAPPEAR_-1", "SOMETHING_2"] ),
            ( "Special token in the middle that is dropped by tokenizer in context", ["SOMETHING_2", "DISAPPEAR_-1", "ANOTHER_4"] ),
            ( "Special token in the end that is dropped by tokenizer in context", ["SOMETHING_3", "DISAPPEAR_-1"] ),
            ( "Input longer than tokenizer's limit", ["SOMETHING_100", "ANOTHER_300", "MIDDLE_110", "FINAL_100"] ),
            ( "Input exactly equal to tokenizer's limit", ["SOMETHING_100", "ANOTHER_300", "FINAL_110"] ),
            ( "Input longer than tokenizer's limit with break in the middle of tokenization", ["SOMETHING_100", "ANOTHER_300", "FINAL_200"] ),
            ( "Input exactly equal to tokenizer's limit with dropped token", ["SOMETHING_100", "ANOTHER_300", "FINAL_110", "DISAPPEAR_-1"] ),
            ( "Input longer than tokenizer's limit with break at dropped token", ["SOMETHING_100", "ANOTHER_300", "MIDDLE_110", "DISAPPEAR_-1", "FINAL_100"] ),
            ( "Input longer than tokenizer's limit with dropped token", ["SOMETHING_100", "DISAPPEAR_-1", "ANOTHER_300", "MIDDLE_110", "FINAL_100"] ),

        ]
        cls.tests_data = []

        # Create mock model and tokenizer
        model_mock = MagicMock()
        tokenizer_mock = MagicMock()
        tokenizer_mock.all_special_tokens = ["[CLS]", "[UNK]", "[SEP]"]
        tokenizer_mock.unk_token = "[UNK]"
        tokenizer_mock.model_max_length = 512

        tokenization_mapping = {k: i for i, k in enumerate(tokenizer_mock.all_special_tokens)}
        tokenization_mapping['a'] = len(tokenization_mapping)

        def word_to_subwords(word):
            if '_' not in word:
                return [word]
            actual_word, occurrence = word.split('_')
            occurrence = int(occurrence)
            if occurrence == 0:
                return [tokenizer_mock.unk_token]
            elif occurrence < 0:
                return []
            else:
                subwords = []
                for o_idx in range(occurrence):
                    if o_idx == 0:
                        subwords.append(f'{actual_word}_{o_idx+1}')
                    else:
                        subwords.append(f'@@{actual_word}_{o_idx+1}')
                return subwords

        for _, sentence in sentences:
            for word in sentence:
                for subword in word_to_subwords(word):
                    if subword not in tokenization_mapping:
                        tokenization_mapping[subword] = len(tokenization_mapping)

        inverse_tokenization_mapping = {v: k for k,v in tokenization_mapping.items()}

        def tokenization_side_effect(arg):
            # Truncate input at 512 tokens
            arg = arg[:512]

            return [tokenization_mapping[w] for w in arg]
        tokenizer_mock.convert_tokens_to_ids.side_effect = tokenization_side_effect

        def inverse_tokenization_side_effect(arg):
            # Truncate input at 512 tokens
            arg = arg[:512]

            return [inverse_tokenization_mapping[i] for i in arg]
        tokenizer_mock.convert_ids_to_tokens.side_effect = inverse_tokenization_side_effect

        def encode_side_effect(arg, **kwargs):
            tokenized_sentence = ["[CLS]"]
            for w in arg.split(' '):
                tokenized_sentence.extend(word_to_subwords(w))
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

            idx = [counter]
            tokenized_sentence = ["[CLS]"]
            for k in model_mock_output:
                tmp = torch.rand((1, cls.num_neurons_per_layer))
                model_mock_output[k].append(tmp)

            counter += 1

            for w in sentence:
                idx.append(counter)
                subwords = word_to_subwords(w)
                counter += len(subwords)
                tokenized_sentence.extend(subwords)

                for k in model_mock_output:
                    num_subtokens = len(subwords)
                    # Account for truncation in the middle of tokenization
                    num_extra_tokens = len(tokenized_sentence) - (tokenizer_mock.model_max_length - 1)
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
                    first_expected_output[k].append(tmp[0,:]) # Pick first subword idx
                    last_expected_output[k].append(tmp[-1,:]) # Pick last subword idx
                    average_expected_output[k].append(tmp.mean(axis=0))

                # Check if input is too long already (and account
                # for [SEP] token)
                if len(tokenized_sentence) > (tokenizer_mock.model_max_length - 1):
                    break

            idx.append(counter)
            tokenized_sentence.append("[SEP]")
            for k in model_mock_output:
                tmp = torch.rand((1, cls.num_neurons_per_layer))
                model_mock_output[k].append(tmp)

            counter += 1

            model_mock_output = tuple([torch.cat(model_mock_output[k]).unsqueeze(0) for k in range(cls.num_layers)])
            first_expected_output = tuple([torch.stack(first_expected_output[k]) for k in range(cls.num_layers)])
            last_expected_output = tuple([torch.stack(last_expected_output[k]) for k in range(cls.num_layers)])
            average_expected_output = tuple([torch.stack(average_expected_output[k]) for k in range(cls.num_layers)])

            cls.tests_data.append((
                test_description,
                sentence,
                model_mock_output,
                first_expected_output,
                last_expected_output,
                average_expected_output
            ))

        cls.model = model_mock
        cls.tokenizer = tokenizer_mock


    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def run_test(self, testcase, **kwargs):
        _, sentence, model_mock_output, first_expected_output, last_expected_output, average_expected_output = testcase
        if kwargs['aggregation'] == "first":
            expected_output = first_expected_output
        if kwargs['aggregation'] == "last":
            expected_output = last_expected_output
        if kwargs['aggregation'] == "average":
            expected_output = average_expected_output
        self.model.return_value = ("placeholder", model_mock_output)

        words = sentence
        hidden_states, extracted_words = transformers_extractor.extract_sentence_representations(" ".join(words), self.model, self.tokenizer, **kwargs)
        self.assertEqual(
            len(extracted_words), len(words)
        )
        self.assertEqual(
            hidden_states.shape[1], len(words)
        )

        # Test output from all layers
        for l in range(self.num_layers):
            np.testing.assert_array_almost_equal(hidden_states[l,:,:], expected_output[l][:, :].numpy())

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

    def test_extract_sentence_representations_first_aggregation_mutliple_subwords_with_unk(self):
        "First aggregation: Multiple subword tokens with unknown token"
        self.run_test(self.tests_data[6], aggregation="first")

    def test_extract_sentence_representations_first_aggregation_all_unk(self):
        "First aggregation: All unknown tokens"
        self.run_test(self.tests_data[7], aggregation="first")

    def test_extract_sentence_representations_first_aggregation_special_dropped_token(self):
        "First aggregation: Special token that is dropped by tokenizer"
        self.run_test(self.tests_data[8], aggregation="first")

    def test_extract_sentence_representations_first_aggregation_special_dropped_token_beginning(self):
        "First aggregation: Special token in the beginning that is dropped by tokenizer in context"
        self.run_test(self.tests_data[9], aggregation="first")

    def test_extract_sentence_representations_first_aggregation_special_dropped_token_middle(self):
        "First aggregation: Special token in the middle that is dropped by tokenizer in context"
        self.run_test(self.tests_data[10], aggregation="first")

    def test_extract_sentence_representations_first_aggregation_special_dropped_token_end(self):
        "First aggregation: Special token in the end that is dropped by tokenizer in context"
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

    def test_extract_sentence_representations_last_aggregation_mutliple_subwords_with_unk(self):
        "Last aggregation: Multiple subword tokens with unknown token"
        self.run_test(self.tests_data[6], aggregation="last")

    def test_extract_sentence_representations_last_aggregation_all_unk(self):
        "Last aggregation: All unknown tokens"
        self.run_test(self.tests_data[7], aggregation="last")

    def test_extract_sentence_representations_last_aggregation_special_dropped_token(self):
        "Last aggregation: Special token that is dropped by tokenizer"
        self.run_test(self.tests_data[8], aggregation="last")

    def test_extract_sentence_representations_last_aggregation_special_dropped_token_beginning(self):
        "Last aggregation: Special token in the beginning that is dropped by tokenizer in context"
        self.run_test(self.tests_data[9], aggregation="last")

    def test_extract_sentence_representations_last_aggregation_special_dropped_token_middle(self):
        "Last aggregation: Special token in the middle that is dropped by tokenizer in context"
        self.run_test(self.tests_data[10], aggregation="last")

    def test_extract_sentence_representations_last_aggregation_special_dropped_token_end(self):
        "Last aggregation: Special token in the end that is dropped by tokenizer in context"
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

    def test_extract_sentence_representations_average_aggregation_mutliple_subwords(self):
        "Average aggregation: Multiple subword tokens"
        self.run_test(self.tests_data[5], aggregation="average")

    def test_extract_sentence_representations_average_aggregation_mutliple_subwords_with_unk(self):
        "Average aggregation: Multiple subword tokens with unknown token"
        self.run_test(self.tests_data[6], aggregation="average")

    def test_extract_sentence_representations_average_aggregation_all_unk(self):
        "Average aggregation: All unknown tokens"
        self.run_test(self.tests_data[7], aggregation="average")

    def test_extract_sentence_representations_average_aggregation_special_dropped_token(self):
        "Average aggregation: Special token that is dropped by tokenizer"
        self.run_test(self.tests_data[8], aggregation="average")

    def test_extract_sentence_representations_average_aggregation_special_dropped_token_beginning(self):
        "Average aggregation: Special token in the beginning that is dropped by tokenizer in context"
        self.run_test(self.tests_data[9], aggregation="average")

    def test_extract_sentence_representations_average_aggregation_special_dropped_token_middle(self):
        "Average aggregation: Special token in the middle that is dropped by tokenizer in context"
        self.run_test(self.tests_data[10], aggregation="average")

    def test_extract_sentence_representations_average_aggregation_special_dropped_token_end(self):
        "Average aggregation: Special token in the end that is dropped by tokenizer in context"
        self.run_test(self.tests_data[11], aggregation="average")

    ############################# Embedding tests ##############################
    def test_extract_sentence_representations_include_embeddings(self):
        "Extraction with embedding layer"
        _, sentence, model_mock_output, _, expected_output, _ = self.tests_data[1]
        self.model.return_value = ("placeholder", model_mock_output)

        hidden_states, extracted_words = transformers_extractor.extract_sentence_representations(" ".join(sentence), self.model, self.tokenizer, include_embeddings=True)

        self.assertEqual(hidden_states.shape[0], self.num_layers)

        for l in range(self.num_layers):
            np.testing.assert_array_almost_equal(hidden_states[l,:,:], expected_output[l][:, :].numpy())

    def test_extract_sentence_representations_exclude_embeddings(self):
        "Extraction without embedding layer"
        _, sentence, model_mock_output, _, expected_output , _  = self.tests_data[1]
        self.model.return_value = ("placeholder", model_mock_output)

        hidden_states, extracted_words = transformers_extractor.extract_sentence_representations(" ".join(sentence), self.model, self.tokenizer, include_embeddings=False)

        self.assertEqual(hidden_states.shape[0], self.num_layers - 1)

        for l in range(1, self.num_layers):
            np.testing.assert_array_almost_equal(hidden_states[l-1,:,:], expected_output[l][:, :].numpy())

    @patch('sys.stdout', new_callable=StringIO)
    def test_extract_sentence_representations_long_input(self, mock_stdout):
        "Input longer than tokenizer's limit"
        _, sentence, model_mock_output, _, expected_output , _  = self.tests_data[12]
        self.model.return_value = ("placeholder", model_mock_output)

        hidden_states, extracted_words = transformers_extractor.extract_sentence_representations(" ".join(sentence), self.model, self.tokenizer)

        self.assertIn("Input truncated because of length", mock_stdout.getvalue())

        for l in range(1, self.num_layers):
            np.testing.assert_array_almost_equal(hidden_states[l,:,:], expected_output[l][:, :].numpy())

    def test_extract_sentence_representations_long_input_exact_length(self):
        "Input exactly equal to tokenizer's limit"
        _, sentence, model_mock_output, _, expected_output , _  = self.tests_data[13]
        self.model.return_value = ("placeholder", model_mock_output)

        hidden_states, extracted_words = transformers_extractor.extract_sentence_representations(" ".join(sentence), self.model, self.tokenizer)

        # self.assertIn("Input truncated because of length", mock_stdout.getvalue())

        for l in range(1, self.num_layers):
            np.testing.assert_array_almost_equal(hidden_states[l,:,:], expected_output[l][:, :].numpy())

    @patch('sys.stdout', new_callable=StringIO)
    def test_extract_sentence_representations_long_input_tokenization_break(self, mock_stdout):
        "Input longer than tokenizer's limit with break in the middle of tokenization"
        _, sentence, model_mock_output, _, expected_output , _  = self.tests_data[14]
        self.model.return_value = ("placeholder", model_mock_output)

        hidden_states, extracted_words = transformers_extractor.extract_sentence_representations(" ".join(sentence), self.model, self.tokenizer)

        self.assertIn("Input truncated because of length", mock_stdout.getvalue())

        for l in range(1, self.num_layers):
            np.testing.assert_array_almost_equal(hidden_states[l,:,:], expected_output[l][:, :].numpy())

    def test_extract_sentence_representations_long_input_exact_length_dropped_token(self):
        "Input exactly equal to tokenizer's limit with dropped token"
        _, sentence, model_mock_output, _, expected_output , _  = self.tests_data[15]
        self.model.return_value = ("placeholder", model_mock_output)

        hidden_states, extracted_words = transformers_extractor.extract_sentence_representations(" ".join(sentence), self.model, self.tokenizer)

        # self.assertIn("Input truncated because of length", mock_stdout.getvalue())

        for l in range(1, self.num_layers):
            np.testing.assert_array_almost_equal(hidden_states[l,:,:], expected_output[l][:, :].numpy())

    @patch('sys.stdout', new_callable=StringIO)
    def test_extract_sentence_representations_long_input_dropped_token_break(self, mock_stdout):
        "Input longer than tokenizer's limit with break at dropped token"
        _, sentence, model_mock_output, _, expected_output , _  = self.tests_data[16]
        self.model.return_value = ("placeholder", model_mock_output)

        hidden_states, extracted_words = transformers_extractor.extract_sentence_representations(" ".join(sentence), self.model, self.tokenizer)

        self.assertIn("Input truncated because of length", mock_stdout.getvalue())

        for l in range(1, self.num_layers):
            np.testing.assert_array_almost_equal(hidden_states[l,:,:], expected_output[l][:, :].numpy())

    @patch('sys.stdout', new_callable=StringIO)
    def test_extract_sentence_representations_long_input_dropped_token(self, mock_stdout):
        "Input longer than tokenizer's limit with dropped token"
        _, sentence, model_mock_output, _, expected_output , _  = self.tests_data[17]
        self.model.return_value = ("placeholder", model_mock_output)

        hidden_states, extracted_words = transformers_extractor.extract_sentence_representations(" ".join(sentence), self.model, self.tokenizer)

        self.assertIn("Input truncated because of length", mock_stdout.getvalue())

        for l in range(1, self.num_layers):
            np.testing.assert_array_almost_equal(hidden_states[l,:,:], expected_output[l][:, :].numpy())


class TestModelAndTokenizerGetter(unittest.TestCase):
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModel.from_pretrained')
    def test_get_model_and_tokenizer_normal(self, auto_model_mock, auto_tokenizer_mock):
        """Normal model and tokenizer loading test"""

        expected_model = torch.rand((5,2))
        expected_tokenizer = torch.rand((5,))
        auto_model_mock.return_value = expected_model
        auto_tokenizer_mock.return_value = expected_tokenizer

        model, tokenizer = transformers_extractor.get_model_and_tokenizer("non-existent model")

        self.assertTrue(torch.equal(model, expected_model))
        self.assertTrue(torch.equal(tokenizer, expected_tokenizer))

    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModel.from_pretrained')
    def test_get_model_and_tokenizer_simple(self, auto_model_mock, auto_tokenizer_mock):
        """Randomized model and tokenizer loading test"""

        transformers_extractor.get_model_and_tokenizer("non-existent model", random_weights=True)

        auto_model_mock.return_value.to.return_value.init_weights.assert_called_once()

if __name__ == "__main__":
    unittest.main()
