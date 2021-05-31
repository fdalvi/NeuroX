import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch
import random

import neurox.data.extraction.transformers as transformers_extractor


class TestTransformersExtractorAggregation(unittest.TestCase):
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

class TestTransformersExtractorExtraction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.num_layers = 13
        cls.num_neurons_per_layer = 768

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    @patch('transformers.tokenization_bert.BertTokenizer')
    @patch('transformers.modeling_bert.BertModel')
    def test_extract_sentence_representations(self, model_mock, tokenizer_mock):
        # Input format is "TOKEN_{num_subwords}"
        # UNK is used when num_subwords == 0
        sentences = [
            ["TOKEN_1"], # Single token sentence without any subwords
            ["TOKEN_1", "TOKEN_1"], # Multi token sentence without any subwords
            ["SUBTOKEN_2", "TOKEN_1"], # Subword token in the beginning
            ["STOKEN_1", "SUBTOKEN_2", "ETOKEN_1"], # Subword token in the middle
            ["TOKEN_1", "SUBTOKEN_2"], # Subword token in the middle
            ["SOMETHING_2", "TOKEN_4"], # Multiple subword tokens with unknown token
            ["TOKEN_2", "TOKEN_2", "TOKEN_0"], # Multiple subword tokens with unknown token
            ["SOMETHING_0", "SOMETHING2_0", "SOMETHING3_0"], # All unks
        ]

        # Create mock model and tokenizer
        model_mock = MagicMock()
        tokenizer_mock = MagicMock()
        tokenizer_mock.all_special_tokens = ["[CLS]", "[UNK]", "[SEP]"]
        tokenizer_mock.unk_token = "[UNK]"

        tokenization_mapping = {k: i for i, k in enumerate(tokenizer_mock.all_special_tokens)}
        tokenization_mapping['a'] = len(tokenization_mapping)

        def word_to_subwords(word):
            if '_' not in word:
                return [word]
            actual_word, occurrence = word.split('_')
            occurrence = int(occurrence)
            if occurrence == 0:
                return [tokenizer_mock.unk_token]
            else:
                subwords = []
                for o_idx in range(occurrence):
                    if o_idx == 0:
                        subwords.append(f'{actual_word}_{o_idx+1}')
                    else:
                        subwords.append(f'@@{actual_word}_{o_idx+1}')
                return subwords

        for sentence in sentences:
            for word in sentence:
                for subword in word_to_subwords(word):
                    if subword not in tokenization_mapping:
                        tokenization_mapping[subword] = len(tokenization_mapping)
                
        inverse_tokenization_mapping = {v: k for k,v in tokenization_mapping.items()}
        
        def tokenization_side_effect(arg):
            return [tokenization_mapping[w] for w in arg]
        tokenizer_mock.convert_tokens_to_ids.side_effect = tokenization_side_effect

        def inverse_tokenization_side_effect(arg):
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
        model_mock_outputs = []
        expected_outputs = []
        for sentence in sentences:
            counter = 0
            model_mock_output = {k: [] for k in range(self.num_layers)}
            expected_output = {k: [] for k in range(self.num_layers)}
            idx = [counter]
            tokenized_sentence = ["[CLS]"]
            for k in model_mock_output:
                tmp = torch.rand((1, self.num_neurons_per_layer))
                model_mock_output[k].append(tmp)
                expected_output[k].append(tmp[0,:])
            counter += 1
            
            for w in sentence:
                idx.append(counter)
                subwords = word_to_subwords(w)
                counter += len(subwords)
                tokenized_sentence.extend(subwords)

                for k in model_mock_output:
                    tmp = torch.rand((len(subwords), self.num_neurons_per_layer))
                    model_mock_output[k].append(tmp)
                    expected_output[k].append(tmp.mean(axis=0))

            idx.append(counter)    
            tokenized_sentence.append("[SEP]")
            for k in model_mock_output:
                tmp = torch.rand((1, self.num_neurons_per_layer))
                model_mock_output[k].append(tmp)
                expected_output[k].append(tmp[0,:])
            
            counter += 1

            model_mock_output = tuple([torch.cat(model_mock_output[k]).unsqueeze(0) for k in range(self.num_layers)])
            expected_output = tuple([torch.stack(expected_output[k]) for k in range(self.num_layers)])
            print(expected_output[0].shape)

            model_mock_outputs.append(model_mock_output)
            expected_outputs.append(expected_output)

        for sentence_idx, sentence in enumerate(sentences):
            with self.subTest(i=sentence_idx):
                model_mock.return_value = ("placeholder", model_mock_outputs[sentence_idx])

                words = sentence
                hidden_states, extracted_words = transformers_extractor.extract_sentence_representations(" ".join(words), model_mock, tokenizer_mock, aggregation="average")
                self.assertEqual(
                    len(extracted_words), len(words)
                )
                self.assertEqual(
                    hidden_states.shape[1], len(words)
                )

                print(hidden_states.shape)
                # Test output from all layers
                for l in range(self.num_layers):
                    np.testing.assert_array_almost_equal(hidden_states[l,:,:], expected_outputs[sentence_idx][l][1:-1, :].numpy())


if __name__ == "__main__":
    unittest.main()
