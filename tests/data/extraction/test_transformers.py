import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

import neurox.data.extraction.transformers as transformers_extractor


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
    @patch('transformers.tokenization_bert.BertTokenizer')
    @patch('transformers.modeling_bert.BertModel')
    def setUpClass(cls, model_mock, tokenizer_mock):
        cls.num_layers = 13
        cls.num_neurons_per_layer = 768

        # Input format is "TOKEN_{num_subwords}"
        # UNK is used when num_subwords == 0
        sentences = [
            ( "Single token sentence without any subwords", ["TOKEN_1"] ),
            ( "Multi token sentence without any subwords", ["TOKEN_1", "TOKEN_1"] ),
            ( "Subword token in the beginning", ["SUBTOKEN_2", "TOKEN_1"] ),
            ( "Subword token in the middle", ["STOKEN_1", "SUBTOKEN_2", "ETOKEN_1"] ),
            ( "Subword token in the end", ["TOKEN_1", "SUBTOKEN_2"] ),
            ( "Multiple subword tokens", ["SOMETHING_2", "TOKEN_4"] ),
            ( "Multiple subword tokens with unknown token", ["TOKEN_2", "TOKEN_2", "TOKEN_0"] ),
            ( "All unknown tokens", ["SOMETHING_0", "SOMETHING2_0", "SOMETHING3_0"] ),
        ]
        cls.tests_data = []

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

        for _, sentence in sentences:
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
                first_expected_output[k].append(tmp[0,:])
                last_expected_output[k].append(tmp[0,:])
                average_expected_output[k].append(tmp[0,:])
            counter += 1
            
            for w in sentence:
                idx.append(counter)
                subwords = word_to_subwords(w)
                counter += len(subwords)
                tokenized_sentence.extend(subwords)

                for k in model_mock_output:
                    tmp = torch.rand((len(subwords), cls.num_neurons_per_layer))
                    model_mock_output[k].append(tmp)
                    first_expected_output[k].append(tmp[0,:]) # Pick first subword idx
                    last_expected_output[k].append(tmp[-1,:]) # Pick last subword idx
                    average_expected_output[k].append(tmp.mean(axis=0))

            idx.append(counter)    
            tokenized_sentence.append("[SEP]")
            for k in model_mock_output:
                tmp = torch.rand((1, cls.num_neurons_per_layer))
                model_mock_output[k].append(tmp)
                first_expected_output[k].append(tmp[0,:])
                last_expected_output[k].append(tmp[0,:])
                average_expected_output[k].append(tmp[0,:])
            
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
            np.testing.assert_array_almost_equal(hidden_states[l,:,:], expected_output[l][1:-1, :].numpy())

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

if __name__ == "__main__":
    unittest.main()
